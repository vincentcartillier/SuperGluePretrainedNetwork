import os
import sys
import open3d as o3d
import yaml
import time
import json
import torch
import shutil
import argparse
import random
import copy
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as distrib
from pathlib import Path
from torch.utils import data
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler

from models.superglue import Sinkhorn
from models.superglue import SuperGlue3DSMNet as SuperGlue
from models.superglue import SuperGlue3DSMNet_plusplus as SuperGlue_plusplus
from models.superglue_3dsmnet import Sinkhorn_wGNN
from models.superglue_3dsmnet import Sinkhorn_wZ
from models.superglue_3dsmnet import Sinkhorn_wZatt
from models.superglue_3dsmnet import Sinkhorn_wZatt_Big
from dataloader import DescriptorsEpisode
from matcher_training_utils import get_logger
from metrics import averageMeter
from loss import SuperGlueLoss


def train(rank, world_size, cfg):
    # get eval config
    eval_cfg = cfg['eval_cfg']

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # init distributed compute
    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    tcp_store = torch.distributed.TCPStore(
        master_addr, master_port, world_size, rank==0
    )
    torch.distributed.init_process_group(
        'nccl', store=tcp_store, rank=rank, world_size=world_size
    )

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        assert world_size == 1
        device = torch.device("cpu")

    if rank == 0:
        writer = SummaryWriter(logdir=cfg["logdir"])
        logger = get_logger(cfg["logdir"])
        logger.info("Let the training begin !!")

    # Setup Dataloader
    t_loader = DescriptorsEpisode(cfg["data"], 'train')
    v_loader = DescriptorsEpisode(cfg["data"], 'val')
    t_sampler = DistributedSampler(t_loader)
    v_sampler = DistributedSampler(v_loader)

    if rank == 0:
        print('#Envs in train: %d' % (len(t_loader)))

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"] // world_size,
        num_workers=cfg["training"]["n_workers"],
        drop_last=True,
        pin_memory=True,
        sampler=t_sampler,
        multiprocessing_context='fork',
        collate_fn=t_loader.collate
    )
    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training"]["batch_size"] // world_size,
        num_workers=cfg["training"]["n_workers"],
        pin_memory=True,
        sampler=v_sampler,
        multiprocessing_context='fork',
        collate_fn=v_loader.collate
    )

    # Setup Model
    if cfg['model']['arch'] == 'sinkhorn':
        model = Sinkhorn(cfg['model'])
    elif cfg['model']['arch'] == 'superglue':
        model = SuperGlue(cfg['model'])
    elif cfg['model']['arch'] == 'superglue++':
        model = SuperGlue_plusplus(cfg['model'])
    elif cfg['model']['arch'] == 'sinkhorn_wgnn':
        model = Sinkhorn_wGNN(cfg['model'])
    elif cfg['model']['arch'] == 'sinkhorn_wz':
        model = Sinkhorn_wZ(cfg['model'])
    elif cfg['model']['arch'] == 'sinkhorn_wzatt':
        model = Sinkhorn_wZatt(cfg['model'])
    elif cfg['model']['arch'] == 'sinkhorn_wzatt_big':
        model = Sinkhorn_wZatt_Big(cfg['model'])
    else:
        raise ValueError()
    model = model.to(device)

    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if rank == 0:
        print('# trainable parameters = ', params)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    if rank == 0:
        logger.info("Using optimizer {}".format(optimizer))

    # setup Loss
    loss_fn = SuperGlueLoss()

    if rank == 0:
        logger.info("Using loss {}".format(loss_fn))

    # init training
    start_iter = 0
    start_epoch = 0

    # init metrics
    time_meter = averageMeter()
    val_loss_meter = averageMeter()
    best_val_loss = 100.0
    best_val_matching_acc = 0.0

    # start training
    iter = start_iter
    for epoch in range(start_epoch, cfg["training"]["train_epoch"], 1):

        t_sampler.set_epoch(epoch)

        for batch in trainloader:

            iter += 1
            start_ts = time.time()

            model.train()

            optimizer.zero_grad()

            result = model(batch)
            scores = result['scores']

            loss = loss_fn.compute(scores, batch['gt_matches'])

            loss.backward()

            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (iter % cfg["training"]["print_interval"] == 0):
                distrib.all_reduce(loss)
                loss /= world_size

                if (rank ==0):
                    fmt_str = "Iter: {:d} == Epoch [{:d}/{:d}] == Loss: {:.4f} == Time/Image: {:.4f} == bin_score: {:.4f}"

                    print_str = fmt_str.format(
                        iter,
                        epoch,
                        cfg["training"]["train_epoch"],
                        loss.item(),
                        time_meter.avg / cfg["training"]["batch_size"],
                        model.module.bin_score.item(),
                    )

                    print(print_str)
                    logger.info(print_str)
                    writer.add_scalar("loss/train_loss", loss.item(), iter)
                    writer.add_scalar("bin_score", model.module.bin_score.item(), iter)
                    time_meter.reset()

        model.eval()
        with torch.no_grad():
            for batch_val in valloader:
                result = model(batch_val)
                scores = result['scores']
                loss_val = loss_fn.compute(scores, batch_val['gt_matches'])
                val_loss_meter.update(loss_val.item())

        val_loss_avg = val_loss_meter.avg
        val_loss_avg = torch.FloatTensor([val_loss_avg])
        val_loss_avg = val_loss_avg.to(device)
        distrib.all_reduce(val_loss_avg)
        val_loss_avg /= world_size


        if rank == 0:
            val_matching_acc = eval(model, eval_cfg)

            val_loss_avg = val_loss_avg.cpu().numpy()
            val_loss_avg = val_loss_avg[0]
            writer.add_scalar("loss/val_loss", val_loss_avg, iter)
            writer.add_scalar("loss/val_acc", val_matching_acc, iter)

            logger.info("Iter %d val Loss: %.4f" % (iter, val_loss_avg))
            logger.info("Iter %d val matching acc: %.4f" % (iter, val_matching_acc))

            if val_matching_acc > best_val_matching_acc:
                best_val_matching_acc = val_matching_acc
                state = {
                    "epoch": epoch,
                    'best_val_loss':best_val_loss,
                    'best_val_matching_acc':best_val_matching_acc,
                    "iter": iter,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_mp3d_best_model.pkl".format(cfg["model"]["arch"]),
                )
                torch.save(state, save_path)

        val_loss_meter.reset()

    if rank == 0:
        writer.close()



def eval(net, cfg):
    sys.path.append('/nethome/vcartillier3/3D-SMNet/3D-SMNet-API/')
    from object_map import ObjectMap

    sys.path.append('/nethome/vcartillier3/3D-SMNet/matchers/')
    from hungarian_matcher_full_dataset import HungarianMatcher

    sys.path.append('/nethome/vcartillier3/3D-SMNet/metrics/')
    from matching_accuracy import MatchingAccuracy

    model = copy.deepcopy(net)
    model.to('cuda')
    model.eval()

    matcher = HungarianMatcher()
    metric_helper = MatchingAccuracy()

    split = cfg['split']
    obm_dir = cfg['dir_obm']
    dir_episodes = cfg['dir_episodes']
    if 'skip_ep' in cfg:
        episodes_to_skip = json.load(open(cfg['skip_ep'], 'r'))
        skip_ep = episodes_to_skip['skip_ep'][split]
    else:
        skip_ep = None

    episodes = os.listdir(dir_episodes)
    num = 0
    total = 0
    with torch.no_grad():
        for episode in episodes:
            uid = episode.split('.')[0]
            uid = int(uid)
            if skip_ep is not None:
                if uid in skip_ep:
                    continue

            # -- get OBM - A
            object_map_filename_A = os.path.join(obm_dir,
                                                 f'{uid}_A.json')
            object_map_A = ObjectMap()
            object_map_A.load(object_map_filename_A)
            if len(object_map_A) ==0: continue

            desc0 = []
            for object in object_map_A:
                desc0.append(object.descriptor)
            desc0 = np.array(desc0)
            desc0 = torch.FloatTensor(desc0)

            # -- get OBM - B
            object_map_filename_B = os.path.join(obm_dir,
                                                 f'{uid}_B.json')
            object_map_B = ObjectMap()
            object_map_B.load(object_map_filename_B)
            if len(object_map_B) ==0: continue
            desc1 = []
            for object in object_map_B:
                desc1.append(object.descriptor)
            desc1 = np.array(desc1)
            desc1 = torch.FloatTensor(desc1)

            batch = {'descriptors0':[desc0.T],
                     'descriptors1':[desc1.T],}
            results = model(batch)
            matches0 = results['matches0'][0][0]
            matches1 = results['matches1'][0][0]

            matches = []
            for r, c in enumerate(matches0):
                if c > -1:
                    matches.append((r,c))
            for c, r in enumerate(matches1):
                if r > -1:
                    matches.append((r,c))
            matches = list(set(matches))
            matches = [[x[0], x[1]] for x in matches]
            matches = np.array(matches)

            out = metric_helper.matching_accuracy(matches,
                                                 object_map_A,
                                                 object_map_B)
            num += out['all']['num']
            total += out['all']['total']

    acc = num / total
    return acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="settings.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    name_expe = cfg['name_experiment']

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", name_expe, str(run_id))
    chkptdir = os.path.join("checkpoints", name_expe, str(run_id))

    cfg['checkpoint_dir'] = chkptdir
    cfg['logdir'] = logdir

    print("RUNDIR: {}".format(logdir))
    Path(logdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, logdir)

    print("CHECKPOINTDIR: {}".format(chkptdir))
    Path(chkptdir).mkdir(parents=True, exist_ok=True)

    world_size=1
    mp.spawn(train,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)
