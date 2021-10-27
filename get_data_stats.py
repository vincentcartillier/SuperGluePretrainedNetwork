import open3d as o3d
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from dataloader import DescriptorsEpisode


with open('settings.yml', 'r') as fp:
    cfg = yaml.safe_load(fp)



t_loader = DescriptorsEpisode(cfg["data"], 'train')
trainloader = torch.utils.data.DataLoader(
    t_loader,
    batch_size=1,
    num_workers=1,
    drop_last=False,
    collate_fn=t_loader.collate,
)


overall_pos_sum = []
overall_neg_sum = []
overall_matches_score = []
overall_nomatches_score = []
overall_objects = []
overall_objects_wuid = []
for batch in tqdm(trainloader):
    # get Object maps stats
    obm_A_num_detection = batch['obm_A_num_detection'][0]
    obm_B_num_detection = batch['obm_B_num_detection'][0]
    overall_objects.append(obm_A_num_detection + obm_B_num_detection)
    
    obm_A_num_detection_wuid = batch['obm_A_num_detection_wuid'][0]
    obm_B_num_detection_wuid = batch['obm_B_num_detection_wuid'][0]
    overall_objects_wuid.append(obm_A_num_detection_wuid + obm_B_num_detection_wuid)
    
    # get scores (dot product)
    desc0 = batch['descriptors0'][0]
    desc1 = batch['descriptors1'][0]
    desc0 = desc0.unsqueeze(0)
    desc1 = desc1.unsqueeze(0)

    descriptor_dim = desc0.shape[1]

    scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
    scores = scores / descriptor_dim**.5
    scores = scores[0].numpy()
    
    # get GT matches
    gt_matches = batch['gt_matches'][0]
    gt_matches_matrix = batch['gt_matches_matrix'][0]

    # ratio of positive matches VS negative matches
    pos_sum = np.sum(gt_matches_matrix[:-1,:-1])
    neg_sum = np.sum(gt_matches_matrix[:,-1]) +\
              np.sum(gt_matches_matrix[-1,:])
    
    overall_pos_sum.append(pos_sum)
    overall_neg_sum.append(neg_sum)
    
    # get matches scores
    mask = gt_matches_matrix[:-1,:-1]
    mask = mask.astype(np.bool)
    pos_scores = scores[mask]
    neg_scores = scores[~mask]
    
    overall_matches_score.append(np.mean(pos_scores))
    overall_nomatches_score.append(np.mean(neg_scores))

pos_neg_ratio = np.array(overall_pos_sum) / np.array(overall_neg_sum)

print('AVG number of detections: ', np.mean(overall_objects))
print('MAX number of detections: ', np.max(overall_objects))
print(' ')
print('AVG number of detections with uids: ', np.mean(overall_objects_wuid))
print('MAX number of detections with uids: ', np.max(overall_objects_wuid))
print(' ')
print('AVG pos/neg ratio: ', np.mean(pos_neg_ratio))
print('STD pos/neg ratio: ', np.std(pos_neg_ratio))
print('MIN pos/neg ratio: ', np.min(pos_neg_ratio))
print('MAX pos/neg ratio: ', np.max(pos_neg_ratio))
print(' ')
print('AVG num of pos match per ep: ', np.mean(overall_pos_sum))
print('AVG num of neg match per ep: ', np.mean(overall_neg_sum))
print(' ')
print('AVG score for match: ', np.mean(overall_matches_score))
print('MIN score for match: ', np.min(overall_matches_score))
print('MAX score for match: ', np.max(overall_matches_score))
print(' ')
print('AVG score for non-match: ', np.mean(overall_nomatches_score))
print('MIN score for non-match: ', np.min(overall_nomatches_score))
print('MAX score for non-match: ', np.max(overall_nomatches_score))

