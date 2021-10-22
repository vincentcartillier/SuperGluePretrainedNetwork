import os
import logging
import datetime


def get_logger(logdir):
    logger = logging.getLogger("smnet")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def convert_weights_cuda_cpu(weights, device):
    names = list(weights.keys())
    is_module = names[0].split('.')[0] == 'module'
    if device == 'cuda' and not is_module:
        new_weights = {'module.'+k:v for k,v in weights.items()}
    elif device == 'cpu' and is_module:
        new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items()}
    else:
        new_weights = weights
    return new_weights


