import os
import sys
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

sys.path.append('/nethome/vcartillier3/3D-SMNet/3D-SMNet-API/')
from object_map import ObjectMap

sys.path.append('/nethome/vcartillier3/3D-SMNet/metrics/')
from matching_accuracy import MatchingAccuracy




class DescriptorsEpisode(Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.episode_dir = cfg[f'{split}_episode_dir']
        self.obm_dir = cfg[f'{split}_obm_dir']

        skip_ep = json.load(open(cfg['skip_ep']))
        
        self.episodes = os.listdir(self.episode_dir)
        self.episodes = [x for x in self.episodes\
                         if (int(x.split('.')[0]) not in skip_ep['skip_ep'][split])]

        self.helper = MatchingAccuracy()
    
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        episode_name = self.episodes[item]
        ep_uid = episode_name.split('.')[0]
        ep_uid = int(ep_uid)

        obm_A_filename = os.path.join(self.obm_dir, f"{ep_uid}_A.json")
        obm_A = ObjectMap()
        obm_A.load(obm_A_filename)

        desc0 = []
        for object in obm_A:
            desc0.append(object.descriptor)
        desc0 = np.array(desc0)
        desc0 = torch.FloatTensor(desc0)
        
        obm_B_filename = os.path.join(self.obm_dir, f"{ep_uid}_B.json")
        obm_B = ObjectMap()
        obm_B.load(obm_B_filename)

        desc1 = []
        for object in obm_B:
            desc1.append(object.descriptor)
        desc1 = np.array(desc1)
        desc1 = torch.FloatTensor(desc1)

        obj_uid_A = obm_A.get_object_uids()
        obj_uid_B = obm_B.get_object_uids()
        gt_matches_matrix = self.helper.build_GT_matches_matrix(obj_uid_A,
                                                                obj_uid_B)
        gt_matches = np.nonzero(gt_matches_matrix)
        gt_matches_A = gt_matches[0] #rows
        gt_matches_B = gt_matches[1] #cols

        gt_matches = np.hstack((gt_matches_A[:,np.newaxis], 
                                gt_matches_B[:,np.newaxis]))

        data = {'descriptors0': desc0.T,
                'descriptors1': desc1.T,
                'gt_matches': gt_matches,}

        return data

    def collate(self, batch):
        descriptors0 = [item['descriptors0'] for item in batch]
        descriptors1 = [item['descriptors1'] for item in batch]
        gt_matches = [item['gt_matches'] for item in batch]
        return {'descriptors0':descriptors0,
                'descriptors1':descriptors1,
                'gt_matches':gt_matches,}






