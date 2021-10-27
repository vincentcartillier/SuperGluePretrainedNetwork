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
        
        nA, nB = gt_matches_matrix.shape
        
        # get indices of positive matches
        gt_matches = np.nonzero(gt_matches_matrix[:-1,:-1])
        gt_matches_A = gt_matches[0] #rows
        gt_matches_B = gt_matches[1] #cols

        # get indices of negative matches
        tmp_gt_neg_matches = np.nonzero(gt_matches_matrix[:,-1])
        gt_neg_matches_A = tmp_gt_neg_matches[0]
        gt_neg_matches_B = np.full((len(gt_neg_matches_A)), nB-1)

        tmp_gt_neg_matches = np.nonzero(gt_matches_matrix[-1,:])
        gt_neg_matches_B = np.hstack((gt_neg_matches_B,
                                     tmp_gt_neg_matches[0]))
        gt_neg_matches_A = np.hstack((gt_neg_matches_A,
                                     np.full((len(tmp_gt_neg_matches[0])),
                                             nA-1)))
        
        # select same number of negative matches
        if len(gt_matches_A) < len(gt_neg_matches_A):
            nn = len(gt_matches_A)
            nn_indices = list(range(len(gt_neg_matches_A)))
            nn_indices_choice = np.random.choice(nn_indices, nn, replace=False)
            gt_neg_matches_A = gt_neg_matches_A[nn_indices_choice]
            gt_neg_matches_B = gt_neg_matches_B[nn_indices_choice]
        
        gt_matches_A = np.hstack((gt_matches_A, gt_neg_matches_A))
        gt_matches_B = np.hstack((gt_matches_B, gt_neg_matches_B))

        gt_matches = np.hstack((gt_matches_A[:,np.newaxis], 
                                gt_matches_B[:,np.newaxis]))

        data = {'descriptors0': desc0.T,
                'descriptors1': desc1.T,
                'gt_matches': gt_matches,
                
                # for stat purposes
                'gt_matches_matrix': gt_matches_matrix,
                'obm_A_num_detection': len(obm_A),
                'obm_B_num_detection': len(obm_B),
                'obm_A_num_detection_wuid': obm_A.get_num_objects_wuid(),
                'obm_B_num_detection_wuid': obm_B.get_num_objects_wuid(),
               }

        return data

    def collate(self, batch):
        descriptors0 = [item['descriptors0'] for item in batch]
        descriptors1 = [item['descriptors1'] for item in batch]
        gt_matches = [item['gt_matches'] for item in batch]
        
        # for stat purposes
        gt_matches_matrix = [item['gt_matches_matrix'] for item in batch]
        obm_A_num_detection = [item['obm_A_num_detection'] for item in batch]
        obm_B_num_detection = [item['obm_B_num_detection'] for item in batch]
        obm_A_num_detection_wuid = [item['obm_A_num_detection_wuid'] for item in batch]
        obm_B_num_detection_wuid = [item['obm_B_num_detection_wuid'] for item in batch]
        return {'descriptors0':descriptors0,
                'descriptors1':descriptors1,
                'gt_matches':gt_matches,

                # for stat purposes
                'gt_matches_matrix': gt_matches_matrix,
                'obm_A_num_detection': obm_A_num_detection,
                'obm_B_num_detection': obm_B_num_detection,
                'obm_A_num_detection_wuid': obm_A_num_detection_wuid,
                'obm_B_num_detection_wuid': obm_B_num_detection_wuid,
               }






