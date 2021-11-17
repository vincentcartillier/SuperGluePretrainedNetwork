import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class SuperGlueLoss():
    def compute(self,
                scores: List[torch.FloatTensor],
                gt_matches: List[np.ndarray]) -> torch.FloatTensor:
        loss = 0
        total = 0
        B = len(gt_matches)
        for b in range(B):
            rows = gt_matches[b][:,0]
            cols = gt_matches[b][:,1]
            #tmp_loss = -torch.log(scores[b][rows, cols].exp())
            tmp_loss = -torch.log(scores[b][rows, cols]+0.001)
            loss += torch.sum(tmp_loss)
            total += len(tmp_loss)
            # -- for i in range(len(gt_matches[b])):
            # --     x = gt_matches[b][i][0]
            # --     y = gt_matches[b][i][1]
            # --     loss.append(-torch.log( scores[b][x][y].exp() )) # check batch size == 1 ?
        loss_mean = loss/total 
        #loss_mean = torch.mean(torch.stack(loss))
        loss_mean = torch.reshape(loss_mean, (1, -1))
        return loss_mean
 
