#!/usr/bin/env python
import torch as th
class SILogLoss(th.nn.Module):
    def __init__(self,NORM_CONST = 2**15 - 1):
        super(SILogLoss, self).__init__()
        self.NORM_CONST = NORM_CONST
    def forward(self,prediction,target):
        # https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf Section 3.2
        mask = th.isfinite(target) & (target > 0)
        prediction = th.clamp(prediction, min=1e-8)
        num_vals = mask.sum()
        log_diff = th.log(prediction[mask]) - th.log(target[mask]/self.NORM_CONST)
        #log_diff = prediction[mask] - th.log(target[mask])
        si_log_unscaled = th.sum(log_diff**2) / num_vals - (th.sum(log_diff)**2) / (num_vals**2)
        #print(si_log_unscaled)
        si_log_score = th.sqrt(si_log_unscaled)*100
        return si_log_score
        #return si_log_unscaled

    '''
    def forward(self, pred_depth, true_depth_raw):
        mask = true_depth_raw == 0
        true_depth = true_depth_raw / self.NORM_CONST
        #print(th.max(true_depth),th.min(true_depth))
        T = th.count_nonzero(true_depth)
        #pred_depth = th.clamp(pred_depth, min=1e-8)
        true_depth = th.clamp(true_depth, min=1e-8)
        
        diff = pred_depth - th.log(true_depth)
        return (1/T)*th.sum((diff*mask)**2) - (1/(T**2))*th.sum(diff*mask)**2
    '''





