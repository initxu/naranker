"""
Date: 2021/09/23
Target: loss for transformer

implementation adapted from Slimmable: https://github.com/JiahuiYu/slimmable_networks.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = F.log_softmax(output, dim=1)              # [b_sz, n_class]
        target = target.unsqueeze(1)                                # [b_sz, 1, n_class]
        output_log_prob = output_log_prob.unsqueeze(2)              # [b_sz, n_class, 1]
        cross_entropy_loss = -torch.bmm(target, output_log_prob)    # [b_sz, 1, n_class]*[b_sz, n_class, 1)=[b_sz, 1]
        return cross_entropy_loss.mean()                            # 求batch loss的平均

class GlobalClassDistanceRegularization(nn.Module):
    def forward(self, output, target):
        index = torch.tensor([v+1 for v in range(target.size(1))]).cuda(output.device)
        output_prob = F.softmax(output, dim=1)
        output_expec = (output_prob*index).sum(dim=1)
        target_expec = (target*index).sum(dim=1)

        return (torch.norm(target_expec - output_expec))

class TopRankRegularization(nn.Module):
    def forward(self, output, score):
        
        output_log_prob = F.softmax(output, dim=1)
        prob_val, prob_idx = torch.topk(output_log_prob, k=1, dim=1)
        prob_val = prob_val.squeeze(dim=1)
        prob_idx = prob_idx.squeeze(dim=1)
        
        t1_idx = torch.where(prob_idx == 0)
        t1_pred_prob =  prob_val[t1_idx]
        t1_gt_score = score[t1_idx]
        
        reg_term = 0
        if t1_gt_score.size(0) == 0:
            return 0
        else:    
            for i in range(len(t1_pred_prob)-1):
                tmp = (t1_pred_prob[i] - t1_pred_prob[i+1]) * torch.sign(t1_gt_score[i]-t1_gt_score[i+1])
                reg_term += torch.log(1+torch.exp(-tmp))

        return reg_term