"""
Date: 2021/09/23
Target: loss for transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossSoft(nn.Module):
    """Modified from Slimmable: https://github.com/JiahuiYu/slimmable_networks.git"""
    def forward(self, output, target):
        output_log_prob = F.log_softmax(output, dim=1)              # [b_sz, n_class]
        target = target.unsqueeze(1)                                # [b_sz, 1, n_class]
        output_log_prob = output_log_prob.unsqueeze(2)              # [b_sz, n_class, 1]
        cross_entropy_loss = -torch.bmm(target, output_log_prob)    # [b_sz, 1, n_class]*[b_sz, n_class, 1)=[b_sz, 1]
        
        return cross_entropy_loss.mean()                            # 求batch loss的平均


class RankLoss(nn.Module):
    """modified from ReNAS: https://github.com/huawei-noah/Efficient-Computing.git"""
    def forward(self, outputs, labels):
        output = outputs.unsqueeze(1)
        output1 = output.repeat(1,outputs.shape[0])
        label = labels.unsqueeze(1)
        label1 = label.repeat(1,labels.shape[0])
        tmp = (output1-output1.t())*torch.sign(label1-label1.t())   # 行代表每一个预测精度与batch中其他预测精度的对比，对角线为自己与自己对比=0
        tmp = torch.log(1+torch.exp(-tmp))                          # hinge function
        eye_tmp = tmp*torch.eye(len(tmp)).cuda(device=outputs.device)
        new_tmp = tmp - eye_tmp
        loss = torch.sum(new_tmp)/(outputs.shape[0]*(outputs.shape[0]-1))

        return loss


class CrossEntropyLossLabelSmooth(nn.Module):
    def forward(self, output, target, eps=0.1):
        bz = target.size(0)
        n_class = target.size(1)
        
        smoothed_labels = torch.full(size=(bz,n_class), fill_value=eps/(n_class-1)).cuda(device=target.device)
        _, idx = torch.where(target == 1)
        smoothed_labels.scatter_(dim=1, index=idx.unsqueeze(dim=1), value=1 - eps) # scatter待验证
        output_log_prob = F.log_softmax(output, dim=1)                          # [b_sz, n_class]
        smoothed_labels = smoothed_labels.unsqueeze(1)                          # [b_sz, 1, n_class]
        output_log_prob = output_log_prob.unsqueeze(2)                          # [b_sz, n_class, 1]
        cross_entropy_loss = -torch.bmm(smoothed_labels, output_log_prob)       # [b_sz, 1, n_class]*[b_sz, n_class, 1)=[b_sz, 1]
        return cross_entropy_loss.mean()


class DeepExpectation(nn.Module):
    def forward(self, output, target):
        index = torch.tensor([v+1 for v in range(target.size(1))]).cuda(output.device)
        output_prob = F.softmax(output, dim=1)
        output_expec = (output_prob*index).sum(dim=1)
        target_expec = (target*index).sum(dim=1)

        return (torch.norm(target_expec - output_expec))
