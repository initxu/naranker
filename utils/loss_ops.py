"""
Date: 2021/09/23
Target: loss for transformer

implementation adapted from Slimmable: https://github.com/JiahuiYu/slimmable_networks.git
"""

import torch
import torch.nn.functional as F

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = F.log_softmax(output, dim=1)              # [b_sz, n_class]
        target = target.unsqueeze(1)                                # [b_sz, 1, n_class]
        output_log_prob = output_log_prob.unsqueeze(2)              # [b_sz, n_class, 1]
        cross_entropy_loss = -torch.bmm(target, output_log_prob)    # [b_sz, 1, n_class]*[b_sz, n_class, 1)=[b_sz, 1]
        return cross_entropy_loss.mean()                            # 求batch loss的平均