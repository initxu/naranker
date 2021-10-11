"""
Data: 2021/10/11
Target: Sample subnets under FLOPS and Parameters contraints for [Nasbench201]

input:
"""
import random

from dataset import NASBench201Dataset


class ArchSampler201(object):
    def __init__(self, reuse_step):

        self.reuse_step = reuse_step
        return

    def reset_parameters():
        raise NotImplementedError

    def sample_arch(self,
                    batch_statics_dict,
                    n_subnets,
                    dataset: NASBench201Dataset,
                    kl_thred=[2, 2],
                    max_trails=100,
                    force_uniform=False):
        
        self.reset_parameters(batch_statics_dict)
        flops_kl_thred = kl_thred[0]
        params_kl_thred = kl_thred[1]

        sampled_arch = []
        sampled_arch_datast_idx = []
        flops, params, n_nodes = 0, 0, 0

        self.reuse_step = 1 if self.reuse_step is None else self.reuse_step
        assert self.reuse_step > 0, 'the reuse step must greater than 0'

        while len(sampled_arch) < n_subnets:
            if len(sampled_arch) % self.reuse_step == 0:


