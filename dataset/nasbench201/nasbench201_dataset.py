import torch
import copy

from torch.utils.data import Dataset

from architecture import feature_tensor_encoding_201

from .nasbench201_database import NASBench201DataBase

class NASBench201Dataset(Dataset):
    def __init__(self, database: NASBench201DataBase, seed):
        self.database = database
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)

        self.index_list = torch.randperm(self.database.size, generator=g_cpu).tolist()
        self.keys_list = list(self.database.index_iterator())

    def __getitem__(self, index):
        arch_id = self.keys_list[self.index_list[index]]
        arch = self.database.query_by_id(arch_id)

        # TODO: 待完成arch_feature_encoding后，继续提取数据
        arch_feature = feature_tensor_encoding_201(copy.deepcopy(arch))
        for net_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
            if net_type == 'cifar10-valid':
                val_acc = arch['cifar10_val_acc']
                test_acc = arch['cifar10_test_acc']
                rank = arch['cifar10_rank']
            elif net_type == 'ImageNet16-120':
                val_acc = arch['imagenet16_val_acc']
                test_acc = arch['imagenet16_test_acc']
                rank = arch['imagenet16_rank']
            else:
                val_acc = arch['cifar100_val_acc']
                test_acc = arch['cifar100_test_acc']
                rank = arch['cifar100_rank']
            
            params = arch['{}_total_params'.format(net_type)]
            flops = arch['{}_total_flops'.format(net_type)]
            # n_edges = arch['{}']
            




        
        

        



