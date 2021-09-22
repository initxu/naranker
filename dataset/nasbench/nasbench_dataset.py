import torch
import copy
from torch.utils.data import Dataset

from architecture import feature_tensor_encoding

from .nasbench_database import NASBenchDataBase


class NASBenchDataset(Dataset):
    def __init__(self, database: NASBenchDataBase, seed):
        self.database = database
        g_cpu = torch.Generator()   # 因为是在cpu读json数据并shuffle
        g_cpu.manual_seed(seed)
        self.index_list = torch.randperm(self.database.size, generator=g_cpu).tolist()
        self.keys_list = list(self.database.hash_iterator())


    def __getitem__(self, index):
        model_hash = self.keys_list[self.index_list[index]]
        arch = self.database.query_by_hash(model_hash)
        
        arch_feature = feature_tensor_encoding(copy.deepcopy(arch))
        validation_accuracy = arch['avg_validation_accuracy']
        test_accuracy = arch['avg_test_accuracy']
        params = arch['trainable_parameters']
        flops = arch['flops']
        
        return arch_feature, validation_accuracy, test_accuracy, params, flops

    def __len__(self):
        return self.database.size