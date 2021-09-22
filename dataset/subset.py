import copy
from torch.utils.data import Dataset

from architecture import ModelSpec


class SplitSubet(Dataset):
    def __init__(self, full_dataset, indices: list):
        self.full_dataset = full_dataset
        self.indices = indices

        self.subset = [self.full_dataset[i] for i in self.indices]

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.indices)

    def query_stats_by_spec(self, model_spec: ModelSpec):
        # step1: 找到spec对应的hash
        arch_dict = self.full_dataset.database.check_arch_inside_dataset(model_spec)
        if arch_dict is None:
            return None, None, None
        model_hash = arch_dict['unique_hash']

        # step2: 找到hash在keys_list的下标
        hash_list_idx = self.full_dataset.keys_list.index(model_hash)
        # step3: 找到index的的下标
        index_list_idx = self.full_dataset.index_list.index(hash_list_idx)

        # step4: 判断下标是否在subset范围内
        if index_list_idx not in self.indices:
            return None, None, None

        return arch_dict['flops'], arch_dict['trainable_parameters'], index_list_idx
