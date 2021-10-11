import json
import time
import copy


class NASBench201DataBase(object):
    def __init__(self, data_file):
        self.archs = {}

        self._load_json_file(data_file)

    def _load_json_file(self, data_file):
        print('Loading nasbench201 dataset from file')
        start = time.time()

        with open(data_file, 'r') as f:
            dataset = json.load(f)
        f.close()

        for arch in dataset:
            self.archs.update(arch)

        self._sort()

        elapsed = time.time() - start
        print('Loaded dataset in {:.4f} seconds'.format(elapsed))

    def query_by_id(self, arch_id):
        arch_data_dict = copy.deepcopy(self.archs[arch_id])
        return arch_data_dict

    def _sort(self):
        for network_type in ['cifar10', 'cifar100', 'imagenet16']:
            sorted_list = []
            for id, arch in self.archs.items():
                sorted_list.append((id, arch['{}_test_acc'.format(network_type)]))

            sorted_list = sorted(sorted_list, key=lambda item:item[1],reverse=True)

            for rank, (id, _) in enumerate(sorted_list, start=1):
                self.archs[id]['{}_rank'.format(network_type)] = rank

    def index_iterator(self):
        return self.archs.keys()

    @property
    def size(self):
        return len(self.archs)
