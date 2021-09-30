"""
Data: 2021/09/18
Target: Sample subnets under FLOPS and Parameters contraints
Method: 1 sample target FLOPS and Parameters constraints according to top tier distribution
        2 sample n_nodes according to top tier distribution
        3 Given the sampled n_nodes, sample edges and nodes type according to uniform distribution
        4 Check whether the sampled subnets satisfy the FLOPS and Parameters constraints

        random包: https://docs.python.org/3/library/random.html

input: 1 batch_statics_dict = 
        {
            'flops':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'params':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'n_nodes':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            } # tier是counts_dict
"""
import random

from architecture import ModelSpec, seq_decode_to_arch
from dataset import SplitSubet

from .prob_calculate import select_distri, extract_dict_value_to_list

def sample_helper(disti_dict):
    target_list = extract_dict_value_to_list(disti_dict, is_key= True)      # 提取计数, key
    prob_list = extract_dict_value_to_list(disti_dict)                      # 提取概率，value
    return target_list, prob_list

class ArchSampler(object):
    def __init__(self, top_tier, last_tier, batch_factor, node_type_dict, max_edges = 9, reuse_step=None):
        self.top_tier = top_tier
        self.last_tier = last_tier
        self.batch_factor = batch_factor
        self.reuse_step = reuse_step
        self.node_type_dict = node_type_dict        # type_dict = {'input':1, 'conv1x1-bn-relu':2 , 'conv3x3-bn-relu':3, 'maxpool3x3':4, 'output':5}
        self.max_edges = max_edges

    def reset_parameters(self, batch_statics_dict):
        self.batch_flops_list = batch_statics_dict['flops']
        self.batch_params_list = batch_statics_dict['params']
        self.batch_n_nodes_list = batch_statics_dict['n_nodes']

    def _sample_target_value(self, candi_list, threshold_kl_div=5):
        batch_size = 0
        for dic in candi_list:
            batch_size += sum(extract_dict_value_to_list(dic))
        target_distri = select_distri(candi_list, self.top_tier, self.last_tier, threshold_kl_div, batch_size, self.batch_factor)
        
        if target_distri is not None:
            target_list, prob_list = sample_helper(target_distri)
            target_value = random.choices(target_list, weights=prob_list)[0]
            return target_value
        else:
            target_list = extract_dict_value_to_list(candi_list[0], is_key=True)
            target_value = random.choice(target_list)
            return target_value

    def _sample_edges_and_types(self, n_nodes, arch_struct_list:list):
        type_candi_list = extract_dict_value_to_list(self.node_type_dict)[1:-1]

        for node_idx in range(1, n_nodes):                            # input后的第一个点开始，到output点，为每个点决定之前的连接
            # 1 sample previous node
            pre_node_id = random.choice([i for i in range(node_idx)])   # 从i节点之前的i-1个节点中选取一个座连接
            arch_struct_list.append(pre_node_id)

            # 2 sample node type
            if node_idx == n_nodes-1: # 最后一个节点是output不需要做决定
                break
            node_opt_type = random.choice(type_candi_list)
            arch_struct_list.append(node_opt_type)
            
        # 3 sampel rest edges
        # max edges = 9, the rest edges are (max_edges -(n_nodes-1))
        for i in range(self.max_edges-n_nodes+1):           # 遍历剩余边的start和end节点
            # sample begin node for the edge
            begin_node_idx = random.choice([b_idx for b_idx in range(n_nodes)])
            if begin_node_idx == n_nodes-1:   # if begin node is the output node, this edge does not exist
                arch_struct_list += [None, None]
                continue
            
            # sample end node for the edge
            end_node_idx = random.choice([l_idx for l_idx in range(begin_node_idx, n_nodes)])
            if end_node_idx == begin_node_idx:
                arch_struct_list += [None, None]
                continue

            arch_struct_list += [begin_node_idx, end_node_idx]

        # 1st for loop: (n_nodes-1)times each add 2(edge and opt type) excep the last output node, yield (n_nodes-1)*2-1
        # 2nd for loop: (max_edges-n_nodes+1)times each add 2(start and end nodes), yield (max_edges-n_nodes+1)*2
        # n_nodes stored in the first place of the list, yield 1
        # total (n_nodes-1)*2-1+(max_edges-n_nodes+1)*2+1
        assert len(arch_struct_list) == self.max_edges * 2, 'Wrong length of sampled arch_struct_list'
        return arch_struct_list

    def sample_arch(self, batch_statics_dict, n_subnets, dataset: SplitSubet, kl_thred=[5,8,1], max_trails=100):
        self.reset_parameters(batch_statics_dict)       # update batch statics with new batch
        flops_kl_thred = kl_thred[0]
        params_kl_thred = kl_thred[1]
        n_nodes_kl_thred = kl_thred[2]

        sampled_arch = []
        sampled_arch_datast_idx = []
        flops, params, n_nodes = 0,0,0

        self.reuse_step = 1 if self.reuse_step is None else self.reuse_step
        assert self.reuse_step > 0, 'the reuse step must greater than 0'
        
        reuse_count = 0
        while len(sampled_arch) < n_subnets:
            # FLOPS和params可重复若干个采样过程使用，提高采样速度
            # step1 sample flops and params constraints
            if reuse_count % self.reuse_step == 0:
                # sample target flops
                flops = self._sample_target_value(self.batch_flops_list, flops_kl_thred)

                # sample target params
                params = self._sample_target_value(self.batch_params_list, params_kl_thred)

            for trail in range(max_trails+1):
                arch_struct_list = []   # store arch
                
                # step2 sample target n_nodes
                # n_nodes_list一个cell总的节点数, 也即是包括input和output节点的
                n_nodes = self._sample_target_value(self.batch_n_nodes_list, n_nodes_kl_thred)
                arch_struct_list.append(n_nodes)
                
                # step3 sample nodes type and connectoin
                arch_struct_list = self._sample_edges_and_types(n_nodes, arch_struct_list)       # 这里的节点要包括input和output节点数

                # step4 check wheth satisfy the flops and params constraints
                matrix, opt = seq_decode_to_arch(arch_struct_list)
                arch_spec = ModelSpec(matrix=matrix, ops=opt)

                # 查询训练集，query到flops和params，判断是否满足，满足则sampled_arch.append()且break，否则continue
                f, p, dataset_idx = dataset.query_stats_by_spec(arch_spec)
                if dataset_idx is None:  # 检查采样结构是否在训练集中，不在则继续采样
                    continue
                if f <= flops and p <= params:  # 检查采样结构是否满足constrains，满足则停止采样
                    break
            
            sampled_arch_datast_idx.append(dataset_idx)
            if dataset_idx is None:
                sampled_arch.append(None)
            else:
                sampled_arch.append(arch_spec)  # append请: 1满足条件时：break时append，2若不满足条件，采样max_trails+1个放入
            
            reuse_count +=1

        return sampled_arch, sampled_arch_datast_idx
    
        

if __name__ == '__main__':

    # for test _sample_target_value
    # import copy
    # counts_dict = {5758991: 232, 11517982: 93, 17276973: 38, 23035964: 31, 28794955: 8, 34553946: 18, 40312937: 2, 46071928: 1}
    # counts_dict2 = {5758991: 1, 11517982: 2, 17276973:18, 23035964: 8, 28794955: 31, 34553946: 38, 40312937: 93, 46071928: 232}
    # counts_dict3 = copy.deepcopy(counts_dict)
    # counts_dict4 = copy.deepcopy(counts_dict2)
    # counts_dict5 = copy.deepcopy(counts_dict)
    # counts_dict6 = copy.deepcopy(counts_dict2)
    # list_candi1 = [counts_dict,counts_dict2]
    # list_candi2 = [counts_dict3,counts_dict4]
    # list_candi3 = [counts_dict5,counts_dict6]
    # candi_dict = {}
    # candi_dict['flops'] = list_candi1
    # candi_dict['params'] = list_candi2
    # candi_dict['n_nodes'] = list_candi3
    # sampler = ArchSampler(n_tier=1,threshold_kl_div=27,batch_size=423,batch_factor=1,reuse_step=10)
    # v = sampler.sample_subnets(candi_dict, 100, kl_factor_list=[1,1,1])
    # print(v)
    
    #  for test __sample_edges_and_types
    type_dict = {'input':1, 'conv1x1-bn-relu':2 , 'conv3x3-bn-relu':3, 'maxpool3x3':4, 'output':5}
    sampler = ArchSampler(top_tier=1,threshold_kl_div=27,batch_size=423,batch_factor=1,reuse_step=10,node_type_dict=type_dict)
    v = sampler._sample_edges_and_types(7,[7])
    print(v)

    
    