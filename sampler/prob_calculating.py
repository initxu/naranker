""""
Data: 2021/09/16
Target: Calculate the probability and distribution of the FLOPS, Parametes, opts
Method: Distribution and probability are modified from AttentiveNAS
"""
import torch
import math
import torch.nn.functional as F

def compute_kl_div(input_seq, target_seq): 
    # 直接接受计数list，在本函数内做log_softmax的归一
    # 不在input后加log()是因为若计数为0，频数做的概率也为0，加log会产生inf，所以这里归一
    assert len(input_seq)==len(target_seq)
    
    input_seq = torch.tensor(input_seq, dtype = torch.float) if not isinstance(input_seq, torch.Tensor) else input_seq
    target_seq = torch.tensor(target_seq, dtype = torch.float) if not isinstance(target_seq, torch.Tensor) else target_seq
    input_seq = F.log_softmax(input_seq, dim=-1)
    target_seq = F.log_softmax(target_seq, dim=-1)
    
    kl_value = F.kl_div(input_seq, target_seq, reduction='batchmean', log_target=True)  # KL_div要求input is the log-prob, the target is prob，所以log_target设置为True
    return kl_value


def compute_constraint_value(value, step):
    return int(math.ceil(value/step) * step)            # ceil

def convert_count_to_prob(counts_dict):

    total = sum(counts_dict.values())
    for idx in counts_dict:
        counts_dict[idx] = 1.0 * counts_dict[idx] / total
    
    return counts_dict

def build_counts_dict(raw_list, batch_min, batch_max, bins=8):        
    # 输入的是raw_list是tier的list, 但是min和max是batch的最大最小值，因为对于一个batch预测出的多个tier，为了对比分布的差异，起始和终止点应该相同
    
    raw_list.sort()
    step = int((batch_max-batch_min)/(bins-1))
    
    counts_dict = {}
    for i in range(1, bins+1):  # 生成constraint bin 字典，存储每个小于constraint的子网的计数
        counts_dict[i * step] = 0
    
    for value in raw_list:
        value = compute_constraint_value(value, step)
        counts_dict[value] += 1

    return counts_dict

if __name__ == '__main__':

    # for debug compute_kl_div
    # 见 ArchSamp

    # for debug 
    filename = "/home/ubuntu/workspace/nar/data/nasbench101/nasbench_only108_423.json"
    import json
    with open(filename,'r') as f:
        dataset = json.load(f)
    f.close()
    params_list = []
    for _,subnet in enumerate(dataset):
        params_list.append(subnet['trainable_parameters'])
    max_p, min_p = max(params_list), min(params_list)
    l1 = build_counts_dict(params_list, batch_min = min_p, batch_max = max_p)
    print(l1)
    l2 = convert_count_to_prob(l1)
    print(l2)
