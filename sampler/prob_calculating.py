""""
Data: 2021/09/16
Target: Calculate the probability and distribution of the FLOPS, Parametes, opts
Method: random.choices(seq, weights=prob)https://docs.python.org/3/library/random.html
"""

import torch 
import random
import math
import torch.nn.functional as F

def compute_kl_div(input_seq, target_seq): 
    assert len(input_seq)==len(target_seq)

    kl_value = F.kl_div(input_seq,target_seq.log(),reduction='batchmean')  # input is the log-prob, the target is prob
    return kl_value


def compute_constraint_value(value, step):
    return int(math.ceil(value/step) * step)            # ceil

def convert_count_to_prob(counts_dict):

    total = sum(counts_dict.values())
    for idx in counts_dict:
        counts_dict[idx] = 1.0 * counts_dict[idx] / total
    
    return counts_dict

def build_prob_list(raw_list, batch_min, batch_max, bins=8):        
    # 输入的是raw_list是tier的list, 但是min和max是batch的最大最小值，因为对于一个batch预测出的多个tier，为了计算其分布的相同，起始和终止点应该相同
    
    raw_list.sort()
    step = int((batch_max-batch_min)/(bins-1))
    
    counts_dict = {}
    for i in range(1, bins+1):  # 生成constraint bin 字典，存储每个小于constraint的子网的计数
        counts_dict[i * step] = 0
    
    for value in raw_list:
        value = compute_constraint_value(value, step)
        counts_dict[value] += 1

    counts_dict = convert_count_to_prob(counts_dict)

    return counts_dict

if __name__ == '__main__':

    # # for debug compute_kl_div
    # a = torch.randn(50).softmax(-1)
    # b = torch.randn(50).softmax(-1)
    # print(compute_kl_div(a,b))

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
    l1 = build_prob_list(params_list, batch_min = min_p, batch_max = max_p)
    print(l1)
    




    

# random.choice(seq, weights=prob)