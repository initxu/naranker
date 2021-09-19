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

def extract_dict_value_to_list(target_dict, is_key=False):
    assert type(target_dict) == dict, 'Not a dictionary'
    if is_key:
        return list(target_dict.keys())
    return list(target_dict.values())

def select_distri(candi_list, n_tier, threshold_kl_div, batch_size, batch_factor):
# 先确定分布的样本数量是否足够多，再确定候选分布与差结构的分布的相似性
# threahold_kl_div的应该随着batch增加越来越大，因为后期5个层级的分布可能趋同
# 输入[{tier1},{tier2},{tier3},{tier4},{tier5}] list of counts_dict
# 输出None or prob_dict
    assert n_tier < len(candi_list), 'The candidates tier indexs should be smaller than the length of candidates list'
    for i in range(n_tier):
        candi = candi_list[i]                                   # 提取list存储的counts_dict作为candidates
        if sum(candi.values()) < batch_factor * batch_size:     # 若candi的计数小于batch的batch_factor时，这个分布不能用，继续遍历第二个
            continue
        
        candi_counts = extract_dict_value_to_list(candi)
        for j in range(n_tier, len(candi_list)):
            low_tier_counts = extract_dict_value_to_list(candi_list[j])
            if compute_kl_div(low_tier_counts, candi_counts) < threshold_kl_div:        # 送两个原始计数序列去计算
                # 当差结构的分布与candi的分布相近时，表明candi分布不能代表优秀的结构
                return None

        return convert_count_to_prob(candi)

if __name__ == '__main__':

    # # 1.for debug build_counts_dict&convert_count_to_prob         ########
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
    l3 = extract_dict_value_to_list(l2)
    print(l3)
    l4 = extract_dict_value_to_list(l2,is_key=True)
    print(l4)

    # # 2.for debug extract_dict_value_to_list        #########
    counts_dict = {5758991: 232, 11517982: 93, 17276973: 38, 23035964: 31, 28794955: 8, 34553946: 18, 40312937: 2, 46071928: 1}
    counts_dict2 = {5758991: 1, 11517982: 2, 17276973:18, 23035964: 8, 28794955: 31, 34553946: 38, 40312937: 93, 46071928: 232}
    # # 直接用counts送入kl散度，kl散度函数中用log_softmax归一就好
    # counts_dict = convert_count_to_prob(counts_dict)
    # counts_dict2 = convert_count_to_prob(counts_dict2)
    counts_list = extract_dict_value_to_list(counts_dict)
    counts_list2 = extract_dict_value_to_list(counts_dict2)
    v = compute_kl_div(counts_list,counts_list2)    
    print(v)

    # # 3.for debug select_distri                   #########
    # list_candi = [counts_dict,counts_dict2]
    # v = select_distri(list_candi,1,21,423,0.1)
    # print(v)
