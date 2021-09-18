"""
random包: https://docs.python.org/3/library/random.html
"""

from prob_calculating import compute_kl_div

import random

def extract_dict_value_to_list(prob_dict):
    assert type(prob_dict) == dict, 'Not a dictionary'
    return list(prob_dict.values())

def select_distri(list_candi, n_tier, threshold_kl_div, batch_size, batch_factor):
# 先确定分布的样本数量是否足够多，再确定候选分布与差结构的分布的相似性
# threahold_kl_div的应该随着batch增加越来越大，因为后期5个层级的分布可能趋同
    for i in range(n_tier):
        candi = list_candi[i]                                   # 提取list存储的counts_dict作为candidates
        if sum(candi.values()) < batch_factor * batch_size:     # 若candi的计数小于batch的batch_factor时，这个分布不能用，继续遍历第二个
            continue
        
        candi_counts = extract_dict_value_to_list(candi)
        
        for j in range(n_tier, len(list_candi)):
            low_tier_counts = extract_dict_value_to_list(list_candi[j])
            if compute_kl_div(low_tier_counts, candi_counts) < threshold_kl_div:        # 送两个原始计数序列去计算
                # 当差结构的分布与candi的分布相近时，表明candi分布不能代表优秀的结构
                return None

        return candi

if __name__ == '__main__':
    # seq = []
    # for i in range(10):
    #     seq.append(random.randint(0,100))
    # prob = []
    # for j in range(10):
    #     prob.append(random.random())

    # a = random.choices(seq, weights=prob)
    # print(a)



    # # 1.for debug extract_dict_value_to_list        #########
    counts_dict = {5758991: 232, 11517982: 93, 17276973: 38, 23035964: 31, 28794955: 8, 34553946: 18, 40312937: 2, 46071928: 1}
    counts_dict2 = {5758991: 1, 11517982: 2, 17276973:18, 23035964: 8, 28794955: 31, 34553946: 38, 40312937: 93, 46071928: 232}
    # # 直接用counts送入kl散度，kl散度函数中用log_softmax归一就好
    # # counts_dict = convert_count_to_prob(counts_dict)
    # # counts_dict2 = convert_count_to_prob(counts_dict2)
    # counts_list = extract_dict_value_to_list(counts_dict)
    # counts_list2 = extract_dict_value_to_list(counts_dict2)
    # v = compute_kl_div(counts_list,counts_list2)    
    # print(v)

    # 2.for debug select_distri                   #########
    list_candi = [counts_dict,counts_dict2]
    v = select_distri(list_candi,1,27,423,0.1)
    print(v)