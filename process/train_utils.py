import torch

from architecture import Bucket
from sampler.prob_calculate import build_counts_dict, build_n_nodes_counts_dict, compute_kl_div, extract_dict_value_to_list

def get_target(score, n_tier, batch_sz):
    _, idx = score.sort(descending=True)
    step = int(batch_sz / n_tier)
    t_flag = []

    target = torch.zeros(batch_sz, n_tier)

    for i in range(n_tier):
        if i == n_tier - 1:
            t_flag = idx[i * step:]
        else:
            t_flag = idx[i * step:(i + 1) * step]

        for j in t_flag:
            target[j, i].add_(1)

    return target

def init_tier_list(args):
    tier_list = []
    # initial tier emb
    for i in range(args.ranker.n_tier):
        t = Bucket(flag_tier=i,
                   name_tier='tier_{}'.format(i + 1),
                   n_arch_patch=args.ranker.n_arch_patch,
                   d_patch_vec=args.ranker.d_patch_vec)
        tier_list.append(t)

    return tier_list

def get_tier_emb(tier_list:list, deivce):
    tier_emb_list = []
    for item in tier_list:
        tier_emb_list.append(item.get_bucket_emb().cuda(deivce))
        
    return torch.cat(tier_emb_list, dim=0)


def classify_tier_emb_by_target(total_embedding_list, tier_list, target):
    for i in range(len(total_embedding_list)):
        idx = torch.where(target[:,i] == 1)
        tier_emb = total_embedding_list[i][idx]
        if tier_emb.size(0)==0 or tier_emb.size(1)==0 or tier_emb.size(2)==0: # 本tier此时没有分到结构编码
            continue
        else:
            tier_list[i].updata_bucket_emb(tier_emb)

def classify_tier_emb_by_pred(total_embedding_list, tier_list, pred):
    _, index = torch.topk(pred, k=1, dim=1)
    index = index.squeeze(dim=1)
    for i in range(len(total_embedding_list)):
        idx = torch.where(index == i)
        tier_emb = total_embedding_list[i][idx]
        if tier_emb.size(0)==0 or tier_emb.size(1)==0 or tier_emb.size(2)==0: # 本tier此时没有分到
            continue
        else:
            tier_list[i].updata_bucket_emb(tier_emb)

def classify_tier_counts_by_target(params, flops, n_nodes, tier_list, target, bins):
    params = torch.ceil(params/1e3)
    max_p, min_p = max(params), min(params)
    flops = torch.ceil(flops/1e6)
    max_f, min_f = max(flops), min(flops)
    max_n, min_n = max(n_nodes), min(n_nodes)

    for i in range(len(tier_list)):
        idx = torch.where(target[:,i] == 1)
        #{3716836: 0, 7433672: 0, 11150508: 0, 14867344: 0, 18584180: 0, 22301016: 0, 26017852: 0, 29734688: 0}
        p_counts = build_counts_dict(params[idx].tolist(),batch_min=min_p,batch_max=max_p, bins=bins,scail=1e3) 
        f_counts = build_counts_dict(flops[idx].tolist(),batch_min=min_f, batch_max=max_f, bins=bins, scail=1e6)
        n_counts = build_n_nodes_counts_dict(n_nodes[idx].tolist(), batch_min=int(min_n), batch_max=int(max_n))
        tier_list[i].update_counts_dict(p_counts,f_counts,n_counts)

def classify_tier_counts_by_pred(params, flops, n_nodes, tier_list, pred, bins):
    params = torch.ceil(params/1e3)
    max_p, min_p = max(params), min(params)
    flops = torch.ceil(flops/1e6)
    max_f, min_f = max(flops), min(flops)
    max_n, min_n = max(n_nodes), min(n_nodes)

    _, index = torch.topk(pred, k=1, dim=1)
    index = index.squeeze(dim=1)
    for i in range(len(tier_list)):
        idx = torch.where(index == i)
        #{3716836: 0, 7433672: 0, 11150508: 0, 14867344: 0, 18584180: 0, 22301016: 0, 26017852: 0, 29734688: 0}
        p_counts = build_counts_dict(params[idx].tolist(),batch_min=min_p,batch_max=max_p, bins=bins,scail=1e3) 
        f_counts = build_counts_dict(flops[idx].tolist(),batch_min=min_f, batch_max=max_f, bins=bins, scail=1e6)
        n_counts = build_n_nodes_counts_dict(n_nodes[idx].tolist(), batch_min=int(min_n), batch_max=int(max_n))
        tier_list[i].update_counts_dict(p_counts,f_counts,n_counts)

def get_batch_statics(tier_list):
    p_list = []
    f_list = []
    n_list = []
    
    for i in range(len(tier_list)):
        tier_counts = tier_list[i].get_bucket_counts()
        p_list.append(tier_counts['params'])
        f_list.append(tier_counts['flops'])
        n_list.append(tier_counts['n_nodes'])
    return {'params':p_list, 'flops':f_list, 'n_nodes':n_list}

def compare_kl_div(batch_statics:list):
    t1 = batch_statics[0]
    t1 = extract_dict_value_to_list(t1)
    kl_dict = {}
    for i in range(1, len(batch_statics)):
        t = extract_dict_value_to_list(batch_statics[i])
        kl_v = compute_kl_div(t, t1)
        kl_dict['t{}||t1'.format(i+1)] = kl_v

    return kl_dict