import torch
import time
import random
import torch.utils.data
import torch.nn.functional as F

from dataset import NASBenchDataset, SplitSubet
from sampler import ArchSampler

from .train_utils import *

def evaluate_sampled_batch(model, sampler : ArchSampler, tier_list, batch_statics_dict, dataset : NASBenchDataset, it, args, device, writer, logger, flag):

    sample_start = time.time()

    sample_size = int(args.sampler.sample_size * (1 - args.sampler.noisy_factor))
    kl_thred = [
        args.sampler.flops_kl_thred, 
        args.sampler.params_kl_thred
    ]

    # sample from entire dataset
    sampled_arch, sampled_arch_datast_idx = sampler.sample_arch(batch_statics_dict, sample_size, dataset, kl_thred, args.sampler.max_trails, args.sampler.force_uniform)
    sampled_arch_datast_idx = [v for _, v in enumerate(sampled_arch_datast_idx) if v != None]
    if writer:
        writer.add_scalar('{}/number_sampled_archs'.format(flag), len(sampled_arch_datast_idx), it)

    # complement
    noisy_len = args.sampler.sample_size - len(sampled_arch_datast_idx)
    noisy_samples = random.choices(list(range(len(dataset))), k=noisy_len)
    sampled_arch_datast_idx += noisy_samples

    random.shuffle(sampled_arch_datast_idx)  # in_place
    assert len(sampled_arch_datast_idx) == args.sampler.sample_size, 'Not enough sampled dataset'

    sampled_trainset = SplitSubet(dataset, sampled_arch_datast_idx)
    sampled_dataloader = torch.utils.data.DataLoader(
        sampled_trainset,
        batch_size=args.sampler.search_size,  # train on sampled dataset, but the sampler batchsize
        num_workers=args.data_loader_workers,
        pin_memory=True)

    model.eval()

    results_tpk1 = []
    results_tpk5 = []
    for _, batch in enumerate(sampled_dataloader):
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes, rank = batch
        arch_feature = arch_feature.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, total_embedding_list = model(arch_feature, tier_feature)
        prob = F.softmax(output, dim=1)

        classify_tier_emb_by_pred(total_embedding_list, tier_list, output)
        classify_tier_counts_by_pred(params, flops, n_nodes, tier_list, output, args.bins)

        # find best pred arch
        val, index = torch.topk(prob, k=1, dim=1)
        index = index.squeeze(dim=1)
        val = val.squeeze(dim=1)
        idx = torch.where(index == 0)   # tier 1 对应的下标
        t1_prob = val[idx]
        t1_test_acc = test_acc[idx]
        t1_rank = rank[idx]

        tpk5_prob, tpk5_idx = torch.topk(t1_prob, k=5)
        tpk5_rank = t1_rank[tpk5_idx]
        tpk5_test_acc = t1_test_acc[tpk5_idx]
        tpk5_best_rank = min(tpk5_rank)
        tpk5_best_test_acc = max(tpk5_test_acc)

        tpk1_prob, tpk1_idx = torch.topk(t1_prob, k=1)
        tpk1_rank = t1_rank[tpk1_idx]
        tpk1_test_acc = t1_test_acc[tpk1_idx]

        results_tpk1.append((tpk1_test_acc.item(), tpk1_rank.item()))
        results_tpk5.append((tpk5_best_test_acc.item(), tpk5_best_rank.item()))
        
    (best_acc_at1, best_rank_at1) = sorted(results_tpk1, key=lambda item:item[0], reverse=True)[0]
    (best_acc_at5, best_rank_at5) = sorted(results_tpk5, key=lambda item:item[0], reverse=True)[0]

    best_rank_percentage_at1 = best_rank_at1/len(dataset)
    best_rank_percentage_at5 = best_rank_at5/len(dataset)


    sample_time = time.time() - sample_start
    # logger.info('[{}][iter:{:2d}] Time: {:.2f} Test Acc@top1: {:.8f} Rank: {:5d}(top{:6.2%})'.format(
    #         flag, it-args.ranker_epochs,
    #         sample_time,
    #         best_acc_at1,  
    #         best_rank_at1,
    #         best_rank_percentage_at1))
    logger.info('[{}][iter:{:2d}] Time: {:.2f} Test Acc@top5: {:.8f} Rank: {:5d}(top {:6.2%})'.format(
            flag, it-args.ranker_epochs,
            sample_time,
            best_acc_at5,
            best_rank_at5,
            best_rank_percentage_at5))

    batch_statics_dict = get_batch_statics(tier_list)

    return batch_statics_dict, best_acc_at1, best_rank_at1, best_acc_at5, best_rank_at5