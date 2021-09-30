import torch
import time
import copy

from architecture import Bucket
from utils.metric import compute_accuracy

from .train_utils import *

def sampler_train_iter(model, train_dataloader, criterion, optimizer, lr_scheduler,
                device, args, logger, writter, iter, flag):

    model.train()

    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)

    for _, batch in enumerate(train_dataloader):
        batch_start = time.time()
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.sampler.sample_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, total_embedding_list = model(arch_feature, tier_feature)  # arch shape [256,19,7,7], tier_feature [5,19,512],后者detach
        
        loss = criterion(output, target)
        writter.add_scalar('{}/iter_loss'.format(flag), loss, iter)
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('{}/iter_lr'.format(flag), optimizer.param_groups[0]['lr'], iter)
        optimizer.step()

        classify_tier_emb_by_target(total_embedding_list, tier_list, target)
        classify_tier_counts_by_target(params, flops, n_nodes, tier_list, target, args.bins)
        batch_statics_dict = get_batch_statics(tier_list)
        
        for k in batch_statics_dict:
            candi_dic = compare_kl_div(copy.deepcopy(batch_statics_dict[k]))
            writter.add_scalars('{}/{}_div'.format(flag, k), candi_dic, iter)

        acc = compute_accuracy(output, target)
        writter.add_scalar('{}/iter_accuracy'.format(flag), acc, iter)
        
        batch_end = time.time() - batch_start

        logger.info('[{}][Iter: {:5d}/{:5d}] Time: {:.2f} Acc: {:.4f} Loss: {:.6f}'.format(
            flag,
            iter, args.sampler_epochs, 
            batch_end,
            acc,
            loss))

    return acc, loss, batch_statics_dict

def sampler_validate(model, val_dataloader, criterion, device, args, logger, iter, flag):
    iter_start = time.time()

    model.eval()
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    for _, batch in enumerate(val_dataloader):
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.sampler.sample_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, total_embedding_list = model(arch_feature, tier_feature)
        
        loss = criterion(output, target)

        classify_tier_emb_by_pred(total_embedding_list, tier_list, output)  # note, validate according to prediction

        acc = compute_accuracy(output, target)
        
    iter_time = time.time() - iter_start
    logger.info('[{}][Iter:{:2d}] Time: {:.2f} Iter Acc: {:.4f} Iter Loss: {:.6f}'.format(flag, iter, iter_time, acc, loss))
        
    return acc, loss