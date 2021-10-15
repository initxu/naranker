import torch
import time
import copy

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy

from .train_utils import *

def train_epoch_201(model, train_dataloader, criterion, rank_reg, top_reg, optimizer, lr_scheduler,
                device, args, logger, writter, epoch, flag):

    epoch_start = time.time()

    all_iter = (args.ranker_epochs - args.start_epochs)*len(train_dataloader) - 1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    
    model.train()

    distri_list = []
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)

    for it, batch in enumerate(train_dataloader):
        batch_start = time.time()
        total_iter = epoch * len(train_dataloader) + it
        
        (arch_feature, val_acc, test_acc, params, flops, edges_type_counts, rank) = batch[args.network_type]
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        edges_type_counts = edges_type_counts.float().cuda(device)

        assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        if args.strategy == 'multi_obj':
            score = val_acc / (params * flops + 1e-9)  # shape = [batchsize]
        if args.strategy == 'val_acc':
            score = val_acc

        target = get_target(score, args.ranker.n_tier, args.train_batch_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, total_embedding_list = model(arch_feature, tier_feature)  # arch shape [256,19,7,7], tier_feature [5,19,512],后者detach
        
        rank_reg_term = args.reg.ranking_reg_factor * rank_reg(output, target)
        top_reg_term = args.reg.top_reg_factor * top_reg(output, score)
        loss = criterion(output, target) + rank_reg_term  + top_reg_term
        
        writter.add_scalar('{}/iter_loss'.format(flag), loss, total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('{}/iter_lr'.format(flag), optimizer.param_groups[0]['lr'], total_iter)
        optimizer.step()

        classify_tier_emb_by_target(total_embedding_list, tier_list, target)
        classify_tier_counts_by_target_201(params, flops, edges_type_counts, tier_list, target, args.bins)
        batch_statics_dict = get_batch_statics(tier_list)
        distri_list.append(batch_statics_dict)
        
        for k in batch_statics_dict:
            candi_dic = compare_kl_div(copy.deepcopy(batch_statics_dict[k]))
            writter.add_scalars('{}/{}_div'.format(flag, k), candi_dic, total_iter)

        acc = compute_accuracy(output, target)
        writter.add_scalar('{}/iter_accuracy'.format(flag), acc, total_iter)
        
        b_sz = arch_feature.size(0)
        batch_time.update(time.time() - batch_start, n=1)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        
        logger.info('[{}][Epoch:{:2d}][Iter:{:4d}/{:4d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f})'.format(
            flag, epoch,
            total_iter, all_iter, 
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg))

    epoch_time = time.time() - epoch_start
    logger.info('[{}][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f}'.format(flag, epoch, epoch_time, batch_acc.avg, batch_loss.avg))
    
    return batch_acc.avg, batch_loss.avg, distri_list

def validate_201(model, val_dataloader, criterion, device, args, logger, epoch, flag):
    epoch_start = time.time()

    total_iter = len(val_dataloader)-1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()

    model.eval()

    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    for it, batch in enumerate(val_dataloader):
        batch_start = time.time()
        (arch_feature, val_acc, test_acc, params, flops, edges_type_counts, rank) = batch[args.network_type]
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        edges_type_counts = edges_type_counts.float().cuda(device)

        assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        if args.strategy == 'multi_obj':
            score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        if args.strategy == 'val_acc':
            score = val_acc

        target = get_target(score, args.ranker.n_tier, args.val_batch_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, total_embedding_list = model(arch_feature, tier_feature)
        
        loss = criterion(output, target)

        classify_tier_emb_by_target(total_embedding_list, tier_list, target)

        acc = compute_accuracy(output, target)
        
        b_sz = arch_feature.size(0)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        batch_time.update(time.time() - batch_start, n=1)
        
        logger.info('[{}][Epoch:{:2d}][Iter:{:2d}/{:2d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f})'.format(
            flag, epoch,
            it, total_iter,
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg))
        
    epoch_time = time.time() - epoch_start
    logger.info('[{}][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f}'.format(flag, epoch, epoch_time, batch_acc.avg, batch_loss.avg))
        
    return batch_acc.avg, batch_loss.avg
