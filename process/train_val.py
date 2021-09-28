import torch
import time

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy

from .train_utils import *

def train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler,
                device, args, logger, writter, epoch):

    epoch_start = time.time()

    all_iter = (args.ranker_epochs - args.start_epochs)*len(train_dataloader) - 1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    
    model.train()

    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)

    for iter, batch in enumerate(train_dataloader):
        batch_start = time.time()
        total_iter = epoch * len(train_dataloader) + iter
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.batch_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, total_embedding_list = model(arch_feature, tier_feature)  # arch shape [256,19,7,7], tier_feature [5,19,512],后者detach
        
        loss = criterion(output, target)
        writter.add_scalar('train/iter_loss', loss, total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], total_iter)
        optimizer.step()

        classify_tier_emb_by_target(total_embedding_list, tier_list, target)
        classify_tier_counts_by_target(params, flops, n_nodes, tier_list, target, args.bins)
        # batch_statics_dict = get_batch_statics(tier_list)

        acc = compute_accuracy(output, target)
        writter.add_scalar('train/iter_accuracy', acc, total_iter)
        
        b_sz = arch_feature.size(0)
        batch_time.update(time.time() - batch_start, n=1)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        
        logger.info('[Train][Epoch:{:2d}][Iter: {:5d}/{:5d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f})'.format(
            epoch,
            total_iter, all_iter, 
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg))

    epoch_time = time.time() - epoch_start
    logger.info('[Train][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f}'.format(epoch, epoch_time, batch_acc.avg, batch_loss.avg))
    
    return batch_acc.avg, batch_loss.avg

def validate(model, val_dataloader, criterion, device, args, logger, epoch):
    epoch_start = time.time()
    
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()

    model.eval()
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    for iter, batch in enumerate(val_dataloader):
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.batch_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, total_embedding_list = model(arch_feature, tier_feature)
        
        loss = criterion(output, target)

        classify_tier_emb_by_pred(total_embedding_list, tier_list, output)  # note, validate according to prediction

        acc = compute_accuracy(output, target)
        
        b_sz = arch_feature.size(0)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        
    epoch_time = time.time() - epoch_start
    logger.info('[Validate][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f}'.format(epoch, epoch_time, batch_acc.avg, batch_loss.avg))
        
    return batch_acc.avg, batch_loss.avg
