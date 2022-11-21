import torch
import time
import copy

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy, compute_lat_acc

from .train_utils import *

def train_epoch_201_latency(model, train_dataloader, criterion, aux_criterion, optimizer, lr_scheduler,
                device, args, logger, writter, epoch, flag, latency_criterion):

    epoch_start = time.time()

    all_iter = (args.ranker_epochs - args.start_epochs)*len(train_dataloader) - 1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_lat_rmse = AverageMeter()
    batch_lat_acc5 = AverageMeter()
    batch_lat_acc10 = AverageMeter()
    
    model.train()

    distri_list = []
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)

    for it, batch in enumerate(train_dataloader):
        batch_start = time.time()
        total_iter = epoch * len(train_dataloader) + it
        
        (arch_feature, val_acc, test_acc, params, flops, edges_type_counts, rank, latency, label) = batch[args.network_type]
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        edges_type_counts = edges_type_counts.float().cuda(device)
        label = label.cuda(device)
        latency = (latency*1000).float().cuda(device)

        # assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        # if args.strategy == 'multi_obj':
        #     score = val_acc / (params * flops + 1e-9)  # shape = [batchsize]
        # if args.strategy == 'val_acc':
        #     score = val_acc

        # target = get_target(score, args.ranker.n_tier, args.train_batch_size)
        # target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, enc_output, val_acc_pred, latency_pred = model(arch_feature, tier_feature)  # latency_pred.shape =128*1

        loss = criterion(output, label)
        loss_latency = latency_criterion(latency_pred.squeeze(-1), latency)
        loss += args.latency_factor * loss_latency
        if aux_criterion:
            aux_loss = aux_criterion(val_acc_pred.squeeze(1), val_acc)
            loss += args.loss_factor * aux_loss
        
        writter.add_scalar('{}/iter_loss'.format(flag), loss, total_iter)
        writter.add_scalar('{}/iter_lat_loss'.format(flag), loss_latency, total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('{}/iter_lr'.format(flag), optimizer.param_groups[0]['lr'], total_iter)
        optimizer.step()

        classify_enc_emb_by_target(enc_output.clone().detach(), tier_list, label)
        classify_tier_counts_by_target_201(params, flops, edges_type_counts, tier_list, label, args.bins)
        batch_statics_dict = get_batch_statics(tier_list)
        distri_list.append(batch_statics_dict)
        
        for k in batch_statics_dict:
            candi_dic = compare_kl_div(copy.deepcopy(batch_statics_dict[k]))
            writter.add_scalars('{}/{}_div'.format(flag, k), candi_dic, total_iter)

        acc = compute_accuracy(output, label)
        latency_rmse = torch.sqrt(loss_latency.detach())
        lat_acc5 = compute_lat_acc(latency_pred, latency, threshold=0.05)
        lat_acc10 = compute_lat_acc(latency_pred, latency, threshold=0.1)
        # writter.add_scalar('{}/iter_lat_rmse'.format(flag), latency_rmse, total_iter)
        # writter.add_scalar('{}/iter_lat_acc5'.format(flag), lat_acc5, total_iter)
        # writter.add_scalar('{}/iter_lat_acc10'.format(flag), lat_acc10, total_iter)
        # writter.add_scalar('{}/iter_accuracy'.format(flag), acc, total_iter)
        
        b_sz = arch_feature.size(0)
        batch_time.update(time.time() - batch_start, n=1)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        batch_lat_rmse.update(latency_rmse, b_sz)
        batch_lat_acc5.update(lat_acc5, b_sz)
        batch_lat_acc10.update(lat_acc10, b_sz)
        
        logger.info('[{}][Epoch:{:2d}][Iter:{:4d}/{:4d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f}) \
            RMSE: {:.4f} ({:.4f}) \
            LatAcc5: {:.4f} ({:.4f}) \
            LatAcc10: {:.4f} ({:.4f})'.format(
            flag, epoch,
            total_iter, all_iter, 
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg,
            batch_lat_rmse.val, batch_lat_rmse.avg,
            batch_lat_acc5.val, batch_lat_acc5.avg,
            batch_lat_acc10.val, batch_lat_acc10.avg
            ))

    epoch_time = time.time() - epoch_start
    logger.info('[{}][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f} Epoch RMSE: {:.4f} Epoch LatAcc5: {:.4f} Epoch LatAcc10: {:.4f}'.format(\
        flag, epoch, epoch_time, batch_acc.avg, batch_loss.avg, batch_lat_rmse.avg, batch_lat_acc5.avg, batch_lat_acc10.avg))
    
    writter.add_scalar('{}/epoch_lat_rmse'.format(flag), batch_lat_rmse.avg, epoch)
    writter.add_scalar('{}/epoch_lat_acc5'.format(flag), batch_lat_acc5.avg, epoch)
    writter.add_scalar('{}/epoch_lat_acc10'.format(flag), batch_lat_acc10.avg, epoch)
    return batch_acc.avg, batch_loss.avg, distri_list

def validate_201_latency(model, val_dataloader, criterion, aux_criterion, device, args, logger, epoch, flag, latency_criterion, writter):
    epoch_start = time.time()

    total_iter = len(val_dataloader)-1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_lat_rmse = AverageMeter()
    batch_lat_acc5 = AverageMeter()
    batch_lat_acc10 = AverageMeter()

    model.eval()

    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    for it, batch in enumerate(val_dataloader):
        batch_start = time.time()
        (arch_feature, val_acc, test_acc, params, flops, edges_type_counts, rank, latency, label) = batch[args.network_type]
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        edges_type_counts = edges_type_counts.float().cuda(device)
        label = label.cuda(device)
        latency = (latency*1000).float().cuda(device)

        # assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        # if args.strategy == 'multi_obj':
        #     score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        # if args.strategy == 'val_acc':
        #     score = val_acc

        # target = get_target(score, args.ranker.n_tier, args.val_batch_size)
        # target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, enc_output, val_acc_pred, latency_pred = model(arch_feature, tier_feature)

        loss = criterion(output, label)
        loss_latency = latency_criterion(latency_pred.squeeze(-1), latency)
        loss += args.latency_factor * loss_latency
        if aux_criterion:
            aux_loss = aux_criterion(val_acc_pred.squeeze(1), val_acc)
            loss += args.loss_factor * aux_loss

        classify_enc_emb_by_target(enc_output.clone().detach(), tier_list, label)

        acc = compute_accuracy(output, label)
        latency_rmse = torch.sqrt(loss_latency.detach())
        lat_acc5 = compute_lat_acc(latency_pred, latency, threshold=0.05)
        lat_acc10 = compute_lat_acc(latency_pred, latency, threshold=0.1)
        
        b_sz = arch_feature.size(0)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        batch_time.update(time.time() - batch_start, n=1)
        
        batch_lat_rmse.update(latency_rmse, b_sz)
        batch_lat_acc5.update(lat_acc5, b_sz)
        batch_lat_acc10.update(lat_acc10, b_sz)
        
        logger.info('[{}][Epoch:{:2d}][Iter:{:2d}/{:2d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f}) Epoch RMSE: {:.4f} ({:.4f}) Epoch LatAcc5: {:.4f} ({:.4f}) Epoch LatAcc10: {:.4f} ({:.4f})'.format(
            flag, epoch,
            it, total_iter,
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg,
            batch_lat_rmse.val, batch_lat_rmse.avg,
            batch_lat_acc5.val, batch_lat_acc5.avg,
            batch_lat_acc10.val, batch_lat_acc10.avg
            ))
        
    epoch_time = time.time() - epoch_start
    logger.info('[{}][Epoch:{:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f} Epoch RMSE: {:.4f} Epoch LatAcc5: {:.4f} Epoch LatAcc10: {:.4f}'.format(flag, epoch, epoch_time, batch_acc.avg, batch_loss.avg, batch_lat_rmse.avg, batch_lat_acc5.avg, batch_lat_acc10.avg))
    writter.add_scalar('{}/epoch_lat_rmse'.format(flag), batch_lat_rmse.avg, epoch)
    writter.add_scalar('{}/epoch_lat_acc5'.format(flag), batch_lat_acc5.avg, epoch)
    writter.add_scalar('{}/epoch_lat_acc10'.format(flag), batch_lat_acc10.avg, epoch)
    return batch_acc.avg, batch_loss.avg, batch_lat_rmse.avg, batch_lat_acc5.avg, batch_lat_acc10.avg
