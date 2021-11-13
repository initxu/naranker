import torch
import time
import copy

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy, compute_kendall_tau

from .train_utils import *

def train_epoch(model, train_dataloader, criterion, aux_criterion, optimizer, lr_scheduler,
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
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes, rank, label = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)
        label = label.cuda(device)

        # assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        # if args.strategy == 'multi_obj':
        #     score = val_acc / (params * flops + 1e-9)  # shape = [batchsize]
        # if args.strategy == 'val_acc':
        #     score  = val_acc

        # target = get_target(score, args.ranker.n_tier, args.batch_size)
        # target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, enc_output, val_acc_pred = model(arch_feature, tier_feature)
        
        # if (epoch ==34 and total_iter == 559): #epoch ==0 and total_iter == 15: #: # or (epoch == 9 and total_iter == 159)or (epoch ==19 and total_iter == 319)
        #     a = {'output':output.clone().detach(),
        #         'enc_output':enc_output.clone().detach(),
        #         'target':label}
        #     torch.save(a, "{}_feature_true.pt".format(epoch))
        
        loss = criterion(output, label)

        if aux_criterion:
            aux_loss = aux_criterion(val_acc_pred.squeeze(1), val_acc)
            loss += args.loss_factor * aux_loss

        writter.add_scalar('{}/iter_loss'.format(flag), loss, total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('{}/iter_lr'.format(flag), optimizer.param_groups[0]['lr'], total_iter)
        optimizer.step()

        classify_enc_emb_by_target(enc_output.clone().detach(), tier_list, label)
        classify_tier_counts_by_target(params, flops, n_nodes, tier_list, label, args.bins)
        batch_statics_dict = get_batch_statics(tier_list)
        distri_list.append(batch_statics_dict)
        
        for k in batch_statics_dict:
            candi_dic = compare_kl_div(copy.deepcopy(batch_statics_dict[k]))
            writter.add_scalars('{}/{}_div'.format(flag, k), candi_dic, total_iter)

        acc = compute_accuracy(output, label)
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

def validate(model, val_dataloader, criterion, aux_criterion, device, args, logger, epoch, flag):
    epoch_start = time.time()

    total_iter = len(val_dataloader)-1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_ktau = AverageMeter()

    model.eval()
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    for it, batch in enumerate(val_dataloader):
        batch_start = time.time()
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes, rank, label= batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)
        label = label.cuda(device)

        # assert args.strategy in ['multi_obj', 'val_acc'], 'Wrong strategy'
        # if args.strategy == 'multi_obj':
        #     score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        # if args.strategy == 'val_acc':
        #     score  = val_acc

        # target = get_target(score, args.ranker.n_tier, args.batch_size)
        # target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, enc_output, val_acc_pred = model(arch_feature, tier_feature)
        
        loss = criterion(output, label)

        # a = {   'pred_val': val_acc_pred.clone().detach(),
        #         'rank':     rank.clone().detach(),
        #         'label': label
        #     }
        # torch.save(a, "{}_{}_rank_point.pt".format(epoch, it))

        if aux_criterion:
            aux_loss = aux_criterion(val_acc_pred.squeeze(1), val_acc)
            loss += args.loss_factor * aux_loss

        classify_enc_emb_by_target(enc_output.clone().detach(), tier_list, label)

        acc = compute_accuracy(output, label)
        
        # ktau = compute_kendall_tau(val_acc_pred, val_acc)
        
        b_sz = arch_feature.size(0)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        # batch_ktau.update(ktau, n=1)
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
