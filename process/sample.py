import torch
import time
import copy
import random
import torch.utils.data

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy
from dataset import NASBenchDataset, SplitSubet

from .train_utils import *

def sampler_train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler,
                device, args, logger, writter, epoch, flag):

    epoch_start = time.time()

    all_iter = (args.sampler_epochs - args.ranker_epochs)*len(train_dataloader) - 1
    
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()

    model.train()

    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)

    for it, batch in enumerate(train_dataloader):
        batch_start = time.time()
        total_iter = (epoch-args.ranker_epochs)*len(train_dataloader) + it
        
        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.sampler.batch_size) # notice: sampler batchsize
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, total_embedding_list = model(arch_feature, tier_feature)  # arch shape [256,19,7,7], tier_feature [5,19,512],后者detach
        
        loss = criterion(output, target)
        writter.add_scalar('{}/iter_loss'.format(flag), loss, total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()
        writter.add_scalar('{}/iter_lr'.format(flag), optimizer.param_groups[0]['lr'], total_iter)
        optimizer.step()

        classify_tier_emb_by_target(total_embedding_list, tier_list, target)
        classify_tier_counts_by_target(params, flops, n_nodes, tier_list, target, args.bins)
        batch_statics_dict = get_batch_statics(tier_list)
        
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

    return batch_acc.avg, batch_loss.avg, batch_statics_dict


def evaluate(model, sampler, tier_list, batch_statics_dict, dataset : NASBenchDataset, epoch, args, writer, device, flag):

    sample_size = int(args.sampler.sample_size * (1 - args.sampler.noisy_factor))
    kl_thred = [
        args.sampler.flops_kl_thred, 
        args.sampler.params_kl_thred,
        args.sampler.n_nodes_kl_thred
    ]
    import pdb;pdb.set_trace()
    # sample from entire dataset
    sampled_arch, sampled_arch_datast_idx = sampler.sample_arch(batch_statics_dict, sample_size, dataset, kl_thred, args.sampler.max_trails)
    sampled_arch_datast_idx = [v for _, v in enumerate(sampled_arch_datast_idx) if v != None]
    writer.add_scalar('{}/number_sampled_archs'.format(flag), len(sampled_arch_datast_idx), epoch)

    # add noisy
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

    for it, batch in enumerate(sampled_dataloader):
        batch_start = time.time()

        arch_feature, val_acc, test_acc, params, flops, n_nodes = batch
        arch_feature = arch_feature.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)
        n_nodes = n_nodes.float().cuda(device)

        tier_feature = get_tier_emb(tier_list, device)
        assert not (torch.any(torch.isnan(tier_feature)) or torch.any(torch.isinf(tier_feature))), 'tier feature is nan or inf'

        output, total_embedding_list = model(arch_feature, tier_feature)

        classify_tier_emb_by_pred(total_embedding_list, tier_list, output)
        classify_tier_counts_by_pred(params, flops, n_nodes, tier_list, output, args.bins)

        # find best arch
        # TODO: 提取 test acc 和 对应的全局排名
        _, index = torch.topk(output, k=1, dim=1)
        index = index.squeeze(dim=1)
        idx = torch.where(index == 0)   # tier 1 对应的下标
        results = test_acc[idx]











#         sample_size = int(args.sampler.sample_size * (1-args.sampler.noisy_factor))
#         kl_thred = [args.sampler.flops_kl_thred, args.sampler.params_kl_thred, args.sampler.n_nodes_kl_thred]

#         # sample from trainset
#         sampled_arch, sampled_arch_datast_idx = sampler.sample_arch(batch_statics_dict, sample_size, trainset, kl_thred, args.sampler.max_trails)
#         sampled_arch_datast_idx = [v for _, v in enumerate(sampled_arch_datast_idx) if v != None]
#         tb_writer.add_scalar('{}/number_sampled_archs'.format(flag), len(sampled_arch_datast_idx), epoch)

#         # add noisy
#         noisy_len = args.sampler.sample_size - len(sampled_arch_datast_idx)
#         noisy_samples = random.choices(list(range(args.train_size)), k=noisy_len)
#         sampled_arch_datast_idx += noisy_samples

#         random.shuffle(sampled_arch_datast_idx) # in_place
#         assert len(sampled_arch_datast_idx) == args.sampler.sample_size, 'Not enough sampled dataset'

#         sampled_trainset = SplitSubet(dataset, sampled_arch_datast_idx)
#         sampled_train_dataloader = torch.utils.data.DataLoader(
#             sampled_trainset,
#             batch_size=args.sampler.batch_size,     # train on sampled dataset, but the sampler batchsize
#             num_workers=args.data_loader_workers,
#             pin_memory=True)

#         train_acc, train_loss, batch_statics_dict = sampler_train_epoch(ranker, sampled_train_dataloader, criterion, optimizer, lr_scheduler, device, args, logger, tb_writer, epoch, flag)
#         tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), train_acc, epoch)
#         tb_writer.add_scalar('{}/epoch_loss'.format(flag), train_loss, epoch)

#         if (epoch+1) % args.validate_freq == 0:
#             with torch.no_grad():
#                 flag = 'Sampler Validate'
#                 val_acc, val_loss = validate(ranker, val_dataloader, criterion, device, args, logger, epoch, flag)
#                 tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), val_acc, epoch)
#                 tb_writer.add_scalar('{}/epoch_loss'.format(flag), val_loss, epoch)