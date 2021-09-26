import torch
import time
from tqdm import tqdm

from architecture import Bucket
from utils.metric import AverageMeter, compute_accuracy


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

def get_tier_emb(tier_list:list):
    tier_emb_list = []
    for item in tier_list:
        tier_emb_list.append(item.get_bucket_emb())
        
    return torch.cat(tier_emb_list, dim=0)


def classsify_tier_emb(total_embedding_list, tier_list, target):
    for i in range(len(total_embedding_list)):
        idx = torch.where(target[:,i] == 1)
        tier_emb = total_embedding_list[i][idx]
        tier_list[i].updata_bucket_emb(tier_emb)


def train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler,
                device, args, logger, writter, epoch):
    epoch_start = time.time()
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    
    model.train()
    tier_list = init_tier_list(args)

    for iter, batch in enumerate(train_dataloader):
        batch_start = time.time()
        total_iter = epoch * len(train_dataloader) + iter
        
        arch_feature, val_acc, test_acc, params, flops = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)

        score = val_acc / (params * flops + 1e-9)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.batch_size)
        target = target.cuda(device)

        tier_feature = get_tier_emb(tier_list).cuda(device)
        # 在此之前的tensor的grad都不跟踪

        optimizer.zero_grad()
        
        output, total_embedding_list = model(arch_feature, tier_feature)  # arch shape [256,19,7,7], tier_feature [5,19,512],后者detach
        
        loss = criterion(output, target)
        writter.add_scalar('train/iter_loss', loss, total_iter)
        
        loss.backward()
        
        lr_scheduler.update_lr()  # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        writter.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'],
                           total_iter)
        optimizer.step()

        classsify_tier_emb(total_embedding_list, tier_list, target)

        acc = compute_accuracy(output, target)
        writter.add_scalar('train/batch_accuracy', acc, total_iter)
        
        b_sz = arch_feature.size(0)
        batch_time.update(time.time() - batch_start, n=1)
        batch_loss.update(loss, b_sz)
        batch_acc.update(acc, b_sz)
        
        all_iter = (args.ranker_epochs - args.start_epochs)*len(train_dataloader) -1
        logger.info('[Train][Epoch: {:2d}][Iter: {:4d}/{:4d}] Time: {:.2f} ({:.2f}) Acc: {:.4f} ({:.4f}) Loss: {:.6f} ({:.6f})'.format(
            epoch,
            total_iter, all_iter, 
            batch_time.val, batch_time.avg, 
            batch_acc.val, batch_acc.avg, 
            batch_loss.val, batch_loss.avg))

    epoch_time = time.time() - epoch_start
    logger.info('[Train][Epoch: {:2d}] Time: {:.2f} Epoch Acc: {:.4f} Epoch Loss: {:.6f}'.format(epoch, epoch_time, batch_acc.avg, batch_loss.avg))
    
    return batch_acc.avg, batch_loss.avg



