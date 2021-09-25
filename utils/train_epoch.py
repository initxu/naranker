import torch
import time
from tqdm import tqdm


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


def train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler, device, args, logger, writter,epoch):
    model.train()
    total_loss = 0
    start = time.time()

    iter = 0
    for batch in tqdm(train_dataloader, mininterval=2, desc='[Training]: ', leave=False):
        
        total_iter = epoch*len(train_dataloader) + iter
        arch_feature, val_acc, test_acc, params, flops = batch
        arch_feature = arch_feature.float().cuda(device)
        val_acc = val_acc.float().cuda(device)
        params = params.float().cuda(device)
        flops = flops.float().cuda(device)

        score = val_acc / (params * flops)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.batch_size)
        target = target.cuda(device)
        # 在此之前的tensor的grad都不跟踪

        # get_bucket

        optimizer.zero_grad()
        tier_feature = torch.randn(5,19,512).cuda(device)
        output, total_embedding_list = model(arch_feature, tier_feature)            # arch shape [256,19,7,7], tier_feature [5,19,512]
        loss = criterion(output, target)
        writter.add_scalar('ranker/loss', loss,total_iter)
        loss.backward()
        
        lr_scheduler.update_lr()    # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        # logger.info('Training with lr {}'.format(optimizer.param_groups[0]['lr']))
        writter.add_scalar('ranker/lr',optimizer.param_groups[0]['lr'],total_iter)
        optimizer.step()
        

        iter+=1

