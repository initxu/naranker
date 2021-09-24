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


def train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler, device, args, logger):
    model.train()
    total_loss = 0
    start = time.time()
    
    for batch in tqdm(train_dataloader, mininterval=2, desc='[Training]: ', leave=False):

        arch_feature, val_acc, test_acc, params, flops = batch
        arch_feature.float().cuda(device)
        val_acc = torch.tensor(val_acc, dtype=torch.float).cuda(device)
        params = torch.tensor(params, dtype=torch.float).cuda(device)
        flops = torch.tensor(flops, dtype=torch.float).cuda(device)

        score = val_acc / (params * flops)  # 假设shape = [batch]
        target = get_target(score, args.ranker.n_tier, args.batch_size)
        target.cuda(device)

        # ################注意判断在模型输入之前，梯度是否跟踪

        output, total_embedding_list = model(arch_feature, tier_feature)            # arch shape [256,19,7,7], tier_feature [5,19,512]
        loss = criterion(output, target)
        loss.backward()
        
        lr_scheduler.update_lr()    # 由于初始lr是由lr_scheduler设定，因此更新学习率在前
        logger.info('Training lr {}'.format(optimizer.param_groups[0]['lr']))
        optimizer.step()


