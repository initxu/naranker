import torch
import copy
from tqdm import tqdm


def cal_target(score, n_tier, batch_sz):
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
            target[j, i].add_1

    return target


def train_epoch(model, train_dataloader, optimizer, lr_scheduler, device,
                args):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader,
                      mininterval=2,
                      desc='[Training]: ',
                      leave=False):

        arch_feature, val_acc, test_acc, params, flops, = batch
        arch_feature.float().cuda(device)
        val_acc = torch.tensor(val_acc, dtype=torch.float).cuda(device)
        params = torch.tensor(params, dtype=torch.float).cuda(device)
        flops = torch.tensor(flops, dtype=torch.float).cuda(device)

        score = val_acc / (params * flops)  # 假设shape = [batch]
        target = cal_target(score, args.ranker.n_tier, args.batch_size)
        target.cuda(device)