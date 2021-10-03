import os
import torch
import shutil


def save_checkpoint(save_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    epoch,
                    is_best=False):
    save_state = {
        'epoch': epoch + 1, # 读取时，从epoch+1开始更新学习率
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.__dict__   # 'lr_mul': 0.05, 'd_model': 256, 'n_warmup_steps': 300, 'n_steps': 0 step是自动更新的，直接load进去开始训即可接上
    }

    best_model_path = os.path.join(
        os.path.dirname(save_path),
        'ckp_best.pth.tar')

    with open(save_path, 'wb') as f:
        torch.save(save_state, f, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(save_path, best_model_path)