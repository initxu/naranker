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
        'lr_scheduler': lr_scheduler.__dict__
    }

    best_model_path = os.path.join(
        os.path.dirname(save_path),
        'best_{}'.format(os.path.basename(save_path)))

    with open(save_path, 'wb') as f:
        torch.save(save_state, f, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(save_path, best_model_path)