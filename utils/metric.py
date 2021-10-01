import torch

def compute_accuracy(output, target):
    batch_sz = target.size(0)   # both target and output are b_sz * n_class
    _, pred = output.topk(k=1,dim=1)
    _, label = target.topk(k=1,dim=1)
    correct = pred.eq(label).sum()

    return float(torch.true_divide(correct,batch_sz))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count