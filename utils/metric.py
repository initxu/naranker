import torch
import torch.nn.functional as F


def compute_accuracy(output, target):
    batch_sz = target.size(0)  # both target and output are b_sz * n_class
    _, pred = output.topk(k=1, dim=1)
    _, label = target.topk(k=1, dim=1)
    correct = pred.eq(label).sum()

    return float(torch.true_divide(correct, batch_sz))


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_kendall_tau(a, b):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    assert len(a) == len(b), "Sequence a and b should have the same length while computing kendall tau."
    length = len(a)
    count = 0
    total = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            count += _sign(a[i] - a[j]) * _sign(b[i] - b[j])
            total += 1
    Ktau = count / total
    return Ktau


def t1_kendall_tau(output, score):
    output_log_prob = F.softmax(output, dim=1)
    prob_val, prob_idx = torch.topk(output_log_prob, k=1, dim=1)
    prob_val = prob_val.squeeze(dim=1)
    prob_idx = prob_idx.squeeze(dim=1)

    t1_idx = torch.where(prob_idx == 0)
    t1_pred_prob = prob_val[t1_idx]
    t1_gt_score = score[t1_idx]
    if t1_gt_score.size(0) == 0:
        t1_Ktau = None
    else:
        t1_Ktau = compute_kendall_tau(t1_pred_prob, t1_gt_score)

    return t1_Ktau


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