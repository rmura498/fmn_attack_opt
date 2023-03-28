from typing import Optional
from functools import partial

import torch
from torch import Tensor


def difference_of_logits(logits: Tensor, labels: Tensor, labels_infhot: Optional[Tensor] = None) -> Tensor:
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return class_logits - other_logits


def difference_of_logits_ratio(logits: Tensor, labels: Tensor, labels_infhot: Optional[Tensor] = None,
                               targeted: bool = False, epsilon: float = 0) -> Tensor:
    """Difference of Logits Ratio from https://arxiv.org/abs/2003.01690. This version is modified such that the DLR is
    always positive if argmax(logits) == labels"""
    logit_dists = difference_of_logits(logits=logits, labels=labels, labels_infhot=labels_infhot)

    if targeted:
        top4_logits = torch.topk(logits, k=4, dim=1).values
        logit_normalization = top4_logits[:, 0] - (top4_logits[:, -2] + top4_logits[:, -1]) / 2
    else:
        top3_logits = torch.topk(logits, k=3, dim=1).values
        logit_normalization = top3_logits[:, 0] - top3_logits[:, -1]

    return (logit_dists + epsilon) / (logit_normalization + 1e-8)


def accuracy(model, samples, labels):
    preds = model(samples)
    acc = (preds.argmax(dim=1) == labels).float().mean()
    return acc.item()


def loss_fmn_fn(inputs, labels, model, multiplier, labels_infhot=None, logit_diff_func=None):
    logits = model(inputs)

    labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
    logit_diff_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

    logit_diffs = logit_diff_func(logits=logits)
    loss = (multiplier * logit_diffs)

    return -loss.sum().item(), logits
