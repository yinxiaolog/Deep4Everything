import torch


def accuracy(pred: torch.Tensor, label: torch.Tensor):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = pred.argmax(dim=1)

    ans = pred.type(label.dtype) == label
    return float(ans.type(label.dtype).sum())
