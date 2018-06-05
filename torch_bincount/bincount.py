import torch
import numpy as np

if torch.cuda.is_available():
    import bincount_cuda


def bincount(src, size=None):
    if src.is_cuda:
        size = src.max() + 1 if size is None else size
        return bincount_cuda.bincount(src, size)
    else:
        out = np.bincount(src.long().numpy(), minlength=size)
        return torch.from_numpy(out)
