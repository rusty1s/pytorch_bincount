import torch
import numpy as np

if torch.cuda.is_available():
    import bincount_cuda


def bincount(src, size=None):
    """Counts the number of occurrences of each value in a non-negative tensor.

    Args:
        src (Tensor): The input tensor.
        size (int, optional): The maximum number of bins for the output array.
            (default: `None`)

    :rtype: :class:`LongTensor`
    """

    src = src.contiguous().view(-1)

    if src.is_cuda:
        size = src.max() + 1 if size is None else size
        return bincount_cuda.bincount(src, size)
    else:
        out = np.bincount(src.long().numpy(), minlength=size)
        return torch.from_numpy(out)
