from itertools import product

import pytest
from torch_bincount import bincount

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_bincount(dtype, device):
    src = tensor([2, 1, 1, 2, 4, 4, 2], dtype, device)
    out = bincount(src)

    assert out.tolist() == [0, 2, 3, 0, 2]
