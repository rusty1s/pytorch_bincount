[pypi-image]: https://badge.fury.io/py/torch-bincount.svg
[pypi-url]: https://pypi.python.org/pypi/torch-bincount
[build-image]: https://travis-ci.org/rusty1s/pytorch_bincount.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_bincount
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_bincount/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_bincount?branch=master

# PyTorch BinCount

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

This package consists of a small extension library of a highly optimized `bincount` operation for the use in [PyTorch](http://pytorch.org/), which is missing in the main package.
The operation works on varying data types and is implemented both for CPU and GPU.

## Installation

Ensure that at least PyTorch 0.4.1 is installed and verify that `cuda/bin` and `cuda/install` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.1

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/install:...
```

Then run:

```
pip install torch-scatter torch-bincount
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_bincount/issues).

## Usage

```
torch_bincount.bincount(src, size=None) -> LongTensor
```

Counts the number of occurrences of each value in a non-negative tensor.

### Parameters

* **src** *(Tensor)* - The input tensor.
* **size** *(int, optional)* - The maximum number of bins for the output array. (default: `None`)

### Returns

* **out** *(LongTensor)* - The result of binning the input tensor.

### Example

```py
import torch
from torch_bincount import bincount

src = torch.tensor([2, 1, 1, 2, 4, 4, 2])
out = bincount(src)
```

```
print(out)
tensor([ 0,  2,  3,  0,  2])
```

## Running tests

```
python setup.py test
```
