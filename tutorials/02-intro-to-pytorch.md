Starting with the basics in this notebook based on UvA’s [Tutorial
2](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html)

``` r
library(reticulate)
library(here)
```

    ## here() starts at C:/stats/uva-dl-tutorials

Here’s my python configuration for this project.

``` r
py_config()
```

    ## python:         C:/anaconda3/python.exe
    ## libpython:      C:/anaconda3/python311.dll
    ## pythonhome:     C:/anaconda3
    ## version:        3.11.3 | packaged by Anaconda, Inc. | (main, Apr 19 2023, 23:46:34) [MSC v.1916 64 bit (AMD64)]
    ## Architecture:   64bit
    ## numpy:          C:/anaconda3/Lib/site-packages/numpy
    ## numpy_version:  1.24.3
    ## 
    ## NOTE: Python version was forced by RETICULATE_PYTHON_FALLBACK

I wonder which of the standard Python libraries I have already? Let’s
check.

``` python
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm
```

All imported with no issues. Very nice. Thanks, conda. \<3

# The Basics of PyTorch

Now we’ll import torch.

``` python
import torch
print("Using torch", torch.__version__, "with CUDA", torch.cuda_version)
```

    ## Using torch 2.0.1 with CUDA 11.7

## Tensors

OK, now we’ll move on to some `tensor` stuff. In ML/DL context, tensors
are basically multidimensional arrays. If you want to go down a rabbit
hole about why they’re called tensors when tensors are something very
specific in mathematics, then you can [start
here](https://stats.stackexchange.com/a/198395). 🐇

``` python
x = torch.Tensor(2, 3, 4)
print(x)
```

    ## tensor([[[0.0000e+00, 8.1585e-25, 7.3288e-43, 1.6204e-18],
    ##          [7.3288e-43, 0.0000e+00, 0.0000e+00, 1.6204e-18],
    ##          [7.3288e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00]],
    ## 
    ##         [[0.0000e+00, 4.0638e-44, 0.0000e+00, 1.6204e-18],
    ##          [7.3288e-43, 4.2039e-44, 0.0000e+00, 1.6204e-18],
    ##          [7.3288e-43, 4.0638e-44, 3.9236e-44, 0.0000e+00]]])

Oooh, neat. Apparently memory is allocated when we use `torch.Tensor`
but the values it initializes with are those that have already been in
memory. I like that (I don’t know why, but I do).

Other ways to specify values for tensors:

-   `torch.zeros` - values filled with zeros  
-   `torch.ones` - values filled with ones  
-   `torch.rand` - values filled with samples drawn from a uniform
    distribution between 0 and 1  
-   `torch.randn` - values filled with samples drawn from a normal
    distribution with mean 0 and variance 1  
-   `torch.arange` - values are filled with N, N+1, N+2, …, M. The step
    can be non-integer, e.g.

``` python
torch.arange(0, 1, 0.05)
```

    ## tensor([0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000,
    ##         0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500,
    ##         0.9000, 0.9500])

We can use nested lists to specify the elements of a tensor directly.

``` python
x = torch.Tensor([[1, 2], [3, 4]])
print(x)
```

    ## tensor([[1., 2.],
    ##         [3., 4.]])

The methods to obtain the shape of a tensor are `size` and `shape`.

``` python
print("Shape:", x.shape)
```

    ## Shape: torch.Size([2, 2])

``` python
print("Size:", x.size())
```

    ## Size: torch.Size([2, 2])

## Tensor to Numpy and vice versa

We use the `from_numpy` to go from a numpy array to a tensor.

``` python
np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

print(tensor)
```

    ## tensor([[1, 2],
    ##         [3, 4]], dtype=torch.int32)

And `.numpy()` to go from a PyTorch tensor to a numpy array.

``` python
tensor = torch.arange(4)
np_arr = tensor.numpy()

print("I'm a", tensor)
```

    ## I'm a tensor([0, 1, 2, 3])

``` python
print("And here I am as an Numpy array: ", np_arr)
```

    ## And here I am as an Numpy array:  [0 1 2 3]

An important note from the tutorial:

> The conversion of tensors to numpy require the tensor to be on the
> CPU, and not the GPU (more on GPU support in a later section). In case
> you have a tensor on GPU, you need to call .cpu() on the tensor
> beforehand. Hence, you get a line like np_arr = tensor.cpu().numpy().

## Operations

Check the [PyTorch docs](https://pytorch.org/docs/stable/tensors.html#)
for the full set of tensor operations available.

We can either create new tensors using operations or use methods to
perform in-place operations that will modify the tensor. These usually
have an underscore postfix.

``` python
x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
print("x1 before:\n", x1)

# Now add to x1
```

    ## x1 before:
    ##  tensor([[0.8832, 0.4899, 0.5844],
    ##         [0.4281, 0.3085, 0.1028]])

``` python
x1.add_(x2)
```

    ## tensor([[1.4092, 0.6000, 0.6119],
    ##         [1.4230, 0.5273, 0.8290]])

``` python
print("x1 after:\n", x1)
```

    ## x1 after:
    ##  tensor([[1.4092, 0.6000, 0.6119],
    ##         [1.4230, 0.5273, 0.8290]])

To re-shape tensors we can use `view` and `permute` operations.

View will add on a row and column shape.

``` python
x = torch.arange(6)
print(x, "\nshapes into")
```

    ## tensor([0, 1, 2, 3, 4, 5]) 
    ## shapes into

``` python
x.view(2, 3)
```

    ## tensor([[0, 1, 2],
    ##         [3, 4, 5]])

Permute swaps the dimensions specified.

``` python
x = x.view(2, 3)
x.permute(0, 1)
```

    ## tensor([[0, 1, 2],
    ##         [3, 4, 5]])

You have to make sure you permute using the same number of dimensions as
the tensor.

Other operations covered in the tutorial are for matrix multiplication.

-   `torch.matmul` - performs matrix product based on the dimensions of
    the tensors. If both are 2-dim then it will be a standard matrix
    product. Higher dimensional inputs will use
    [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics).
    Can also be written as `a @ b`, which is similar to numpy.

-   `torch.mm` - matrix product over two matrices but doesn’t support
    broadcasting.

-   `torch.bmm` - Performs the matrix product with a support batch
    dimension. If the first tensor is of shape (b, n, m), and the second
    tensor (b, m, p), the output is of shape (b, n, p). Basically it
    uses the first dimension as an index and multiplies the matrices of
    dimension (n,m) from the first argument by the matrices of dimension
    (n,p) from the second argument at each index.

-   `torch.einsum` - Performs matrix multiplications and more (i.e. sums
    of products) using the Einstein summation convention.

Shneat. I gotta try some of these batch mm examples to make sense of
that.
