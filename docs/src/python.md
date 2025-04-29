# Python support

## Installation
The recommended way to install MuSink in python is via `juliacall`.
First, run `pip install juliacall`.
Afterwards, start a python shell and type
```python
from juliacall import Main as jl
jl.seval("using Pkg; Pkg.add(url = \"https://github.com/tscode/MuSink.jl.git\")")
```
This should take care of all necessary installations.
From now on, you can run
```python
from juliacall import Main as jl
jl.seval("using MuSink")
ms = jl.MuSink
```
to access all functionality of the package via `ms.[function name]`.

## Usage
One important difference between `julia` and `python` is that the former advocates the usage of the symbol `!` as a postfix to function names that alter its arguments.
Since this results in invalid identifiers in python, the postfix `_b` is by used in `juliacall` ('b' for 'bang').
For example, the function `MuSink.step!`, which implements the basic Sinkhorn step, has to be called via `ms.step_b` in python.
A complete code example:
```python
# import numpy
import numpy as np

# import MuSink
from juliacall import Main as jl
jl.seval("using MuSink")
ms = jl.MuSink

# Create the cost tree
root = ms.Tree.Root(1)
child2 = ms.Tree.new_child_b(root, 2)
child3 = ms.Tree.new_child_b(root, 3)

# Define the target images
img1 = np.zeros((3, 7))
img1[:,0] = 1

img2 = np.zeros((3, 7))
img2[:,3] = 1

img3 = np.zeros((3, 7))
img3[:,6] = 1

# Create a problem and workspace
problem = ms.Problem(root, [img1, img2, img3], cost = ms.Lp(3, 7, p = 2), penalty = ms.TotalVariation())
workspace = ms.Workspace(problem, eps = 0.01, rho = np.inf)

# Do 100 Sinkhorn steps
ms.step_b(workspace, 100)

# Look at different properties of the workspace
# Arrays can be converted to numpy via `np.array(...)` on the results of these functions
ms.target(workspace, 1)
ms.marginal(workspace, 1)
ms.transport(workspace, 1, 3, (1, 1))
```

A more in-depth usage tutorial that guides you through the main functionality of MuSink is provided [here](Usage).

## GPU support
If you want to enable GPU support, you have to install and import the suitable package.
For example, for CUDA support, you will have to run
```python
jl.seval("using Pkg; Pkg.add(\"CUDA\")") # installation
jl.seval("using MuSink, CUDA")           # import
```
Then, you should be able to add the argument `atype = "cuda64"` when creating a workspace.

The same procedure should work for all other extensions of MuSink.jl.
