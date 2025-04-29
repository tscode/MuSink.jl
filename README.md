
# MuSink

This package efficiently implements unbalanced multi-marginal optimal transport
(UMOT) between measures on uniform two dimensional grids (i.e., images). It is
written in [julia](https://julialang.org) but can also conveniently be accessed
from both [R](https://www.r-project.org) and [python](https://www.python.org)
(see below).

The problem formulation and the algorithmic approach in this package is
inspired by the publication [Beier, Lindheim, Neumayer, and Steidl
(2022)](https://arxiv.org/abs/2103.10854), which we use as main reference. Some
of the core features of MuSink are:

* very efficient implementation of Sinkhorn updates taylored to image data,
* supports both exponential (fast but instable) and logarithmic (stable but slower) domain,
* threading support,
* CUDA support (with custom kernels for the logdomain),
* Metal support (GPU of Apple M1 and M2 processors),
* oneAPI support (Intel GPUs)
* low memory footprint,
* general tree-based cost topologies,
* easily extendable from outside the package.

## Installation

This package is not yet registered as an official julia package. To install
it, run
```julia
using Pkg
Pkg.add(url = "https://gitlab.gwdg.de/staudt1/musink")
```
in a julia shell. This command will take care of installing all necessary
dependencies. Note that only julia versions `1.6` and newer are tested.

## Usage

For an overview of the basic functionality of MuSink you can consult [this
tutorial](https://gitlab.gwdg.de/staudt1/musink/-/wikis/Usage).

### Python

Instructions for installation and usage of MuSink in python are provided
[here](https://gitlab.gwdg.de/staudt1/musink/-/wikis/Python).

### R

Documentation of how to install and use MuSink from `R` will follow soon.

## Optional dependencies

MuSink supports the packages `LoopVectorization`, `CUDA`, `Metal`, and `oneAPI`
as optional dependencies. Installing and `using` them will unlock additional
functionality of MuSink.

* LoopVectorization: Sinkhorn steps on the CPU in the logdomain will be faster,
  at the cost of a higher first execution time.
* CUDA: Workspace constructors (see the [tutorial](https://gitlab.gwdg.de/staudt1/musink/-/wikis/Usage))
  take the optional keyword argument `atype = :cuda32` or `:cuda64`.
  Sinkhorn steps are then performed on cuda-enabled GPUs.
* Metal: Workspace constructors take the optional keyword argument `atype = :metal32`.
  Sinkhorn steps are then performed on the GPU of Apple M1 or M2 processors.
* oneAPI: Workspace constructors take the optional keyword argument `atype = :one32` or `:one64`.
  Sinkhorn steps are then performed on the GPU of Intel GPU devices.
