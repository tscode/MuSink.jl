# MuSink

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://tscode.github.io/MuSink.jl/stable/)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://tscode.github.io/MuSink.jl/dev/)
[![Build Status](https://github.com/tscode/MuSink.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tscode/StructPack.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package efficiently implements unbalanced multi-marginal optimal transport (UMOT) between measures on uniform two dimensional grids (i.e., images).

It is written in [julia](https://julialang.org) but can also be accessed from both [R](https://www.r-project.org) and [python](https://www.python.org) (see the [documentation](https://tscode.github.io/MuSink.jl/dev/)).

The problem formulation and the algorithmic approach is inspired by [this publication](https://arxiv.org/abs/2103.10854).

Some of the core features of MuSink.jl are:

* very efficient implementation of Sinkhorn updates taylored to image data,
* supports both exponential (fast but instable) and logarithmic (stable but slower) domain,
* threading support,
* CUDA, Metal, and oneAPI support (with custom kernels for the logdomain),
* low memory footprint,
* general tree-based cost topologies,
* extendable from outside the package.

## Installation

To install it, run
```julia
using Pkg
Pkg.add("MuSink")
```
in a julia shell.
This command will take care of installing all necessary dependencies.
Note that only julia versions `1.9` and newer are tested.

## Usage

For an overview of the basic functionality of MuSink.jl you can consult [this tutorial](https://tscode.github.io/MuSink.jl/dev/usage/).

### Python

Instructions for installation and usage of MuSink.jl in python are provided [here](https://tscode.github.io/MuSink.jl/dev/python/).

### R

Documentation of how to install and use MuSink.jl from R will follow.

## Optional dependencies

MuSink supports the packages `LoopVectorization`, `CUDA`, `Metal`, and `oneAPI` as optional dependencies.
Installing and `using ...` them will unlock additional functionality of MuSink.jl.

* **LoopVectorization.jl**: Sinkhorn steps on the CPU in the logdomain will be faster,
  at the cost of a higher first execution time.
* **CUDA.jl**: Workspace constructors (see the [tutorial](https://gitlab.gwdg.de/staudt1/musink/-/wikis/Usage))
  accept the optional keyword argument `atype = :cuda32` or `:cuda64`.
  Sinkhorn steps are then performed on cuda-enabled GPUs.
* **Metal.jl**: Workspace constructors accept the optional keyword argument `atype = :metal32`.
  Sinkhorn steps are then performed on the GPU of Apple M1 or M2 processors.
* **oneAPI.jl**: Workspace constructors accept the optional keyword argument `atype = :one32` or `:one64`.
  Sinkhorn steps are then performed on the GPU of Intel GPU devices.
