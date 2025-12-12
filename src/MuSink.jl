
module MuSink

using Random, LinearAlgebra, Statistics, SparseArrays
using FFTW

## Ground costs and marginal penalties.

include("cost.jl")

## Marginal penalties for unbalanced constraint

include("penalty.jl")

"""
Low-level CPU and GPU implementation of the inner loops that power the Sinkhorn
algorithm.
"""
module Ops

  using Random, LinearAlgebra, Statistics, SparseArrays
  using FFTW

  import ..MuSink
  import ..MuSink: Cost, SeparableCost, Lp
  import ..MuSink: matrix, matrixCSC, window, axes, matrix_axes, matrixCSC_axes

  include("ops.jl")

end # module Ops


## Simple tree implementation that is independent of other parts of MuSink.
## Used to realize tree-like ground cost structures in the type `Problem`.
module Tree

  include("tree.jl")

end # module Tree

import .Tree: Node

## Representation of an unbalanced multi-marginal optimal transport problem
## via the type `Problem`.

include("problem.jl")

## Functionality to generate and modify `Workspace`s, which carry the working
## data (e.g., dual potentials) for unbalanced multi-marginal sinkhorn
## iterations.

include("workspace.jl")

## Implements the extraction of pairwise-couplings of the multi-marginal
## transport plan.

include("coupling.jl")

## Reduction functions (efficient integrals over the transport plan)
module Reductions

  import ..MuSink: Workspace, Coupling, Ops, Tree
  import ..MuSink: Cost, SeparableCost, GenericCost, GenericSeparableCost
  import ..MuSink: window, axes

  include("reductions.jl")

end # module Reductions

## Scalar values, like mass and costs, that can be extracted from workspaces
## and couplings

include("metrics.jl")

## Auxiliary diagnostics meant to give an overview of the state and health of
## a workspace.

include("diagnostics.jl")

## Auxiliary function to iterate Sinkhorn steps until convergence

include("converge.jl")

## Convenience implementation of a remote workspace that can run on a different
## machine via ssh
module Remote

  import Distributed: RemoteChannel

  import ..MuSink
  import ..MuSink: Workspace

  include("remote.jl")

end

"""
Module that implements entropic optimal transport between two arbitrary discrete
measures of the same weight.

This is an auxiliary extension to MuSink and is primarily used for testing
purposes.
"""
module Empirical

  using LinearAlgebra, Statistics

  import ..MuSink
  import ..MuSink: dense, converge!, absfinite, get_eps
  import ..MuSink: step!, set_eps!, set_domain!, set_stepmode!


  include("empirical.jl")

end

const RemoteWorkspace = Remote.RemoteWorkspace
const EmpiricalWorkspace = Empirical.EmpiricalWorkspace


## Dynamic extensions of core MuSink

export Problem, Workspace, Coupling, Tree
export Lp, LpNotSep, KullbackLeibler, TotalVariation
export Chain, Barycenter

export RemoteWorkspace
export EmpiricalWorkspace

export Reductions
export target, marginal, mass, cost, objective, dense

export step!
export get_eps, get_reach, get_weight, get_rho, get_domain, get_stepmode
export set_eps!, set_reach!, set_weight!, set_rho!, set_domain!, set_stepmode!
export converge!

export transport, transport_window

end # module MuSink

