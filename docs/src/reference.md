
```@meta
CurrentModule = MuSink
```

# API Reference

## Main API

```@docs
Problem
Problem(::Dict{Node, Array{Float64, 3}}; kwargs...)
Workspace
Workspace(::Problem; kwargs...)
Chain
Barycenter
step!
converge!
marginal
target
```

## Couplings and transport

```@docs
Coupling
transport
transport_window
```

## Reductions
```@docs
Reductions.Reduction
Reductions.reduce
```

### Predefined Reductions
```@docs
Reductions.ishift
Reductions.jshift
Reductions.ishiftsq
Reductions.jshiftsq
Reductions.ivar
Reductions.jvar
Reductions.var
Reductions.std
Reductions.imap
Reductions.jmap
Reductions.coloc
```

## Trees
```@docs
Tree.Node
Tree.root
Tree.parent
Tree.children
Tree.descendants
```

## Querying workspaces
```@docs
get_eps
get_rho
get_reach
get_weight
get_stepmode
get_domain
potential
nodes
edges
```

## Modifying workspaces
```@docs
set_eps!
set_rho!
set_reach!
set_weight!
set_stepmode!
set_domain!
```

## Empirical Workspaces
```@docs
Empirical.EmpiricalWorkspace
```

## Remote Workspaces
```@docs
Remote.RemoteWorkspace
Remote.init
```

