
"""
Dictionary that maps symbols to array types that are supported by `Workspace`s.
"""
const ATYPES = Dict{Symbol, Any}(
  :array32 => Array{Float32},
  :array64 => Array{Float64}
)

function atype(T :: Type{<: AbstractArray})
  for S in values(ATYPES)
    if T <: S
      return S
    end
  end
  @assert false "Array type $T is not supported"
end

function atype(sym :: Symbol)
  @assert haskey(ATYPES, sym) "Array type symbol :$sym is not supported"
  ATYPES[sym]
end

function atype(str :: String)
  sym = Symbol(lowercase(str))
  atype(sym)
end

function register_atype!(sym, T)
  if haskey(ATYPES, sym)
    if !(ATYPES[sym] == T)
      @warn "Cannot register type symbol $sym: already registered"
    end
  else
    ATYPES[sym] = T
  end
end


"""
Core structure for solving UMOT problems. Besides the static properties
provided by a `Problem` (tree topology, target and reference measures, cost and
penalty functions), a `Workspace` also stores various parameters and auxiliary
variables of the optimization problem.

Workspaces are powered by fast implementations of the inner loop of Sinkhorn
iterations. These iterations can be performed either in the exponential domain
(fast, but breaks down for small regularization parameters `eps`) or the
logarithmic domain (slower, but remains stable for small `eps`). The domain can
be specified via the keyword argument `logdomain`. The default is `logdomain
= true`.

### Sinkhorn steps and update rules

Sinkhorn steps that modify the `Workspace` (or, more precisely, the potentials
and auxiliary α-arrays stored in the workspace) can conveniently be performed
via the `step! ` or `steps!` function. Based on the array type of the workspace,
the properties of the cost function (separable or not), and the operating
domain (logdomain or expdomain), specialized kernels are used.

Since the order of the updates can influence the stability / convergence of the
algorithm, several different update rules (accessible via the keyword argument
`stepmode`) are implemented:

* `stepmode = :legacy`: a backwards pass through the tree (i.e., updating
  the value of α-arrays from parents to leaves) is followed by a forward pass
  (i.e., updating the value of α-arrays from leaves to parents) that also
  updates the potentials. This update rule is proposed by the authors of the
  original UMOT manuscript and seems to work fine in most configurations.
  However, in certain situations, the update rule blocks the propagation
  of information through the tree, such that potential updates within
  a step remain unaware of one another. A possible consequence is mass
  oscillation, which prevents convergence. This behavior is explicit in the
  barycenter problem (e.g., a star-shaped tree topology where the center
  node has fixed potentials via `rho = 0`) **IF** the root node is set to the
  center. If the root node is set to a leaf of the tree, `stepmode = :legacy`
  should always work.

* `stepmode = :stable`: a backwards pass through the tree is followed by a
  mixed forward / backward pass that updates the potentials. This update rule
  fixes the issues of the `:legacy` method but also requires more calculations
  per step.

* `stepmode = :alternate`: only a mixed forward / backward pass that updates
  the potentials is conducted. This update rule uses as many computations per
  step as the `:legacy` method (i.e., it is faster than the `:stable` method)
  but should be protected against harmful information barriers. Still, this
  method is more aggressive and might thus be more susceptible to instabilities
  than `:stable`.

* `stepmode = :symmetric`: a backwards pass is followed by a forward pass. Both
  of these passes update the values of α only. Afterwards, all potentials are
  updated simultaneously. **This update rule is severely broken for more than
  two measures and is subject to change or deprecation**

### Parameters

The UMOT problem relies on several parameters to be chosen by the user, like
the parameters `eps` (strength of the entropic regularization), `reach` (the
interaction radius), or `rho` (strength of the marginal penalty). The latter
can be set per node. It is additionally possible to specify the weight of
individual edges of the tree, which corresponds to a scaling of the cost
function along that edge.
"""
mutable struct Workspace{T <: AbstractArray, F}

  # the original problem
  problem :: Problem

  # store the root of the cost tree
  root :: Node

  # options for the algorithm
  logdomain :: Bool
  stepmode :: Symbol

  # parameters
  eps :: F
  eps_inv :: F
  reach :: Int
  rho :: F
  rhos :: Dict{Node, F} # support non-default values of rho for specific nodes
  weights :: Dict{NTuple{2, Node}, F} # cost-weights for the edges

  # problem data in suitable form
  targets :: Dict{Node, T}
  references :: Dict{Node, T}

  # potentials and cumulative potentials in suitable form
  potentials :: Dict{Node, T}
  alphas :: Dict{NTuple{2, Node}, T}

  # save last generation of values for progress indication
  potentials_prev :: Dict{Node, T}
  alphas_prev :: Dict{NTuple{2, Node}, T}

  # data buffer to prevent temporary allocations
  buffer :: T

  # cost-associated kernel operator
  kernel_op! :: Function
  kernel_ops :: Dict{NTuple{2, Node}, Function} # weighted kernel operators

  # whether the workspace is synchronized and it is safe to rely on the alphas
  synced :: Bool

  # how often was the `step!` function called on the workspace
  steps :: Int
end

"""
    Workspace(problem::Problem; <keyword arguments>)

Create a workspace in which `problem` can be solved by calling [`step!`](@ref)
or [`converge!`](@ref).

All optional arguments `arg` can also be set via `set_[arg]!(ws, ...)` after
construction.

## Arguments
* `eps = 1`. The Sinkhorn scaling parameter.
* `rho = Inf`. The default value of the marginal penalty parameter `rho`.
* `rhos`. Dictionary that maps nodes to `rho` values.
* `weights`. Dictionary that maps edges to values that weight the cost.
* `reach = Inf`. Reach parameter (radius of maximally allowed transport).
* `logdomain = true`. Whether to work in the log or exp domain.
* `stepmode = :stable`. Strategy that determines the order of potential \
  updates. See the documentation of [`Workspace`](@ref) for details.
* `atype = :array64`. Array type used in the backend.
"""
function Workspace( p :: Problem
                  ; eps = 1
                  , rho = Inf
                  , rhos = Dict{Node, Any}()
                  , weights = Dict{NTuple{2, Node}, Any}()
                  , reach = max_reach(p.cost)
                  , logdomain :: Bool = true
                  , stepmode :: Symbol = :stable
                  , atype = :array64 )

  # Strict marginal constraints are usually not compatible with mass differences
  # between the targets.
  # Make a check and warn the user.
  nodes_inf = filter(nodes(p)) do a
    get(rhos, a, rho) == Inf
  end
  ms_inf = map(nodes_inf) do a
    sum(target(p, a), dims = (1, 2))
  end
  if !isempty(nodes_inf) && !all(isapprox(ms_inf[1]), ms_inf)
    @warn """
    Combining strict marginal constraints (rho = Inf) with targets of \
    different masses is not recommended, as it can cause the Sinkhorn \
    algorithm to oscillate indefinitely.
    """
  end

  # get the actual array type and work with the correct element type
  T = MuSink.atype(atype) 
  F = eltype(T)
  eps = F(eps)
  rho = F(rho)

  # create buffers and data storage
  buffer = T(undef, p.dims)
  buffer .= zero(F)

  targets = Tree.node_store(p.root, buffer)
  references = Tree.node_store(p.root, buffer)
  potentials = Tree.node_store(p.root, buffer)
  potentials_prev = Tree.node_store(p.root, buffer)

  alphas = Tree.edge_store(p.root, buffer)
  alphas_prev = Tree.edge_store(p.root, buffer)

  # initialize potentials
  for potential in values(potentials)
    if logdomain
      potential .= zero(F) 
    else
      potential .= one(F)
    end
  end

  # initialize target measures
  for node in keys(targets)
    target = p.targets[node]
    reference = p.references[node]
    if logdomain
      references[node] .= convert(T, eps .* log.(reference))
      targets[node] .= convert(T, eps .* log.(target))

      # get target measure relative to the specified reference measure
      targets[node] .-= references[node]

      # the operation above could introduce NaNs when the reference measures
      # carry zeros. Thus, we force the targets to be zero at these positions
      # as well
      unsupported = references[node] .== typemin(F) # this is -Inf or -Inf32
      targets[node][unsupported] .= typemin(F)
    else
      references[node] .= convert(T, reference)
      targets[node] .= convert(T, target)

      # get target measure relative to the specificed reference measure
      targets[node] ./= references[node]

      # the operation above could introduce NaNs when the reference measures
      # carry zeros. Thus, we force the targets to be zero at these positions
      # as well
      unsupported = references[node] .== zero(F)
      targets[node][unsupported] .= zero(F)
    end
  end

  # check that the provided weights have the correct format
  # since weights are undirected, we always expect lower indices first
  # when specifying an edge
  for (a, b) in keys(weights)
    @assert Tree.index(a) < Tree.index(b)
  end

  # set the value of reach (handles floating point reach)
  reach = clamp(reach, 1, max_reach(p.cost))
  reach = floor(Int, reach)

  # compile kernel operators
  kernel_ops = Dict{NTuple{2, Node}, Function}()

  if logdomain

    kernel_op! = Ops.logdomain(T, p.dims, p.cost, eps, reach)
    for (edge, weight) in weights
      kernel_ops[edge] = Ops.logdomain(T, p.dims, weight * p.cost, eps, reach)
    end

  else

    kernel_op! = Ops.expdomain(T, p.dims, p.cost, eps, reach)
    for (edge, weight) in weights
      kernel_ops[edge] = Ops.expdomain(T, p.dims, weight * p.cost, eps, reach)
    end

  end

  Workspace{T, eltype(T)}( p
                         , p.root 
                         , logdomain
                         , check_stepmode(stepmode)
                         , eps
                         , one(F) / eps
                         , reach
                         , rho
                         , rhos
                         , weights
                         , targets
                         , references
                         , potentials
                         , alphas
                         , potentials_prev
                         , alphas_prev
                         , buffer
                         , kernel_op!
                         , kernel_ops
                         , false
                         , 0 )

end

"""
    atype(ws::Workspace)

Return the array type of `ws`.
"""
function atype(:: Workspace{T}) where {T}
  T
end

"""
    steps(ws::Workspace)

Return the total number of Sinkhorn steps that have been applied to `ws`.
"""
function steps(w :: Workspace{T}) where {T}
  w.steps
end

function Base.show(io :: IO, w :: Workspace{T}) where {T}
  n = length(w.targets)
  dims = size(w.buffer)
  print(io, "Workspace{$T}($n, $dims)")
end

function Base.show(io :: IO, ::MIME"text/plain", w :: Workspace{T}) where {T}
  dims = join(size(w.buffer), "×")
  targets = "targets: $(length(w.targets)) ($dims)"
  eps = "eps: $(w.eps)"
  rho = "rho: $(w.rho)"
  reach = "reach: $(w.reach)"
  log = "logdomain: $(w.logdomain)"
  mode = "stepmode: :$(w.stepmode)"

  println(io, "Workspace{$T}:")
  println(io, "  $targets")
  println(io, "  $eps")
  println(io, "  $rho")
  println(io, "  $reach")
  println(io, "  $log")
  print(io, "  $mode")
end

function Base.copy(w :: Workspace{T}, S :: Type{<: AbstractArray} = T) where {T}
  rhos = copy(w.rhos)
  weights = copy(w.weights)
  ws = Workspace( w.problem
                ; w.eps
                , w.rho
                , rhos
                , weights
                , w.reach
                , w.logdomain
                , w.stepmode
                , atype = S )

  ws.steps = w.steps
  ws.buffer .= convert(S, w.buffer)

  for key in keys(w.potentials)
    ws.potentials[key] .= convert(S, w.potentials[key])
    ws.potentials_prev[key] .= convert(S, w.potentials_prev[key])
  end

  for key in keys(w.alphas)
    ws.alphas[key] .= convert(S, w.alphas[key])
    ws.alphas_prev[key] .= convert(S, w.alphas_prev[key])
  end

  ws
end

function Base.convert(:: Type{Workspace{T}}, w :: Workspace{S}) where {T, S}
  T == S ? w : copy(w, T)
end

function Base.convert(:: Type{Workspace{T, F}}, w :: Workspace{S}) where {T, F, S}
  T == S ? w : copy(w, T)
end

# ------ Sinkhorn steps ------------------------------------------------------ #

"""
    kernel_op(ws::Workspace, a, b)

Returns the kernel operator between nodes `a` and `b`.
"""
function kernel_op(w :: Workspace, a :: Node, b :: Node)
  @assert Tree.index(a) != Tree.index(b)
  if Tree.index(a) < Tree.index(b)
    edge = (a, b)
  else
    edge = (b, a)
  end

  if haskey(w.weights, edge)
    w.kernel_ops[edge]
  else
    w.kernel_op!
  end
end

"""
    update_alpha!(ws::Workspace, a, b)

Update the value of alpha between the nodes `a` and `b`.
"""
function update_alpha!(w :: Workspace, from :: Node, to :: Node)
  alpha = w.alphas[(from, to)]
  w.alphas_prev[(from, to)] .= alpha
  w.buffer .= w.potentials[to]

  # Since we integrate the potential over the reference measure, and not
  # over the counting measure, we have to include it here
  if w.logdomain
    w.buffer .+= w.references[to]
  else
    w.buffer .*= w.references[to]
  end

  for neighbor in Tree.neighbors(to)
    if Tree.index(neighbor) != Tree.index(from)
      if w.logdomain
        w.buffer .+= w.alphas[(to, neighbor)]
      else
        w.buffer .*= w.alphas[(to, neighbor)]
      end
    end
  end

  op! = kernel_op(w, from, to)
  op!(alpha, w.buffer)
end

"""
    update_potential!(ws::Workspace, node)

Update the potential at node `node`.

The update depends on the values of alpha and the costs along the edges `(node,
neighbor)` for any neighbor of `node`.
"""
function update_potential!(w :: Workspace, node :: Node)
  potential = w.potentials[node]
  w.potentials_prev[node] .= potential
  w.buffer .= w.targets[node]

  for neighbor in Tree.neighbors(node)
    if w.logdomain
      w.buffer .-= w.alphas[(node, neighbor)]
    else
      # TODO: this should not use divisions more often than necessary...
      w.buffer ./= w.alphas[(node, neighbor)]
    end
  end

  # If the node has a dedicated value for rho, take it; otherwise use the default
  rho = get(w.rhos, node, w.rho)
  if w.logdomain
    w.buffer .= -w.buffer
    approx!(w.buffer, w.problem.penalty, rho, w.eps)
    potential .= .- w.buffer
  else
    w.buffer .= .- w.eps .* log.(w.buffer)
    approx!(w.buffer, w.problem.penalty, rho, w.eps)
    potential .= exp.(.- w.buffer .* w.eps_inv)
  end

  # updates of the potential imply that re-syncing might be necessary
  w.synced = false
end

"""
    update_potential_symmetric!(ws::Workspace, node)

Symmetric update step for the potential at `node`.

!!! warning

    These update steps only work for trees with two nodes and require
   `ws` to be in the logdomain. In practice, they do not work well.
"""
function update_potential_symmetric!(w :: Workspace, node :: Node)
  @assert w.logdomain "Symmetric potential updates are only supported in logdomain"

  if length(w.problem) > 2
    @warn "Symmetric updates will likely not work for >= 2 targets" maxlog=1
  end

  potential = w.potentials[node]
  w.potentials_prev[node] .= potential
  w.buffer .= w.targets[node]

  for neighbor in Tree.neighbors(node)
    w.buffer .-= w.alphas[(node, neighbor)]
  end

  # If this node has a special value of rho, take it. Otherwise, use the default
  rho = get(w.rhos, node, w.rho)
  w.buffer .= .- 0.5 .* w.buffer .- 0.5 .* potential
  approx!(w.buffer, w.problem.penalty, rho, w.eps)
  potential .= .- w.buffer
end

"""
    backward_pass!(ws::Workspace)

Update the values of alpha for all edges pointing away from the root node.
"""
function backward_pass!(w :: Workspace)
  nodes = Tree.descendants(w.root, false) # don't use the root node itself

  for node in reverse(nodes)
    parent = Tree.parent(node)
    update_alpha!(w, parent, node)
  end
end

"""
    forward_pass!(ws::Workspace; update_potentials = true, callback)

Update the values of alpha for all edges pointing towards the root node.
If `update_potentials = true`, the potentials are updated in the process.
The optional function `callback()` is called after each update.
"""
function forward_pass!( w :: Workspace
                      ; update_potentials = true
                      , callback = () -> nothing )

  update_potentials && update_potential!(w, w.root)

  for node in Tree.descendants(w.root, false)
    parent = Tree.parent(node)
    update_alpha!(w, node, parent)
    update_potentials && update_potential!(w, node)
    callback()
  end
end

"""
    forward_backward_pass!(ws::Workspace)

Update all alphas and all potentials for all edges of the tree in a
mixed forward / backward pass.

This function can be used as a replacement of `forward_pass!(ws,
update_potentials = true)`. It is meant to prevent a shortcoming in the
proposed default algorithm, where the potentials fail to converge due to
lacking propagation of information (e.g., when calculating barycenters
where the root node is assigned `rho = 0`).
"""
function forward_backward_pass!(w :: Workspace)
  update_potential!(w, w.root)

  for node in Tree.descendants(w.root, false)
    parent = Tree.parent(node)
    # first the forward update
    update_alpha!(w, node, parent)
    # now the potential update
    update_potential!(w, node)
    # finally the backward update
    update_alpha!(w, parent, node)
  end
end

"""
    forward_pass_symmetric!(ws::Workspace; update_potentials = true)

Update the values of all alpha and all potentials in a symmetric way.

!!! warning

    In practice, this update strategy does not work well and should be
    avoided.
"""
function forward_pass_symmetric!(w :: Workspace; update_potentials = true)

  for node in Tree.descendants(w.root, false)
    parent = Tree.parent(node)
    update_alpha!(w, node, parent)
  end

  if update_potentials
    for node in Tree.descendants(w.root, true)
      update_potential_symmetric!(w, node)
    end
  end
end

"""
    step!(ws::Workspace, steps = 1; max_time = Inf, stepmode = ws.stepmode)

Perform `steps` Sinkhorn update steps with a given `stepmode`.

The argument `max_time` denotes a time limit in seconds, after which no more
steps are performed.
"""
function step!(w :: Workspace{T}; stepmode = w.stepmode) where {T}
  stepmode = string(stepmode)
  if stepmode in ["legacy"]
    backward_pass!(w)
    forward_pass!(w)
  elseif stepmode in ["alternate", "alternating"]
    forward_backward_pass!(w)
  elseif stepmode in ["stable", "default"]
    backward_pass!(w)
    forward_backward_pass!(w)
  elseif stepmode in [:symmetric, "symmetric"]
    backward_pass!(w)
    forward_pass_symmetric!(w)
  else
    error("Invalid stepmode :$stepmode")
  end
  w.steps += 1
  1
end

function step!(w :: Workspace, number :: Real; max_time = Inf, kwargs...)
  count = 0
  start_time = time_ns()
  dt = 0
  while count < number && dt < max_time
    step!(w; kwargs...)
    dt = (time_ns() - start_time) / 1e9
    count += 1
  end
  count
end

"""
    sync_arrays(ws::Workspace)

Synchronize the arrays in `ws`.

Can be used by asynchronous array implementations, like `CuArray`s, to make sure
that all pending operations are carried out.
"""
function sync_arrays(:: Workspace{T}) where {T}
  sync_arrays(T)
end

function sync_arrays(:: Type{<: AbstractArray})
  nothing
end

function sync!(w :: Workspace{T}) where {T}
  if !w.synced
    backward_pass!(w)
    forward_pass!(w, update_potentials = false)
    w.synced = true
    sync_arrays(w)
  end
end


# ------ Modifying workspaces ------------------------------------------------ #

function update_operators!(w :: Workspace{T}) where {T}
  p = w.problem
  if w.logdomain
    w.kernel_op! = Ops.logdomain(T, p.dims, p.cost, w.eps, w.reach)
    for (edge, weight) in w.weights
      w.kernel_ops[edge] = Ops.logdomain(T, p.dims, weight * p.cost, w.eps, w.reach)
    end
  else
    w.kernel_op! = Ops.expdomain(T, p.dims, p.cost, w.eps, w.reach)
    for (edge, weight) in w.weights
      w.kernel_ops[edge] = Ops.expdomain(T, p.dims, weight * p.cost, w.eps, w.reach)
    end
  end
  nothing
end

"""
    set_eps!(ws::Workspace, eps)

Set the value of epsilon to `eps`. The logdomain potentials are kept.
"""
function set_eps!(w :: Workspace{T}, eps) where {T}
  eps_old = w.eps
  eps_inv_old = 1 / eps_old

  w.eps = eps
  w.eps_inv = 1 / eps
  w.synced = false

  if w.logdomain
    # In logdomain, targets and references change
    for node in keys(w.targets)
      w.targets[node] .= (w.eps * eps_inv_old) .* w.targets[node]
      w.references[node] .= (w.eps * eps_inv_old) .* w.references[node]
    end

  else
    # In expdomain, potentials change but targets and references remain
    for node in keys(w.potentials)
      w.potentials[node] .= eps_old .* log.(w.potentials[node])
      w.potentials[node] .= exp.(w.potentials[node] .* w.eps_inv)
      w.potentials_prev[node] .= eps_old .* log.(w.potentials_prev[node])
      w.potentials_prev[node] .= exp.(w.potentials_prev[node] .* w.eps_inv)
    end
  end

  update_operators!(w)
end

"""
    set_reach!(ws::Workspace, reach::Integer)

Set the reach of `ws` to `reach`.
"""
function set_reach!(w :: Workspace{T}, reach :: Integer) where {T}
  p = w.problem
  reach = clamp(reach, 1, max_reach(p.cost))
  w.reach = reach
  w.synced = false
  update_operators!(w)
end

"""
    set_rho!(ws::Workspace, rho)
    set_rho!(ws::Workspace, a, rho)

Set the default value of rho to `rho`. If a node (or node index) `a` is
provided, only set the value for this specific node.
"""
function set_rho!(w :: Workspace, rho)
  w.rho = rho
  nothing
end

function set_rho!(w :: Workspace, node :: Node, rho)
  if isnothing(rho)
    delete!(w.rhos, node)
  else
    w.rhos[node] = rho
  end
  nothing
end

function set_rho!(w :: Workspace, index :: Int, rho)
  node = Tree.descendant(w.root, index)
  set_rho!(w, node, rho)
end

"""
    set_weight!(w::Workspace, a, b, weight)

Set the cost weight of the edge between `a` and `b` to `weight`
"""
function set_weight!(w :: Workspace{T}, a :: Node, b :: Node, weight) where {T}
  @assert Tree.index(a) != Tree.index(b)
  if Tree.index(a) < Tree.index(b)
    edge = (a, b)
  else
    edge = (b, a)
  end

  if isnothing(weight)
    delete!(w.weights, edge)
    delete!(w.kernel_ops, edge)

  else
    p = w.problem
    w.weights[edge] = weight
    if w.logdomain
      w.kernel_ops[edge] = Ops.logdomain(T, p.dims, weight * p.cost, w.eps, w.reach)
    else
      w.kernel_ops[edge] = Ops.expdomain(T, p.dims, weight * p.cost, w.eps, w.reach)
    end
  end
  nothing
end

function set_weight!(w :: Workspace, index_a :: Int, index_b :: Int, weight)
  a = Tree.descendant(w.root, index_a)
  b = Tree.descendant(w.root, index_b)
  set_weight!(w, a, b, weight)
end

"""
    set_domain!(ws::Workspace, logdomain::Bool)

If `logdomain = true`, move `ws` into the logdomain. If `logdomain = false`,
move `ws` into the expdomain.
"""
function set_domain!(w :: Workspace{T}, logdomain) where {T}
  dims = size(w.buffer)
  p = w.problem
  if w.logdomain == logdomain
    nothing
  elseif logdomain
    w.logdomain = true
    w.synced = false
    for node in keys(w.targets)
      w.targets[node] .= w.eps .* log.(w.targets[node])
      w.references[node] .= w.eps .* log.(w.references[node])
      w.potentials[node] .= w.eps .* log.(w.potentials[node])
      w.potentials_prev[node] .= w.eps .* log.(w.potentials_prev[node])
    end

  else
    w.logdomain = false
    w.synced = false
    for node in keys(w.targets)
      w.targets[node] .= exp.(w.targets[node] .* w.eps_inv)
      w.references[node] .= exp.(w.references[node] .* w.eps_inv)
      w.potentials[node] .= exp.(w.potentials[node] .* w.eps_inv)
      w.potentials_prev[node] .= exp.(w.potentials_prev[node] .* w.eps_inv)
    end
  end
  update_operators!(w)
end

function check_stepmode(stepmode)
  stepmode = stepmode |> string |> lowercase
  options = ["legacy", "alternate", "stable", "symmetric"]
  if stepmode in options
    Symbol(stepmode)
  else
    @warn "stepmode :$stepmode is invalid. Falling back to :stable" 
    :stable
  end
end

"""
    set_stepmode!(ws::Workspace, stepmode)

Set the step mode of `ws` to `stepmode`.
"""
function set_stepmode!(w :: Workspace, stepmode)
  stepmode = check_stepmode(stepmode)
  w.stepmode = stepmode
  w.synced = false
  nothing
end

# ------ Extracting parameters ----------------------------------------------- #

"""
    get_domain(ws::Workspace)

Returns `true` if `ws` is in the logdomain and `false` else.
"""
function get_domain(w :: Workspace)
  w.logdomain
end

"""
    get_eps(ws::Workspace)

Returns the current value of `eps`.
"""
function get_eps(w :: Workspace)
  w.eps
end

"""
    get_rho(ws::Workspace)
    get_rho(ws::Workspace, a)

Returns the current value of `rho` (default or at node `a`).
"""
function get_rho(w :: Workspace)
  w.rho
end

function get_rho(w :: Workspace, node :: Node)
  get(w.rhos, node, w.rho)
end

function get_rho(w :: Workspace, index :: Int)
  node = Tree.descendant(w.root, index)
  get_rho(w, node)
end

"""
    get_reach(ws::Workspace)

Returns the current value of `reach`.
"""
function get_reach(w :: Workspace)
  w.reach
end

"""
    get_weight(ws::Workspace, a, b)

Returns the cost weight between the nodes `a` and `b`.
"""
function get_weight(w :: Workspace{T, F}, a :: Node, b :: Node) where {T, F}
  index_a = Tree.index(a)
  index_b = Tree.index(b)

  @assert index_a != index_b
  @assert Tree.has_neighbor(a, b) "Nodes $index_a and $index_b are not neighbors"

  if index_a < index_b
    edge = (a, b)
  else
    edge = (b, a)
  end

  get(w.weights, edge, F(1.0))
end

function get_weight(w :: Workspace, index_a :: Int, index_b :: Int)
  a = Tree.descendant(w.root, index_a)
  b = Tree.descendant(w.root, index_b)
  get_weight(w, a, b)
end

"""
    get_stepmode(ws::Workspace)

Returns the stepmode of `ws`.
"""
function get_stepmode(w :: Workspace)
  w.stepmode
end


# ------ Extracting node elements -------------------------------------------- #

"""
    nodes(ws::Workspace)

Returns all nodes of the problem cost tree.
"""
nodes(w :: Workspace) = Tree.descendants(w.root)

"""
    edges(ws::Workspace)

Returns all (bidirectional) edges of the problem cost tree.
"""
edges(w :: Workspace) = Tree.edges(w.root)


"""
    edges_outwards(ws::Workspace)

Returns all edges of the problem cost tree that point away from the root.
"""
edges_outwards(w :: Workspace) = Tree.edges_outwards(w.root)

"""
    edges_inwards(ws::Workspace)

Returns all edges of the problem cost tree that point towards the root.
"""
edges_inwards(w :: Workspace) = Tree.edges_inwards(w.root)

"""
    maybe_drop_batchdim(array, keep = false)

Drop the batch dimension of an array. Convenient for suppressing the trailing
dimension in the interface if the problem has batchsize 1.
"""
function maybe_drop_batchdim(x :: AbstractArray{F, N}, keep :: Bool) where {F, N}
  if !keep && size(x, N) == 1
    drop_batchdim(x)
  else
    x
  end
end

drop_batchdim(x :: AbstractVector) = x[1]
drop_batchdim(x :: AbstractArray{F, N}) where {F, N} = dropdims(x, dims = N)


"""
    potential(ws::Workspace, node; keep_batchdim = false)

The potential at `node`. Note that this function always returns
the actual dual UMOT potential (i.e., the potential in the logdomain).

If `keep_batchdim = true`, the singular batch dimension in case of batchsize 1
is not dropped.
"""
function potential(w :: Workspace, node :: Node; keep_batchdim = false)
  if w.logdomain
    output = w.potentials[node]
  else
    output = log.(w.potentials[node]) .* w.eps
  end
  maybe_drop_batchdim(output, keep_batchdim)
end

function potential(w :: Workspace, index :: Int; kwargs...)
  node = Tree.descendant(w.root, index)
  potential(w, node; kwargs...)
end


"""
    target(ws::Workspace, node; keep_batchdim = false)

The target measure at `node` with respect to the counting measure.

If `keep_batchdim = true`, the singular batch dimension in case of batchsize 1
is not dropped.
"""
function target(w :: Workspace{T}, node_or_index; keep_batchdim = false) where {T}
  output = convert(T, target(w.problem, node_or_index))
  maybe_drop_batchdim(output, keep_batchdim)
end


"""
    marginal(ws::Workspace, node; keep_batchdim = false)

The marginal measure at `node` with respect to the counting measure.

If `keep_batchdim = true`, the singular batch dimension in case of batchsize 1
is not dropped.
"""
function marginal(w :: Workspace{T, F}, node :: Node; keep_batchdim = false) where {T, F}
  sync!(w)
  buffer = copy(w.buffer)
  buffer .= w.potentials[node]

  # Return the marginal with respect to the counting measure
  if w.logdomain
    buffer .+= w.references[node]
  else
    buffer .*= w.references[node]
  end
  
  for neighbor in Tree.neighbors(node)
    if w.logdomain
      buffer .+= w.alphas[node, neighbor]
    else
      buffer .*= w.alphas[node, neighbor]
    end
  end

  if w.logdomain
    buffer .= exp.(buffer ./ w.eps)
  end

  maybe_drop_batchdim(buffer, keep_batchdim)
end

function marginal(w :: Workspace{T, F}, index :: Int; kwargs...) where {T, F}
  node = Tree.descendant(w.root, index)
  marginal(w, node; kwargs...)
end


# ------ Convenience presets ------------------------------------------------- #

"""
    Chain(targets; kwargs...)

Directly create a workspace that puts `targets` in a linear cost tree.

Valid keyword arguments are the ones for [`Problem`](@ref) (except for `root`) and
[`Workspace`](@ref).
"""
function Chain( targets
              ; cost = nothing
              , penalty = TotalVariation()
              , references = nothing
              , reference_mass = -1
              , kwargs...)

  # assert that we have at least one target
  len = length(targets)
  @assert len >= 2 "At least two targets must be provided"

  # get the array type and produce the chain-like UMOT problem
  p = Problem(Tree.Sequence(len), targets; cost, references, reference_mass, penalty)
  Workspace(p; kwargs...)
end


"""
    Barycenter(targets; kwargs...)

Directly create a workspace that puts `targets` in a star-shaped cost tree
around a barycenter-node with `rho = 0`. Valid keyword arguments are the ones
for [`Problem`](@ref) (except for `root`) and [`Workspace`](@ref).
"""
function Barycenter( targets
                   ; cost = nothing
                   , penalty = TotalVariation()
                   , references = nothing
                   , reference_mass = -1
                   , kwargs... )

  # assert that we have at least one target
  len = length(targets)
  @assert len >= 1 "At least one target must be provided"

  bary = similar(targets[1])
  bary .= 1
  bary_targets = vcat([bary], targets)

  # get the array type and produce the chain-like UMOT problem
  p = Problem(Tree.Star(len+1), bary_targets; cost, references, reference_mass, penalty)
  w = Workspace(p; kwargs...)

  # the barycenter target receives no marginal penalty
  set_rho!(w, 1, 0.0)
  w
end


"""
    testspace(T; cost, n, m, b, ntargets, eps, rho, logdomain, reach)  

Create a random [`Workspace`](@ref) with sequential cost topology.

Auxiliary function intended for testing purposes.
"""
function testspace(;
                    cost :: Type{<: Cost} = Lp,
                    n = 7,
                    m = 9,
                    b = 1,
                    ntargets = 5,
                    eps = 1,
                    rho = 1,
                    logdomain = true,
                    reach = Inf,
                    atype = Array{Float64}
                  )

  cost = cost(n, m)
  n, m = size(cost)

  root = Tree.Sequence(ntargets)
  targets = map(1:ntargets) do i
    target = rand(n, m, b)
    target ./= sum(target, dims = (1,2)) 
    Tree.descendant(root, i) => target
  end |> Dict

  references = default_references(targets, 1.0)
  problem = Problem(targets; cost, references)

  Workspace(problem; eps, rho, logdomain, reach, atype)
end
