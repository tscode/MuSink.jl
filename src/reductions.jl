
# ------ Reductions over couplings ------------------------------------------- #

"""
Reduction over a coupling. This type performantly implements sums of the form
`sum_{b} pi_{ab} * f(a-b)`, where `pi` is a coupling.

Note that the function `f` must return strictly positive values, since
evaluation of `f` takes place in the logdomain.

!!! note

    Generic and efficient implementations are only possible for neighboring
    nodes. Reductions between non-neighboring nodes are currently not supported.
"""
abstract type Reduction end

struct GenericReduction <: Reduction
  f :: Function
  opcache :: Dict{Any, Function} # Compiled operations that depend on edge weight
end

function Base.show(io :: IO, ::MIME"text/plain", red :: GenericReduction)
  l = length(red.opcache)
  print(io, "GenericReduction($l cached operators)")
end

function Base.show(io :: IO, :: GenericReduction)
  print(io, "GenericReduction()")
end

struct SeparableReduction <: Reduction
  fi :: Function
  fj :: Function
  opcache :: Dict{Any, Function} # Compiled operations that depend on edge weight
end

function Base.show(io :: IO, ::MIME"text/plain", red :: SeparableReduction)
  l = length(red.opcache)
  print(io, "SeparableReduction($l cached operators)")
end

function Base.show(io :: IO, :: SeparableReduction)
  print(io, "SeparableReduction()")
end

"""
    Reduction(f)
    Reduction(fi, fj)

Create a reduction over the function `f(ri, rj, c)`. If separate functions
`fi` and `fj` are provided, the reduction function is `f(ri, rj, c) = fi(ri)
* fj(rj)`.

In the latter case, separability of the resulting kernel can be exploited,
however, the functions `fi` and `fj` do not have access to the cost value
`c` at `ri`, `rj`.
"""
function Reduction(f; kwargs...)
  GenericReduction(f, Dict{Any, Function}())
end

function Reduction(fi, fj; kwargs...)
  SeparableReduction(fi, fj, Dict{Any, Function}())
end

function reduction_operator(f :: Function, plan :: Coupling{T}) where {T}
  dims = size(plan.buffer)
  n, m = size(plan.cost)
  weight = plan.weights[1]

  cc = window(plan.cost, n-1, m-1)

  # create the reduction window
  rcc = zeros(2n - 1, 2m - 1)
  for ir in -(n-1):(n-1), jr in -(m-1):(m-1)
    c = cc[ir + n, jr + m]
    rcc[ir + n, jr + m] = weight * c - log(f(ir, jr, c)) * plan.eps
  end

  rcost = GenericCost(rcc)
  if plan.logdomain
    Ops.logdomain(T, dims, rcost, plan.eps, plan.reach)
  else
    Ops.expdomain(T, dims, rcost, plan.eps, plan.reach)
  end
end

function reduction_operator(fi :: Function, fj :: Function, plan :: Coupling{T}) where {T}
  @assert plan.cost isa SeparableCost """
  Separable reductions are only applicable to separable costs.
  """

  dims = size(plan.buffer)
  n, m = size(plan.cost)
  weight = plan.weights[1]

  ci, cj = axes(plan.cost, n-1, m-1)

  rci = zeros(2n - 1)
  rcj = zeros(2m - 1)

  for ir in -(n-1):(n-1)
    rci[ir + n] = weight * ci[ir + n] - log(fi(ir)) * plan.eps
  end

  for jr in -(m-1):(m-1)
    rcj[jr + m] = weight * cj[jr + m] - log(fj(jr)) * plan.eps
  end

  rcost = GenericSeparableCost(rci, rcj)
  if plan.logdomain
    op! = Ops.logdomain(T, dims, rcost, plan.eps, plan.reach)
  else
    op! = Ops.expdomain(T, dims, rcost, plan.eps, plan.reach)
  end
end

function reduction_operator(red :: SeparableReduction, plan :: Coupling)
  if plan.cost isa SeparableCost
    reduction_operator(red.fi, red.fj, plan)
  else
    f = (ir, jr, _) -> red.fi(ir) * red.fj(jr)
    reduction_operator(f, plan)
  end
end

function reduction_operator(red :: GenericReduction, plan :: Coupling)
  reduction_operator(red.f, plan)
end

function opkey(plan :: Coupling{T}) where {T}
  @assert length(plan.weights) == 1 """
  Cannot provide operator key for non-neighboring nodes.
  """
  dims = size(plan.buffer)
  weight = plan.weights[1]
  (T, dims, plan.cost, plan.eps, plan.reach, plan.logdomain, weight)
end

function cached_operator(red :: Reduction, plan :: Coupling{T}) where {T}
  key = opkey(plan)
  if !haskey(red.opcache, key)
    if length(red.opcache) > 100
      @debug "Clearing operator cache"
      empty!(red.opcache)
    end
    red.opcache[key] = reduction_operator(red, plan)
  end
  red.opcache[key]
end


"""
    reduce(reduction, plan; conditional)
    reduce(reduction, workspace, a, b; conditional)

    reduction(plan; conditional)
    reduction(workspace, a, b; conditional)

Apply the reduction `reduction` to a coupling `plan` or a `workspace` between
nodes `a` and `b`. If `conditional = true`, the result is pointwisely divided
by the marginal measure of node `a`.
"""
function reduce(red :: Reduction, plan :: Coupling; conditional = false)

  @assert Tree.has_neighbor(plan.a, plan.b) """
  Reductions are only supported between neighboring nodes.
  """

  output = similar(plan.buffer)
  op! = cached_operator(red, plan)
  op!(plan.buffer, plan.potential_b)

  if plan.logdomain
    eps_inv = one(plan.eps) / plan.eps
    output .= plan.buffer .+ plan.potential_a
    output .= exp.(output .* eps_inv)
  else
    output .= plan.buffer .* plan.potential_a
  end

  if conditional
    output ./= plan.marginal_a 
  end

  output
end

function reduce(red :: Reduction, w, args...; kwargs...)
  reduce(w, red, args...; kwargs...)
end

function reduce(w :: Workspace, red :: Reduction, a, b; conditional = false)
  plan = Coupling(w, a, b)
  reduce(red, plan; conditional)
end

function (red :: Reduction)(args...; kwargs...)
  reduce(red, args...; kwargs...)
end


# ---- Special reductions -------------------------------------------------- #

"""
Constant value that is added when calculating reductions over shifts (i.e.,
pixel differences in i or j direction). This is necessary since reduction
kernels must be positive functions.
"""
const SHIFT_OFFSET = 1e4

const cost = Reduction((_, _, c) -> c)

const _ishift_offset = Reduction(ir -> ir + SHIFT_OFFSET, jr -> 1.)
const _jshift_offset = Reduction(ir -> 1., jr -> jr + SHIFT_OFFSET)
const _ishiftsq = Reduction(ir -> ir^2, jr -> 1.)
const _jshiftsq = Reduction(ir -> 1., jr -> jr^2)

"""
    ishift(coupling)
    ishift(workspace, a, b)

Pointwise mean shift in the first component of the transport plan from `a` to `b`.
"""
ishift(w :: Workspace, a, b) = ishift(Coupling(w, a, b))

function ishift(plan :: Coupling{T, F}) where {T, F}
  n, _, _ = size(plan.buffer)
  @assert n < SHIFT_OFFSET """
  Shifts in i-direction can currently only be computed for images with first
  dimension < $SHIFT_OFFSET.
  """
  _ishift_offset(plan; conditional = true) .- F(SHIFT_OFFSET)
end

"""
    jshift(coupling)
    jshift(workspace, a, b)

Pointwise mean shift in the second component of the transport plan from `a` to `b`.
"""
jshift(w :: Workspace, a, b) = jshift(Coupling(w, a, b))

function jshift(plan :: Coupling{T, F}) where {T, F}
  _, m, _ = size(plan.buffer)
  @assert m < SHIFT_OFFSET """
  Shifts in j-direction can currently only be computed for images with second
  dimension < $SHIFT_OFFSET.
  """
  _jshift_offset(plan; conditional = true) .- F(SHIFT_OFFSET)
end

"""
    ishiftsq(coupling)
    ishiftsq(workspace, a, b)

Pointwise mean squared shift in the first component of the transport plan from `a` to `b`.
"""
ishiftsq(plan :: Coupling) = _ishiftsq(plan; conditional = true)
ishiftsq(w :: Workspace, a, b) = ishiftsq(Coupling(w, a, b))

"""
    jshiftsq(coupling)
    jshiftsq(workspace, a, b)

Pointwise mean squared shift in the second component of the transport plan from `a` to `b`.
"""
jshiftsq(plan :: Coupling) = _jshiftsq(plan; conditional = true)
jshiftsq(w :: Workspace, a, b) = jshiftsq(Coupling(w, a, b))

"""
    ivar(coupling)
    ivar(workspace, a, b)

Pointwise variance of the first component of the transport plan from `a` to `b`.
"""
function ivar(args...)
  shifts = ishift(args...)
  squares = ishiftsq(args...)
  clamp.(squares .- shifts.^2, 0, Inf)
end

"""
    jvar(coupling)
    jvar(workspace, a, b)

Pointwise variance of the second component of the transport plan from `a` to `b`.
"""
function jvar(args...)
  shifts = jshift(args...)
  squares = jshiftsq(args...)
  clamp.(squares .- shifts.^2, 0, Inf)
end

"""
    var(coupling)
    var(workspace, a, b)

Pointwise variance (both components) of the transport plan from `a` to `b`.
"""
var(args...) = ivar(args...) + jvar(args...)

"""
    var(coupling)
    var(workspace, a, b)

Pointwise standard deviation (both components) of the transport plan from `a` to `b`.
"""
std(args...) = sqrt.(var(args...))


"""
    imap(coupling)
    imap(workspace, a, b)

Pointwise mean position of the first component of the transport plan from `a` to `b`.
"""
imap(w :: Workspace, a, b) = imap(Coupling(w, a,b))

function imap(plan :: Coupling{T, F}) where {T, F}
  n, _, _ = size(plan.buffer)
  offsets = convert(T, 1:n)
  ishift(plan) .+ offsets
end

"""
    jmap(coupling)
    jmap(workspace, a, b)

Pointwise mean position of the second component of the transport plan from `a` to `b`.
"""
jmap(w :: Workspace, a, b) = jmap(Coupling(w, a,b))

function jmap(plan :: Coupling{T, F}) where {T, F}
  _, m, _ = size(plan.buffer)
  offsets = convert(T, 1:m)
  jshift(plan) .+ offsets'
end


"""
    coloc(coupling; threshold, conditional = false)
    coloc(workspace, a, b; threshold, conditional = false)

Colocalization with cost threshold `threshold`.
"""
function coloc(args...; threshold, conditional = false)
  red = Reduction((_, _, c) -> c <= threshold ? 1 : 0)
  red(args...; conditional)
end

