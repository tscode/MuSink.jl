
"""
Structure that facilitates calculating (parts of) the transport plan between
two nodes.

Since the full transport plan is often impractically large, this
type provides a lazy interface that operates on the dual potentials.

Many of the operations on `Coupling`s can only be implemented performantly when
nodes are adjacent. These operations fail for couplings between non-neighboring
nodes.
"""
struct Coupling{T, F}
  logdomain :: Bool

  cost :: Cost

  eps :: F
  reach :: Int

  a :: Node
  b :: Node

  marginal_a :: T
  marginal_b :: T

  potential_a :: T
  potential_b :: T
  potentials_inter :: Vector{T}

  buffer :: T

  weights :: Vector{F}     # weights of edges between a and b
  ops :: Vector{Function}  # kernel operators for edges between a and b
end

"""
    Coupling(workspace, a, b)

Create a coupling object between nodes `a` and `b` that is based on the
current potentials stored in `workspace`.
"""
function Coupling(w :: Workspace{T, F}, a :: Node, b :: Node) where {T, F}


  @assert Tree.has_descendant(w.root, a) "Node $(Tree.index(a)) is not part of the tree"
  @assert Tree.has_descendant(w.root, b) "Node $(Tree.index(b)) is not part of the tree"
  @assert a !== b "Coupling expects different nodes"

  sync!(w)

  marginal_a = marginal(w, a, keep_batchdim = true)
  marginal_b = marginal(w, b, keep_batchdim = true)

  # These potentials are *not* the actual dual potentials,
  # but 'partly marginalized' (alpha-corrected) potentials
  # that already include the reference measures
  potential_a = copy(w.potentials[a])
  potential_b = copy(w.potentials[b])

  if w.logdomain
    potential_a .+= w.references[a]
    potential_b .+= w.references[b]
  else
    potential_a .*= w.references[a]
    potential_b .*= w.references[b]
  end
  
  for neighbor in Tree.neighbors(a)
    if neighbor != Tree.step_towards(a, b)
      if w.logdomain
        potential_a .+= w.alphas[(a, neighbor)]
      else
        potential_a .*= w.alphas[(a, neighbor)]
      end
    end
  end

  for neighbor in Tree.neighbors(b)
    if neighbor != Tree.step_towards(b, a)
      if w.logdomain
        potential_b .+= w.alphas[(b, neighbor)]
      else 
        potential_b .*= w.alphas[(b, neighbor)]
      end
    end
  end

  # Collect edge weights and kernel operators when stepping from a to b
  weights = F[]
  ops = Function[]

  # Collect all intermediate potentials on the way from a to b. Will be
  # empty if a and b are neighbors
  potentials_inter = T[]
  node = a

  while !Tree.has_neighbor(node, b)
    node_from = node
    node = Tree.step_towards(node_from, b)
    node_to = Tree.step_towards(node, b)

    potential = copy(w.potentials[node])

    if w.logdomain
      potential .+= w.references[node]
    else
      potential .*= w.references[node]
    end

    for neighbor in Tree.neighbors(node)
      if neighbor != node_from && neighbor != node_to
        if w.logdomain
          potential .+= w.alphas[(node, neighbor)]
        else
          potential .*= w.alphas[(node, neighbor)]
        end
      end
    end

    push!(potentials_inter, potential)
    push!(ops, kernel_op(w, node_from, node))
    push!(weights, get_weight(w, node_from, node))
  end

  push!(weights, get_weight(w, node, b))
  push!(ops, kernel_op(w, node, b))

  Coupling{T, F}( w.logdomain
                , w.problem.cost
                , w.eps
                , w.reach
                , a
                , b
                , marginal_a
                , marginal_b
                , potential_a
                , potential_b
                , potentials_inter
                , copy(w.buffer)
                , weights
                , ops )
end

function Coupling(w :: Workspace, index_a :: Int, index_b :: Int)
  a = Tree.descendant(w.root, index_a)
  b = Tree.descendant(w.root, index_b)
  Coupling(w, a, b)
end

function Base.show(io :: IO, plan :: Coupling{T}) where T
  idx_a = Tree.index(plan.a)
  idx_b = Tree.index(plan.b)
  print(io, "Coupling{$T}($idx_a ↔  $idx_b, $(size(plan)))")
end

function Base.show(io :: IO, ::MIME"text/plain", plan :: Coupling)
  Base.show(io, plan)
end

function Base.size(plan :: Coupling)
  n, m, b = size(plan.buffer)
  (n, m, n, m, b)
end


# ------ Transport plans ----------------------------------------------------- #

"""
    transport(plan_ab, i, j; conditional = false)
    transport(plan_ab, (i, j); conditional = false)

    transport(plan_ab, is, js; conditional = false)
    transport(plan_ab, (is, js); conditional = false)

    transport(workspace, a, b, i, j; conditional = false)
    transport(workspace, a, b, (i, j); conditional = false)

    transport(workspace, a, b, is, js; conditional = false)
    transport(workspace, a, b, (is, js); conditional = false)

Returns the evaluation of the coupling `plan_ab` at pixel `(i,j)` of node
`a`. If iterables `is` and `js` are provided, a vector of transport arrays
is returned.
If `conditional = true`, the transport arrays sum to one. If `workspace` as
well as two nodes `a` and `b` are provided, the corresponding coupling is
calculated implicitly.
"""
function transport(plan :: Coupling{T, F}, is, js; conditional = false) where {T, F}
  @assert length(is) == length(js)

  r = plan.reach
  n, m, _ = size(plan.buffer)
  eps_inv = one(F) / plan.eps

  weight = plan.weights[1]
  cc = convert(T, window(F, weight * plan.cost, r))

  potentials = T[plan.potentials_inter..., plan.potential_b]
  potential = first(potentials)
  outputs = T[similar(plan.buffer) for _ in 1:length(is)]

  for l in 1:length(is)
    i = is[l]
    j = js[l]
    da = @view plan.potential_a[i:i, j:j, :]

    # view coordinates for the potential
    i_range = max(1, i - r):min(n, i + r)
    j_range = max(1, j - r):min(m, j + r)

    # view coordinates for cost window
    iw_start = r - (i - i_range.start) + 1
    iw_end = r + (i_range.stop - i) + 1
    jw_start = r - (j - j_range.start) + 1
    jw_end = r + (j_range.stop - j) + 1

    iw_range = iw_start:iw_end
    jw_range = jw_start:jw_end

    c = @view(cc[iw_range, jw_range, :])
    dnext = @view(potential[i_range, j_range, :])

    if plan.logdomain
      fill!(plan.buffer, typemin(F))
      plan.buffer[i_range, j_range, :] .= da .+ dnext .- c
    else
      fill!(plan.buffer, zero(F))
      plan.buffer[i_range, j_range, :] .= da .* dnext .* exp.(.- c .* eps_inv)
    end
  
    for (potential, op!) in zip(potentials[2:end], plan.ops[2:end])
      op!(outputs[l], plan.buffer)
      if plan.logdomain
        plan.buffer .= outputs[l] .+ potential
      else
        plan.buffer .= outputs[l] .* potential
      end
    end
    outputs[l] .= plan.buffer

    if plan.logdomain
      outputs[l] .= exp.(outputs[l] .* eps_inv)
    end

    if conditional
      mass_inv = 1 ./ plan.marginal_a[i:i, j:j, :]
      outputs[l] .*= mass_inv
    end
  end

  outputs
end

function transport(plan :: Coupling{T, F}, i :: Integer, j :: Integer; kwargs...) where {T, F}
  transport(plan, [i], [j]; kwargs...)[1]
end

function transport(plan :: Coupling, ij :: Tuple; kwargs...)
  transport(plan, ij...; kwargs...)
end

function transport(w :: Workspace, a, b, args...; kwargs...)
  plan = Coupling(w, a, b)
  transport(plan, args...; kwargs...)
end


"""
    transport_window(coupling_ab, i, j; conditional = false)
    transport_window(coupling_ab, (i, j); conditional = false)

    transport_window(workspace, a, b, i, j; conditional = false)
    transport_window(workspace, a, b, (i, j); conditional = false)

Like `transport(coupling_ab, i, j; conditional)`, but it returns a window of
radius `reach` around the pixel posiiton `(i, j)`.

!!! note

     This function is only implemented for couplings between neighboring nodes.
"""
function transport_window(plan :: Coupling{T, F}, i, j; conditional = false) where {T, F}
  @assert Tree.has_neighbor(plan.a, plan.b) "Transport windows are only supported for neighboring nodes"

  n, m, _ = size(plan.buffer)
  r = plan.reach
  eps_inv = one(F) / plan.eps

  weight = plan.weights[1]
  cc = convert(T, window(F, weight * plan.cost, r))

  output = similar(cc)

  da = @view plan.potential_a[i:i, j:j, :]

  # view coordinates for potential b
  i_range = max(1, i - r):min(n, i + r)
  j_range = max(1, j - r):min(m, j + r)

  # view coordinates for the window
  iw_start = r - (i - i_range.start) + 1
  iw_end = r + (i_range.stop - i) + 1
  jw_start = r - (j - j_range.start) + 1
  jw_end = r + (j_range.stop - j) + 1

  iw_range = iw_start:iw_end
  jw_range = jw_start:jw_end

  c = @view(cc[iw_range, jw_range, :])
  dnext = @view(plan.potential_b[i_range, j_range, :])

  if plan.logdomain
    fill!(output, typemin(F))
    output[iw_range, jw_range, :] .= da .+ dnext .- c
  else
    fill!(output, zero(F))
    output[iw_range, jw_range, :] .= da .* dnext .* exp.(.- c .* eps_inv)
  end

  if plan.logdomain
    output .= exp.(output .* eps_inv)
  end

  if conditional
    mass_inv = 1 ./ plan.marginal_a[i:i, j:j, :]
    output .*= mass_inv
  end

  output
end

function transport_window(plan :: Coupling, ij :: Tuple; kwargs...)
  transport_window(plan, ij...; kwargs...)
end

function transport_window(w :: Workspace, a, b, args...; kwargs...)
  plan = Coupling(w, a, b)
  transport_window(plan, args...; kwargs...)
end

"""
    dense(coupling)

Return the full transport plan of `coupling` as a dense array. For large
problems, this will take an prohibitive amount of memory. It is therefore
limited to problems with `n*m < 65536`.

!!! note

    This function is only implemented for couplings between neighboring nodes
"""
function dense(plan :: Coupling{T, F}) where {T, F}
  @assert Tree.has_neighbor(plan.a, plan.b) "Dense plans are only supported for neighboring nodes"
  n, m, b = size(plan.potential_a)
  @assert n*m < 65536 """
  Plan dimensions ($n,$m) are too large to create a dense plan.
  """
  weight = plan.weights[1]

  cc = matrix(F, weight * plan.cost, plan.reach)
  cc = convert(T, cc)
  cc = reshape(cc, n, m, n, m, 1)
  potential_a = reshape(plan.potential_a, n, m, 1, 1, b)
  potential_b = reshape(plan.potential_b, 1, 1, n, m, b)

  if plan.logdomain
    exponent = (potential_a .+ potential_b .- cc) ./ plan.eps
    exp.(exponent)
  else
    potential_a .* exp.(.- cc ./ plan.eps) .* potential_b
  end
end

