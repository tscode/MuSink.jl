
"""
MuSink problem type that defines the fixed parts of an UMOT problem.

Contains the target and reference measures as well as the cost topology and marginal penalty.
"""
struct Problem
  targets :: Dict{Node, Array{Float64, 3}}
  references :: Dict{Node, Array{Float64, 3}}
  dims :: NTuple{3, Int}
  root :: Node

  # default cost and penalty
  cost :: Cost
  penalty :: Penalty
end

"""
    Problem(targets [; cost, penalty, references, reference_mass])

Default constructor of a MuSink problem.

The arguments `targets` and `references` are expected to be of type `Dict{Node,
Array{Float64, 3}}`, assigning target and reference measures to each node of
the tree. If a `reference_mass` is specified, the references will be scaled such
that their product measure has mass `reference_mass`.
"""
function Problem( targets :: Dict{Node, Array{Float64, 3}}
                ; cost :: Union{Nothing, Cost} = nothing
                , penalty :: Penalty = TotalVariation()
                , references = default_references(targets)
                , reference_mass :: Real = -1.0
                )

  mvals = values(targets) |> collect
  mkeys = keys(targets) |> collect

  # Some sanity checks
  @assert length(mvals) > 1 "At least two marginal measures are required"
  dims = size(first(mvals))
  for marginal in values(mvals)
    @assert size(marginal) == dims "All targets must have the same shape"
  end

  # Find the root node
  root_idx = findfirst(Tree.is_root, mkeys)
  @assert !isnothing(root_idx) "No root element found"
  root = mkeys[root_idx]

  # Make sure that the root is actually the root of the given nodes
  @assert length(targets) == length(root) "Invalid target nodes"
  @assert Set(mkeys) == Set(Tree.descendants(root)) "Invalid target nodes"

  # Default l2-based cost
  n, m, _ = dims
  if isnothing(cost)
    cost = Lp(n, m; p = 2)
  end

  # Check consistency of cost
  @assert size(cost) == (n, m) "Inconsistent default cost size"

  if references == :counting
    references = default_references_counting(targets)
  elseif references == :targets
    references = default_references_targets(targets)
  else
    rnodes = keys(references)
    @assert references isa Dict{Node, Array{Float64, 3}}
    @assert Set(rnodes) == Set(Tree.descendants(root)) "Invalid reference nodes"
  end

  # If a positive reference mass has been provided, rescale the
  # reference measures to hit the mass
  rescale_references!(references, reference_mass)

  Problem(targets, references, dims, root, cost, penalty)
end

function Base.show(io :: IO, p :: Problem)
  nmarg = length(p.targets)
  print(io, "MUSink.Problem($nmarg, $(p.dims))")
end

Base.show(io :: IO, ::MIME"text/plain", p :: Problem) = Base.show(io, p)

function Base.size(p :: Problem)
  size(first(p.targets)[2])
end

function Base.length(p :: Problem)
  length(p.targets)
end


# ------ Convenience constructor --------------------------------------------- #

function adapt_type(vec :: AbstractVector)
  vec = reshape(vec, size(vec)..., 1, 1)
  convert(Array{Float64, 3}, vec)
end

function adapt_type(mat :: AbstractMatrix)
  mat =  reshape(mat, size(mat)..., 1)
  convert(Array{Float64, 3}, mat)
end

function adapt_type(mat :: AbstractArray{F, 3}) where {F}
  convert(Array{Float64, 3}, mat)
end

function adapt_type(mat :: AbstractArray{F, N}) where {F, N}
  throw(ArgumentError("invalid dimension $N (expected 1 <= N <= 3)"))
end

# Adapt the type of the targets
function node_dict(root :: Node, targets :: AbstractDict{Node})
  Dict{Node, Array{Float64, 3}}(
    node => adapt_type(target) for (node, target) in targets
  )
end

# Convert dicts of the form index => targets to target node dicts
function node_dict(root :: Node, targets :: AbstractDict{Int})
  dict = Dict{Node, Any}(
    Tree.descendant(root, index) => target for (index, target) in targets
  )
  node_dict(root, dict)
end

# Convert vectors to target node dicts
function node_dict(root :: Node, targets :: AbstractVector)
  nodes = Tree.descendants(root)
  indices = Tree.index.(nodes)
  sort!(indices)

  @assert length(nodes) == length(targets)

  dict = Dict{Int, Any}(
    indices[i] => targets[i] for i in 1:length(targets)
  )

  node_dict(root, dict)
end

"""
    Problem(root, targets [; cost, penalty, references, reference_mass])

Convenience constructor of a MuSink problem. The arguments `targets` and
`references` can be vectors, integer dictionaries, or node dictionaries and
must be of the same size as the tree defined by `root`. 
"""
function Problem(root :: Node, targets; references = nothing, kwargs...)
  targets = node_dict(root, targets)
  if isnothing(references)
    references = default_references(targets)
  else
    references = node_dict(root, references)
  end
  Problem(targets; references, kwargs...)
end


# ------ Reference measures -------------------------------------------------- #

"""
    default_references_counting(targets, product_mass)

Create uniform reference measures for `problem`. Each measure is normalized
such that the product measure of all references has the mass `product_mass`.
"""
function default_references_counting(targets, product_mass :: Float64 = -1.0)
  n, m, _ = size(first(targets)[2])
  if product_mass <= 0
    mass = Float64(n * m)
  else
    nmargs = length(targets)
    mass = product_mass^(1/nmargs)
  end

  refs = Dict{Node, Array{Float64, 3}}()
  for (node, marginal) in targets
    ref = similar(marginal)
    ref .= mass / (n * m)
    refs[node] = ref
  end
  refs
end

"""
    default_references_targets(targets, product_mass)

Create reference measures for `problem` proportional to the target marginal
measures. The measure are rescaled such that the product of all references has
total mass `product_mass`.
"""
function default_references_targets(targets, product_mass :: Float64 = -1.0)
  refs = Dict{Node, Array{Float64, 3}}()
  for (node, target) in targets
    ref = copy(target)
    refs[node] = ref
  end
  rescale_references!(refs, product_mass)
  refs
end

"""
    default_references(targets, product_mass; counting = true)

Create reference measures for `problem` such that the product of all references
has total mass `product_mass`. If `counting`, then the reference measures are
proportional to the counting measure. Otherwise, they are proportional to the
target measures.
"""
function default_references(args...; counting = true)
  if counting
    default_references_counting(args...)
  else
    default_references_targets(args...)
  end
end

function rescale_references!(references, product_mass :: Real)
  nrefs = length(references)
  if product_mass <= 0
    scaling = 1.0
  else
    mvals = values(references)
    masses = map(mvals) do marginal
      sum(marginal, dims = (1, 2))
    end
    total_mass = reduce(.*, masses)
    scaling = (Float64(product_mass) ./ total_mass).^(1/nrefs)
  end
  for reference in values(references)
    reference .*= scaling
  end
end

function reference_product_mass(p :: Problem; scalar = true)
  b = size(p)[3]
  mass = ones(1,1,b)
  for ref in values(p.references)
    mass .*= sum(ref, dims = (1, 2))
  end
  if b == 1 && scalar
    mass[1,1,1]
  else
    dropdims(mass, dims = (1,2))
  end
end

# TODO
function reach_correction(references, reach :: Int)
  error("Not implemented yet")
end

"""
    target(p::Problem, a)

Returns the target measure of `p` at node `a`.
"""
function target(p :: Problem, node :: Node)
  @assert Tree.has_descendant(p.root, node)
  p.targets[node]
end

function target(p :: Problem, index :: Int)
  @assert Tree.has_descendant(p.root, index)
  node = Tree.descendant(p.root, index)
  target(p, node)
end

"""
    target(p::Problem, a)

Returns the reference measure of `p` at node `a`.
"""
function reference(p::Problem, node :: Node)
  @assert Tree.has_descendant(p.root, node)
  p.references[node]
end

function reference(p::Problem, index :: Node)
  @assert Tree.has_descendant(p.root, index)
  node = Tree.descendant(p.root, index)
  reference(p, node)
end

"""
    nodes(p::Problem)

Returns all nodes of the problem cost tree.
"""
nodes(p::Problem) = Tree.descendants(p.root)

"""
    edges(p::Problem)

Returns all (bidirectional) edges of the problem cost tree.
"""
edges(p::Problem) = Tree.edges(p.root)
