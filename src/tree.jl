
"""
Node type. Carries information about children and its parent node.
Has an associated index that can be used to identify it within a tree.

Note that no dedicated `Tree` type exists. Instead, a root node, which has no
parent, encodes the tree-associated structure.
"""
mutable struct Node
  index :: Int
  parent :: Union{Nothing, Node}  # only root node has no parent
  children :: Vector{Node}
end

# -------- Basics and navigation --------------------------------------------- #

is_leaf(node :: Node) = isempty(node.children)
is_root(node :: Node) =  isnothing(node.parent)

"""
    root(a::Node)

Returns the root node of the tree that contains `a`.
"""
function root(node :: Node)
  while !is_root(node)
    node = node.parent
  end
  node
end

"""
    children(a::Node)

Returns all children nodes of `a`.
"""
children(node :: Node) = node.children

"""
    parent(a::Node)

Returns the parent node of `a`. Returns `nothing` is `a` is the root node.
"""
parent(node :: Node) = node.parent
index(node :: Node) = node.index

function neighbors(node)
  if is_root(node)
    node.children
  else
    vcat(node.parent, node.children)
  end
end

"""
    descendants(a::Node, include_node = true)

Returns all descendants of `a`, including `a` if `include_node = true` is
passed.
"""
function descendants(node :: Node, include_node = true)
  nodes = include_node ? Node[node] : Node[]
  cursors = Int[1]
  while true
    # Step down if more children need to be explored
    if cursors[end] <= length(node.children)
      node = node.children[cursors[end]]
      push!(nodes, node)
      push!(cursors, 1)

    # Step up if all children have been explored
    else
      pop!(cursors)
      isempty(cursors) && break
      node = node.parent
      cursors[end] += 1
    end
  end
  nodes
end

function edges_outwards(root :: Node)
  @assert Tree.is_root(root) "Expected root node"
  edges = Tuple{Node, Node}[]
  for parent in Tree.descendants(root)
    for child in parent.children
      push!(edges, (parent, child))
    end
  end
  edges
end

function edges_inwards(root :: Node)
  @assert Tree.is_root(root) "Expected root node"
  edges = Tuple{Node, Node}[]
  for parent in Tree.descendants(root)
    for child in parent.children
      push!(edges, (child, parent))
    end
  end
  edges
end

function edges(root :: Node)
  @assert Tree.is_root(root) "Expected root node"
  edges = Tuple{Node, Node}[]
  for parent in Tree.descendants(root)
    for child in parent.children
      push!(edges, (parent, child))
      push!(edges, (child, parent))
    end
  end
  edges
end

# function descendants(node :: Node, include_node = true)
#   if is_leaf(node)
#     [node]
#   else
#     desc = mapreduce(descendants, vcat, node.children)
#     include_node ? vcat(node, desc) : desc
#   end
# end

function has_descendant(node :: Node, desc :: Node, include_node = true)
  descendants = Tree.descendants(node, include_node)
  any(n -> n === desc, descendants)
end

function has_descendant(node :: Node, index :: Int, include_node = true)
  descendants = Tree.descendants(node, include_node)
  any(n -> n.index == index, descendants)
end

function has_neighbor(node :: Node, neighbor :: Node)
  neighbors = Tree.neighbors(node)
  any(n -> n === neighbor, neighbors)
end

function has_neighbor(node :: Node, index :: Int)
  neighbors = Tree.neighbors(node)
  any(n -> n.index == index, neighbors)
end

"""
    descendant(node, index, include_node = true)

Find the descendant of `node` with index `index`. If `include_node` is `true`,
`node` itself may be found this way.
"""
function descendant(node :: Node, index, include_node = true)
  descendants = Tree.descendants(node, include_node)
  position = findfirst(n -> n.index == index, descendants)
  isnothing(position) ? nothing : descendants[position]
end

function remove_child!(parent :: Node, child :: Node)
  position = findfirst(n -> n === child, parent.children)
  if isnothing(position)
    err = "Node $(child.index) is not a child of node $(parent.index)"
    ArgumentError(err) |> throw
  else
    deleteat!(parent.children, position)
  end
end

function remove_child!(parent :: Node, index :: Int)
  position = findfirst(n -> Tree.index(n) == index, parent.children)
  if isnothing(position)
    err = "Node $(index) is not a child of node $(parent.index)"
    ArgumentError(err) |> throw
  else
    deleteat!(parent.children, position)
  end
end

"""
    step_towards(a, b)

Find the neighboring node of `a` that leads to `b`.
"""
function step_towards(a :: Node, b :: Node)
  step_towards(a, Tree.index(b))
end

function step_towards(a :: Node, index_b :: Int)
  root = Tree.root(a)
  rooted_a = Tree.reroot(a)
  pos = findfirst(child -> has_descendant(child, index_b), rooted_a.children)

  if isnothing(pos)
    ArgumentError("Cannot step from node $(a.index) towards node $(index_b)") |> throw
  else
    index = Tree.index(rooted_a.children[pos])
    Tree.descendant(root, index)
  end
end


# -------- Rerooting --------------------------------------------------------- #

"""
    reroot(node)
    reroot(root, index)

Create a copy of the tree that `node` is part of, and set it as the tree's
root.
"""
function reroot(node :: Node)
  if is_root(node)
    copy(node)
  else
    # Copy the tree from root's perspective
    index = Tree.index(node)
    root_prev = copy(Tree.root(node))

    # Find the node to be the new root
    node = Tree.descendant(root_prev, index)

    # walk towards node from root_prev and successively make
    # the encountered nodes the temporary root 
    root_tmp = Tree.step_towards(root_prev, node)
    while true
      Tree.remove_child!(root_prev, root_tmp)
      push!(root_tmp.children, root_prev)
      root_prev.parent = root_tmp
      root_tmp.parent = nothing

      root_tmp === node && break

      root_prev = root_tmp
      root_tmp = Tree.step_towards(root_tmp, node)
    end

    node
  end
end

function reroot(root :: Node, index :: Int)
  node = Tree.descendant(root, index)
  reroot(node)
end


"""
    cut(a, b)

Cuts the edge from `a` to `b`. Returns a tuple `(root_a, root_b)`, which are
root nodes for two new trees with roots at `a` respectively `b`.
"""
function cut(a :: Node, b :: Node)
  @assert Tree.has_neighbor(a, b) "Can only split between neighboring nodes"

  root_a = Tree.reroot(a)
  root_b = Tree.reroot(b)

  Tree.remove_child!(root_a, Tree.index(b))
  Tree.remove_child!(root_b, Tree.index(a))

  root_a, root_b
end

# -------- Tree construction ------------------------------------------------- #

"""
    Root(index = 1)

Returns a root node with index `index` and no children.
"""
Root(index :: Int = 1) = Node(index, nothing, [])

"""
    new_child!(parent, index)

Add a new child with index `index` to the node `parent`. Returns
the created child node.
"""
function new_child!(parent :: Node, index :: Int; check_index = true)
  if check_index
    root = Tree.root(parent)
    @assert !has_descendant(root, index) "Index $index already exists in this tree"
  end
  child = Node(index, parent, [])
  push!(parent.children, child)
  child
end

function new_child!(parent :: Node)
  descendants = Tree.descendants(Tree.root(parent))
  index = maximum(Tree.index, descendants) + 1
  child = Node(index, parent, [])
  push!(parent.children, child)
  child
end

"""
    Sequence(n)

Returns the root node of a linear sequence of `n` nodes, indexed from
`1` to `n`.
"""
function Sequence(n :: Int)
  @assert n >= 1 "Cannot create sequence with $n elements"
  root = Tree.Root(1)
  node = root
  for index in 2:n
    node = Tree.new_child!(node, index, check_index = false)
  end
  root
end

function Star(n :: Int)
  @assert n > 1 "Cannot create star with $n elements"
  root = Tree.Root(1)
  for index in 2:n
    Tree.new_child!(root, index, check_index = false)
  end
  root
end


# -------- Node- and edge-associated data ------------------------------------ #

"""
    node_store(root, template)

Return a dictionary that contains a copy of `template` for each node of the
tree centered around `root`.
"""
function node_store(root :: Node, template)
  @assert is_root(root) "Node store expects root node"
  store = Dict{Node, typeof(template)}()
  for node in descendants(root)
    store[node] = copy(template)
  end
  store
end

"""
    edge_store(root, template)

Return a dictionary that contains a copy of `template` for each edge of the
tree centered around `root`. The keys are pairs of nodes.
"""
function edge_store(root :: Node, template)
  @assert is_root(root) "Edge store expects root node"
  store = Dict{NTuple{2, Node}, typeof(template)}()
  for node in descendants(root)
    if !is_root(node)
      parent = node.parent
      store[(node, parent)] = copy(template)
      store[(parent, node)] = copy(template)
    end
  end
  store
end


# -------- Base functionality ------------------------------------------------ #

function stringify(node, mark = nothing)
  index = Tree.index(node)
  main = index == mark ? "$index ←" : "$index"
  if !Tree.is_leaf(node)
    nchildren = length(node.children)
    lines = mapreduce(vcat, 1:nchildren) do pos_child
      children_lines = stringify(node.children[pos_child], mark)
      len = length(children_lines)
      map(1:len) do pos
        if pos == 1 && pos_child == nchildren
          pre = "\u2514\u2574"
        elseif pos == 1
          pre = "\u251c\u2574"
        elseif pos_child == nchildren
          pre = "  "
        else
          pre = "\u2502 "
        end
        pre * children_lines[pos]
      end
    end
    vcat(main, lines)
  else
    [main]
  end
end

function Base.show(io :: IO, node :: Node)
  index = Tree.index(node)
  print(io, "Node($index)")
  # parent = Tree.parent(node)
  # parent_str = isnothing(parent) ? "root" : "parent = $(Tree.index(parent))"
  # children = map(Tree.index, Tree.children(node))
  # children_str = isempty(children) ? "leaf" : "children = $children"
  # print(io, "Node($(node.index), $parent_str, $children_str)")
end

function Base.show(io :: IO, ::MIME"text/plain", node :: Node)
  Base.show(io, node)
  root = Tree.root(node)
  if length(root) <= 20
    lines = stringify(root, Tree.index(node))
    println(io, ":")
    for line in lines
      println(io, " ", line)
    end
  end
end

Base.hash(node :: Node) = Base.hash(node.index)
Base.:(==)(a :: Node, b :: Node) = (a.index == b.index)
Base.length(node :: Node) = Base.length(descendants(node))

function copy_setparent(node, parent)
  new_node = Node(node.index, parent, [])
  new_node.children = map(n -> copy_setparent(n, new_node), node.children)
  new_node
end

function Base.copy(node :: Node)
  if is_root(node)
    root = Node(node.index, nothing, [])
    root.children = map(n -> Tree.copy_setparent(n, root), node.children)
    root
  else
    root = Base.copy(Tree.root(node))
    Tree.descendant(root, node.index)
  end
end
