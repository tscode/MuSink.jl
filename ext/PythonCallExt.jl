
module PythonCallExt

using MuSink
using MuSink: Node
using PythonCall

# Adapt array types that come from python in a dict, such that they have lost
# their original type
function adapt_type(val :: Py)
  val = pyconvert(Array{Float64}, val)
  adapt_type(val)
end

# Convert targets in pydict format
function node_dict(root :: Node, targets :: PyDict)
  dict = Dict{Node, Any}()
  for (index, target) in targets
    index = pyconvert(Int, index)
    node = Tree.descendant(root, index)
    dict[node] = target
  end
  node_dict(root, dict)
end

function Barycenter(T, targets :: PyList; kwargs...)
  targets = pyconvert.(Array, targets)
  Barycenter(T, targets; kwargs...)
end

function Barycenter(targets :: PyList; kwargs...)
  targets = pyconvert.(Array, targets)
  Barycenter(targets; kwargs...)
end

function Chain(T, targets :: PyList; kwargs...)
  targets = pyconvert.(Array, targets)
  Chain(T, targets; kwargs...)
end

function Chain(targets :: PyList; kwargs...)
  targets = pyconvert.(Array, targets)
  Chain(targets; kwargs...)
end

function Reductions.Reduction(lambda :: Py)
  f = (di, dj, c) -> pyconvert(Float64, lambda(di, dj, c))
  Reductions.Reduction(f)
end

function Reductions.Reduction(lambda_i :: Py, lambda_j :: Py)
  fi = di -> pyconvert(Float64, lambda_i(di))
  fj = dj -> pyconvert(Float64, lambda_j(dj))
  Reductions.Reduction(fi, fj)
end

end # PythonCallExt
