
module oneAPIExt

using LinearAlgebra

using oneAPI
using oneAPI.oneL0

import MuSink
import MuSink: Workspace, Cost, SeparableCost
import MuSink: axes, window, matrix, matrix_axes, matrixCSC_axes
import MuSink.Ops
import MuSink.Remote

# Tell MuSink about new array types
function __init__()
  MuSink.register_atype!(:one32, oneArray{Float32})
  MuSink.register_atype!(:one64, oneArray{Float64})
end

MuSink.drop_batchdim(x :: oneVector) = oneAPI.@allowscalar x[1]
MuSink.sync_arrays(:: Type{<: oneArray}) = oneAPI.synchronize()

Ops.register_operator!(Ops.logdomain, oneArray{Float32})
Ops.register_operator!(Ops.logdomain, oneArray{Float64})
Ops.register_operator!(Ops.expdomain_dense, oneArray{Float32})
Ops.register_operator!(Ops.expdomain_dense, oneArray{Float64})

function Ops.logdomain( T :: Type{oneArray{F}}
                      , dims :: NTuple{3, Int}
                      , cost :: Cost
                      , eps :: Real
                      , r :: Int ) where {F}

  # Used for kernel compilation, only type matters
  dummy = convert(T, zeros(F, 1, 1, 1))

  # Cost window for the logdomain algorithm
  cc = convert(T, window(F, cost, r))
  kernel = @oneapi launch=false onekernel_logdomain!(dummy, dummy, cc, F(eps))

  # Launch parameters for the kernel
  elements = prod(dims)
  items = suggest_groupsize(kernel.fun, elements).x
  groups = cld(elements, items)

  (out, x) -> begin
    kernel(out, x, cc, F(eps); items = items, groups = groups)
  end
end

function Ops.logdomain( T :: Type{oneArray{F}}
                      , dims :: NTuple{3, Int}
                      , cost :: SeparableCost
                      , eps :: Real
                      , r :: Int ) where {F}

  # Cost window for the logdomain algorithm
  ci, cj = axes(F, cost, r)
  ci = convert(T, ci)
  cj = convert(T, cj)

  # Buffer to hold intermediate potentials
  buffer = T(undef, dims)

  # Compile and configure the gpu kernels
  kernel_sep = @oneapi launch=false onekernel_logdomain_sep!(buffer, buffer, ci, F(eps))
  kernel_sept = @oneapi launch=false onekernel_logdomain_sept!(buffer, buffer, cj, F(eps))

  elements = prod(dims)
  items = suggest_groupsize(kernel_sep.fun, elements).x
  groups = cld(elements, items)

  (out, x) -> begin
    kernel_sep(buffer, x, ci, F(eps); items = items, groups = groups)
    kernel_sept(out, buffer, cj, F(eps); items = items, groups = groups)
  end
end


function Ops.expdomain_dense( T :: Type{oneArray{F}}
                            , dims :: NTuple{3, Int}
                            , cost :: Cost
                            , eps :: Real
                            , r :: Int ) where {F}
  n, m, b = dims
  @assert size(cost) == (n, m)
  mat = convert(T, matrix(F, cost, r))
  mat .= exp.(.- mat ./ F(eps))

  (out, x) -> begin
    buf = reshape(out, n * m, b)
    x = reshape(x, n * m, b)
    mul!(buf, mat, x)
  end
end

function Ops.expdomain_dense( T :: Type{oneArray{F}}
                            , dims :: NTuple{3, Int}
                            , cost :: SeparableCost
                            , eps :: Real
                            , r :: Int ) where {F}
  n, m, b = dims
  @assert size(cost) == (n, m)
  mi, mj = matrix_axes(F, cost, r) 

  mi = convert(T, mi)
  mj = convert(T, mj)

  mi .= exp.(.- mi ./ F(eps))
  mj .= exp.(.- mj ./ F(eps))

  buffer = T(undef, n, m)

  (out, x) -> begin
    for k in 1:b
      mul!(buffer, mi, @view(x[:,:,k]))
      mul!(@view(out[:,:,k]), buffer, mj)
    end
  end
end


"""
GPU kernel function implementing the inner sinkhorn loop
"""
function onekernel_logdomain!(out, db, cc, eps)

  # Image dimension and batchsize, expect the same shape for out and db
  n, m, _ = size(out)

  # Assume a square cost kernel of siz (2r + 1)^2
  r = div(size(cc, 1) - 1, 2)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Index of the kernel operation
  index = get_global_id()

  if index <= length(out)

    # For some reason, the Int32 cast accelerates the code... ?
    # At other places, like for r, it deteriorates performance...
    ia, ja, k = Tuple(C[index])

    # maximum value
    mx = typemin(eps)
    for jr in -r:r
      jb = ja + jr
      if 1 <= jb <= m
        for ir in -r:r
          ib = ia + ir
          if 1 <= ib <= n
            @inbounds c = cc[ia - ib + r + 1, ja - jb + r + 1]
            @inbounds val = (db[ib, jb, k] - c) * eps_inv
            mx = max(mx, val)
          end 
        end
      end
    end

    # exponential sum
    s = zero(eps)
    if !isinf(mx)
      for jr in -r:r
        jb = ja + jr
        if 1 <= jb <= m
          for ir in -r:r
            ib = ia + ir
            if 1 <= ib <= n
              @inbounds c = cc[ia - ib + r + 1, ja - jb + r + 1]
              @inbounds val = (db[ib, jb, k] - c) * eps_inv
              s += exp(val - mx)
            end
          end
        end
      end
    end

    # logarithm of the exponential sum times eps
    out[ia, ja, k] = eps * (log(s) + mx)

  end

  # kernels must return `nothing`
  nothing
end

function onekernel_logdomain_sep!(out, db, ci, eps)

  # Image height
  n = size(db, 1)

  # Assume a square cost kernel of size (2r + 1)^2
  r = div(length(ci) - 1, 2)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Index of the kernel operation
  index = get_global_id()

  if index <= length(out)

    ia, jb, k = Tuple(C[index])

    mx = typemin(eps)
    for ir in -r:r
      ib = ia + ir
      if 1 <= ib <= n
        @inbounds c = ci[ib - ia + r + 1]
        @inbounds val = (db[ib, jb, k] - c) * eps_inv
        mx = max(mx, val)
      end
    end

    s = zero(eps)
    if !isinf(mx)
      for ir in -r:r
        ib = ia + ir
        if 1 <= ib <= n
          @inbounds c = ci[ib - ia + r + 1]
          @inbounds val = (db[ib, jb, k] - c) * eps_inv
          s += exp(val - mx)
        end
      end
    end

    out[ia, jb, k] = eps * (log(s) + mx)
  end

  nothing
end

function onekernel_logdomain_sept!(out, dab, cj, eps)

  # Image width
  m = size(dab, 2)

  # Assume a square cost kernel of size (2r + 1)^2
  r = div(length(cj) - 1, 2)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Index of the kernel operation
  index = get_global_id()

  if index <= length(out)

    ia, ja, k = Tuple(C[index])

    mx = typemin(eps)
    for jr in -r:r
      jb = ja + jr
      if 1 <= jb <= m
        @inbounds c = cj[jb - ja + r + 1]
        @inbounds val = (dab[ia, jb, k] - c) * eps_inv
        mx = max(mx, val)
      end
    end

    s = zero(eps)
    if !isinf(mx)
      for jr in -r:r
        jb = ja + jr
        if 1 <= jb <= m
          @inbounds c = cj[jb - ja + r + 1]
          @inbounds val = (dab[ia, jb, k] - c) * eps_inv
          s += exp(val - mx)
        end
      end
    end

    out[ia, ja, k] = eps * (log(s) + mx)
  end

  nothing
end

end # oneAPIExt
