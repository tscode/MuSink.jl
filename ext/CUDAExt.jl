
module CUDAExt

using LinearAlgebra, SparseArrays

using CUDA
import CUDA.CUSPARSE: CuSparseMatrixCSR, CuSparseMatrixCSC

import MuSink
import MuSink: Workspace, Cost, SeparableCost
import MuSink: axes, window, matrix, matrix_axes, matrixCSC_axes
import MuSink.Ops
import MuSink.Remote

# Tell MuSink about new array types
function __init__()
  MuSink.register_atype!(:cuda32, CuArray{Float32})
  MuSink.register_atype!(:cuda64, CuArray{Float64})
end

# Specialize some MuSink functions
MuSink.drop_batchdim(x :: CuVector) = CUDA.@allowscalar x[1]
MuSink.sync_arrays(:: Type{<: CuArray}) = CUDA.synchronize()


# ----- CuArray based kernel operations -------------------------------------- #

# This is for integration into the Ops.test_* functions
Ops.register_operator!(Ops.logdomain, CuArray{Float32})
Ops.register_operator!(Ops.logdomain, CuArray{Float64})
Ops.register_operator!(Ops.expdomain_conv, CuArray{Float32})
Ops.register_operator!(Ops.expdomain_conv, CuArray{Float64})
Ops.register_operator!(Ops.expdomain_dense, CuArray{Float32})
Ops.register_operator!(Ops.expdomain_dense, CuArray{Float64})
Ops.register_operator!(Ops.expdomain_sparse, CuArray{Float32})
Ops.register_operator!(Ops.expdomain_sparse, CuArray{Float64})


function Ops.logdomain( T :: Type{CuArray{F}}
                      , dims :: NTuple{3, Int}
                      , cost :: Cost
                      , eps :: Real
                      , r :: Int ) where {F}

  # Used for cuda kernel compilation, only type matters
  dummy = convert(T, zeros(F, 1, 1, 1))

  # Cost window for the logdomain algorithm
  cc = convert(T, window(F, cost, r))

  # Compile and configure the actual cuda function
  cu = @cuda launch=false cukernel_logdomain!(dummy, dummy, cc, F(eps))
  config = launch_configuration(cu.fun)
  threads = min(prod(dims), config.threads)
  blocks = cld(prod(dims), threads)

  (out, x) -> cu(out, x, cc, F(eps); threads, blocks)
end


function Ops.logdomain( T :: Type{CuArray{F}}
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

  # Compile and configure the cuda functions
  cu_sep = @cuda launch=false cukernel_logdomain_sep!(buffer, buffer, ci, F(eps))
  config = launch_configuration(cu_sep.fun)
  threads_sep = min(prod(dims), config.threads)
  blocks_sep = cld(prod(dims), threads_sep)

  cu_sept = @cuda launch=false cukernel_logdomain_sept!(buffer, buffer, cj, F(eps))
  config = launch_configuration(cu_sept.fun)
  threads_sept = min(prod(dims), config.threads)
  blocks_sept = cld(prod(dims), threads_sept)

  (out, x) -> begin
    cu_sep(buffer, x, ci, F(eps); threads = threads_sep, blocks = blocks_sep)
    cu_sept(out, buffer, cj, F(eps); threads = threads_sept, blocks = blocks_sept)
  end
end

function Ops.expdomain_conv( T :: Type{CuArray{F}}
                           , dims :: NTuple{3, Int}
                           , cost :: Cost
                           , eps :: Real
                           , r :: Int ) where {F}

  # determine size of common image for fft padding
  n, m, b = dims
  l = max(n, m) + r + 1 # TODO: this should be optimized!

  input_padded = convert(T, zeros(F, l, l, b))
  output_padded = convert(T, zeros(F, l, l, b))
  kernel_padded = convert(T, zeros(F, l, l, 1))


  plan = CUFFT.plan_rfft(input_padded, [1,2])
  cc = convert(T, window(F, cost, r))
  cc .= exp.(.- cc ./ eps)
  
  kernel_padded[1:(2r+1), 1:(2r+1), 1] .= cc

  kernel_f = CUFFT.rfft(kernel_padded, [1,2])
  buffer_f = CUFFT.rfft(output_padded, [1,2])

  range_i = (r+1):(n+r)
  range_j = (r+1):(m+r)
  
  (out, x) -> begin
    input_padded[1:n, 1:m, :] .= x
    mul!(buffer_f, plan, input_padded)
    buffer_f .= buffer_f .* kernel_f
    ldiv!(output_padded, plan, buffer_f)
    out .= @view output_padded[range_i, range_j, :]
  end
end


function Ops.expdomain_dense( T :: Type{CuArray{F}}
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

function Ops.expdomain_dense( T :: Type{CuArray{F}}
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

function Ops.expdomain_sparse( :: Type{CuArray{F}}
                             , dims :: NTuple{3, Int}
                             , cost :: Cost
                             , eps :: Real
                             , r :: Int ) where {F}
  n, m, b = dims
  @assert size(cost) == (n, m)
  mat = matrixCSC(F, cost, r)
  vals = nonzeros(mat)
  vals .= exp.(.- vals ./ F(eps))
  mat = convert(CuSparseMatrixCSR, mat)
  (out, x) -> begin
    buf = reshape(out, n * m, b)
    x = reshape(x, n * m, b)
    mul!(buf, mat, x)
  end
end


function Ops.expdomain_sparse( T :: Type{CuArray{F}}
                             , dims :: NTuple{3, Int}
                             , cost :: SeparableCost
                             , eps :: Real
                             , r :: Int ) where {F}
  n, m, b = dims
  @assert size(cost) == (n, m)
  mi, mj = matrixCSC_axes(F, cost, r) 
  vi = nonzeros(mi)
  vj = nonzeros(mj)
  vi .= exp.(.- vi ./ F(eps))
  vj .= exp.(.- vj ./ F(eps))

  mi = convert(CuSparseMatrixCSR, mi)
  mj = convert(CuSparseMatrixCSR, mj)

  buffer = T(undef, n, m)

  # TODO: This can certainly be optimized...
  (out, x) -> begin
    for k in 1:b
      buffer .= mi * x[:,:,k]
      output = mj * buffer'
      out[:,:,k] .= output'
    end
  end
end


# ----- CUDA kernel implementations ------------------------------------------ #

"""
GPU kernel function implementing the inner sinkhorn loop
"""
function cukernel_logdomain!(out, db, cc, eps)

  # Image dimension and batchsize, expect the same shape for out and db
  n, m, _ = size(out)

  # Assume a square cost kernel of size (2r + 1)^2
  r = div(size(cc, 1) - 1, 2)

  # Minor performance advantage
  cc = CUDA.Const(cc)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Start index and block-based stride
  start = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x

  for index in start:stride:length(out)

    # For some reason, the Int32 cast accelerates the code... ?
    # At other places, like for r, it deteriorates performance...
    ia, ja, k = Int32.(Tuple(C[index]))

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

  # CUDA kernels must return `nothing`
  nothing
end

function cukernel_logdomain_sep!(out, db, ci, eps)

  # Image height
  n = size(db, 1)

  # Assume a square cost kernel of size (2r + 1)^2
  r = div(length(ci) - 1, 2)

  # Minor performance advantage
  ci = CUDA.Const(ci)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Start index and block-based stride
  start = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x

  for index in start:stride:length(out)

    ia, jb, k = Int32.(Tuple(C[index]))

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

function cukernel_logdomain_sept!(out, dab, cj, eps)

  # Image width
  m = size(dab, 2)

  # Assume a square cost kernel of size (2r + 1)^2
  r = div(length(cj) - 1, 2)

  # Minor performance advantage
  cj = CUDA.Const(cj)

  # Convert from linear to cartesian indexing
  C = CartesianIndices(out)

  # Since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  # Start index and block-based stride
  start = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = gridDim().x * blockDim().x

  for index in start:stride:length(out)

    ia, ja, k = Int32.(Tuple(C[index]))

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

end # CUDAExt
