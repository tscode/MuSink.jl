# ------- Logdomain Kernels / Operators -------------------------------------- #

"""
CPU kernel function implementing the inner sinkhorn loop
"""
function cpukernel_logdomain(ia, ja, k, db, cc, eps)

  # image dimension and batchsize, expect the same shape for out and db
  n, m, _ = size(db)

  # assume a quadratic cost kernel of size (2r + 1)^2
  r = div(size(cc, 1) - 1, 2)

  # since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  ibmin = max(1, ia - r)
  ibmax = min(n, ia + r)
  jbmin = max(1, ja - r)
  jbmax = min(m, ja + r)

  # maximum value
  mx = typemin(eps)
  @inbounds for jb in jbmin:jbmax, ib in ibmin:ibmax
    c = cc[ia - ib + r + 1, ja - jb + r + 1]
    val = (db[ib, jb, k] - c) * eps_inv
    mx = max(mx, val)
  end

  # exponential sum
  s = zero(eps)
  if !isinf(mx)
    @inbounds for jb in jbmin:jbmax, ib in ibmin:ibmax
      c = cc[ia - ib + r + 1, ja - jb + r + 1]
      val = (db[ib, jb, k] - c) * eps_inv
      s += exp(val - mx)
    end
  end

  # logarithm of the exponential sum times eps
  eps * (log(s) + mx)
end

"""
CPU kernel function implementing the inner sinkhorn loop
"""
function cpukernel_logdomain_sep!(out, k, db, ci, eps)
  # image dimension and batchsize, expect the same shape for out and db
  n, m, _ = size(db)

  # assume a quadratic cost kernel of size (2r + 1)^2
  r = div(length(ci) - 1, 2)

  # since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  Threads.@threads for jb in 1:m
    for ia in 1:n
      ibmin = max(1, ia - r)
      ibmax = min(n, ia + r)
   
      mx = typemin(eps)
      @inbounds for ib in ibmin:ibmax
        c = ci[ib - ia + r + 1]
        val = (db[ib, jb, k] - c) * eps_inv
        mx = max(mx, val)
      end

      s = zero(eps)
      if !isinf(mx)
        @inbounds for ib in ibmin:ibmax
          c = ci[ib - ia + r + 1]
          val = (db[ib, jb, k] - c) * eps_inv
          s += @inline exp(val - mx)
        end
      end

      @inbounds out[jb, ia, k] = eps * (log(s) + mx)
    end
  end
end

function cpukernel_logdomain_sept!(out, k, dat, cj, eps)
  # image dimension and batchsize, expect the same shape for out and db
  m, n, _ = size(dat)

  # assume a quadratic cost kernel of size (2r + 1)^2
  r = div(length(cj) - 1, 2)

  # since division is more expensive than multiplication, store inverse value
  eps_inv = one(eps) / eps

  Threads.@threads for ia in 1:n
    for ja in 1:m
      jbmin = max(1, ja - r)
      jbmax = min(m, ja + r)
   
      mx = typemin(eps)
      @inbounds for jb in jbmin:jbmax
        c = cj[jb - ja + r + 1]
        val = (dat[jb, ia, k] - c) * eps_inv
        mx = max(mx, val)
      end

      s = zero(eps)
      if !isinf(mx)
        @inbounds for jb in jbmin:jbmax
          c = cj[jb - ja + r + 1]
          val = (dat[jb, ia, k] - c) * eps_inv
          s += @inline exp(val - mx)
        end
      end

      @inbounds out[ia, ja, k] = eps * (log(s) + mx)
    end
  end
end



"""
    logdomain(T, dims, cost, eps)

One inner sinkhorn operation in the logarithmic domain. Works with array type
`T` on input with dimension `dims`. The regularization is specified by `eps`.
The cost to be used is determined by `cost`.

The returning operator takes two arguments (`buf` and `x`), and
performs the calculation

    buf_{i} = eps * log[ sum_{i} exp(x_i - cost_{i-j} / eps) ]

Compared to calculations in the expdomain, this approach is (a) slower but (b)
remains stable for small values of `eps`.
"""
function logdomain( T :: Type{Array{F}}
                  , dims :: NTuple{3, Int}
                  , cost :: Cost
                  , eps :: Real
                  , r :: Int ) where {F}

  n, m, b = dims
  @assert size(cost) == (n, m)
  cc = convert(T, window(F, cost, r))

  (out, x) -> begin
    for k in 1:b
      Threads.@threads for ja in 1:m
        for ia in 1:n
          @inbounds out[ia, ja, k] = cpukernel_logdomain(ia, ja, k, x, cc, eps)
        end
      end
    end
  end
end

function logdomain( :: Type{Array{F}}
                  , dims :: NTuple{3, Int}
                  , cost :: SeparableCost
                  , eps :: Real
                  , r :: Int ) where {F}

  n, m, b = dims
  @assert size(cost) == (n, m)
  ci, cj = axes(F, cost, r)
  buffer = zeros(F, m, n, b)

  (out, x) -> begin
    for k in 1:b
      cpukernel_logdomain_sep!(buffer, k, x, ci, F(eps))
      cpukernel_logdomain_sept!(out, k, buffer, cj, F(eps))
    end
  end
end


# ------- Expdomain Kernels / Operators -------------------------------------- #

"""
    zero_pad(arr, l)

Pad an array `arr` of dimension 2 or 3 with zeros. The resulting array
will hold l elements in the first and second dimension, while the third
dimension (considered to be batching) is not modified.
"""
function zero_pad(vec :: Array{F, 2}, l) where {F}
  n, b = size(vec)
  @assert n <= l
  out = zeros(F, l, b)
  out[1:n, :] .= vec
  out
end

function zero_pad(mat :: Array{F, 3}, l) where {F}
  n, m, b = size(mat)
  @assert n <= l && m <= l
  out = zeros(F, l, l, b)
  out[1:n, 1:m, :] .= mat
  out
end

"""
  expdomain(T, dims, cost, eps)

One inner sinkhorn operation in the exponential domain. Works with array type
`T` on input with dimension `dims`. The regularization is specified by `eps`.
The cost to be used is determined by the kernel `k`.

The returning operator takes two arguments (`buf` and `x`), and
efficiently performs the calculation

    buf_{i} = sum_{i} exp(- cost_{j-i} / eps) * x_{i}.

Compared to calculations in the logdomain, this approach is (a) faster but (b)
instable for small `eps`.
"""
function expdomain(T, dims, cost, eps, r)
  if cost isa SeparableCost
    expdomain_dense(T, dims, cost, eps, r)
  else
    expdomain_conv(T, dims, cost, eps, r)
  end
end


function expdomain_conv( :: Type{Array{F}}
                       , dims :: NTuple{3, Int}
                       , cost :: Cost
                       , eps :: Real
                       , r :: Int ) where {F}

  # determine size of common image for fft padding
  n, m, b = dims
  l = max(n, m) + r + 1

  input_padded = zeros(F, l, l, b)
  output_padded = zeros(F, l, l, b)
  kernel_padded = zeros(F, l, l, 1)

  plan = FFTW.plan_rfft(input_padded, [1,2])
  cc = window(F, cost, r)
  cc .= exp.(.- cc ./ eps)

  kernel_padded[1:(2r+1), 1:(2r+1), 1] .= cc

  kernel_f = FFTW.rfft(kernel_padded, [1,2])
  buffer_f = FFTW.rfft(output_padded, [1,2])

  i_range = (r+1):(n+r)
  j_range = (r+1):(m+r)
  
  (out, x) -> begin
    input_padded[1:n, 1:m, :] .= x
    mul!(buffer_f, plan, input_padded)
    buffer_f .= buffer_f .* kernel_f
    ldiv!(output_padded, plan, buffer_f)
    out .= @view output_padded[i_range, j_range, :]
  end
end

# ----- Alternative implementations of the kernel operation ------------------ #

function expdomain_dense( :: Type{Array{F}}
                       , dims :: NTuple{3, Int}
                       , cost :: Cost
                       , eps :: Real
                       , r :: Int ) where {F}

  n, m, b = dims
  @assert size(cost) == (n, m)
  mat = matrix(F, cost, r) 
  mat .= exp.(.- mat ./ F(eps))
  (out, x) -> begin
    buf = reshape(out, n * m, b)
    x = reshape(x, n * m, b)
    mul!(buf, mat, x)
  end
end

function expdomain_dense( T :: Type{Array{F}}
                        , dims :: NTuple{3, Int}
                        , cost :: SeparableCost
                        , eps :: Real
                        , r :: Int ) where {F}

  n, m, b = dims
  @assert size(cost) == (n, m)
  mi, mj = matrix_axes(F, cost, r) 
  mi .= exp.(.- mi ./ F(eps))
  mj .= exp.(.- mj ./ F(eps))

  buffer = T(undef, n, m)

  (out, x) -> begin
    for k in 1:b
      # buffer .= mi * x[:,:,k]
      # out[:,:,k] .= buffer * mj
      mul!(buffer, mi, @view(x[:,:,k]))
      mul!(@view(out[:,:,k]), buffer, mj)
    end
  end
end

function expdomain_sparse( :: Type{Array{F}}
                         , dims :: NTuple{3, Int}
                         , cost :: Cost
                         , eps :: Real
                         , r :: Int ) where {F}

  n, m, b = dims
  @assert size(cost) == (n, m)
  mat = matrixCSC(F, cost, r) 
  vals = nonzeros(mat)
  vals .= exp.(.- vals ./ F(eps))
  (out, x) -> begin
    buf = reshape(out, n * m, b)
    x = reshape(x, n * m, b)
    mul!(buf, mat, x)
  end
end

function expdomain_sparse(T :: Type{Array{F}}
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

  buffer = T(undef, n, m)

  (out, x) -> begin
    for k in 1:b
      buffer .= mi * x[:,:,k]
      out[:,:,k] .= buffer * mj
    end
  end
end

const OPERATORS = Any[
  (logdomain, Array{Float32}),
  (logdomain, Array{Float64}),
  (expdomain_conv, Array{Float32}),
  (expdomain_conv, Array{Float64}),
  (expdomain_dense, Array{Float32}),
  (expdomain_dense, Array{Float64}),
  (expdomain_sparse, Array{Float32}),
  (expdomain_sparse, Array{Float64}),
]

function register_operator!(op, atype)
  push!(OPERATORS, (op, atype))
end

function test_correctness(; cost = Lp, n = 30, m = 29, b = 1, r = 5, eps = 1.)
 
  costs = cost(n, m)
  # costs = MuSink.GenericCost(costs)
  inp = rand(Float32, n, m, b)

  outputs = map(OPERATORS) do (operator, T)
    F = eltype(T)
    if operator == logdomain
      op! = operator(T, (n, m, b), costs, eps, r)
      input = convert(T, inp)
      output = copy(input)
      op!(output, input)
      convert(Array, output)
    else
      op! = operator(T, (n, m, b), costs, eps, r)
      input = convert(T, exp.(inp ./ F(eps)))
      output = copy(input)
      op!(output, input)
      convert(Array, F(eps) .* log.(output))
    end
  end

  idx = collect(1:length(OPERATORS))
  diffs = zeros(length(OPERATORS), length(OPERATORS))

  for i in idx, j in idx
    diffs[i,j] = maximum(abs, outputs[i] .- outputs[j])
    diffs[i,j] /= mean(abs, outputs[i])
  end

  diffs
end


function test_logdomain(
                       ; cost = Lp
                       , n = 100
                       , m = n
                       , b = 1
                       , r = div(n, 10) + 1
                       , eps = 0.1 )
  Random.seed!(1)

  costs = cost(n, m)

  print("generating data... ")
  dt = @elapsed begin
    out = rand(n, m, b)
    inp = rand(n, m, b)
  end
  println("$dt seconds")

  for (op, T) in OPERATORS
    if op != logdomain
      continue
    end

    println("$T:")
    print("  converting data... ")
    dt = @elapsed begin
      out = convert(T, rand(n, m, b))
      inp = convert(T, rand(n, m, b))
    end
    println("$dt seconds")

    print("  compiling operation... ")
    dt = @elapsed begin
      op! = logdomain(T, (n, m, b), costs, eps, r)
    end
    println("$dt seconds")

    print("  first execution... ")
    dt = @elapsed (op!(out, inp); MuSink.sync_arrays(T))
    println("$dt seconds")
    
    print("  second execution... ")
    dt = @elapsed (op!(out, inp); MuSink.sync_arrays(T))
    println("$dt seconds")
    println()
  end

  nothing
end

function test_expdomain( operator
                       , F = Float32
                       ; cost = Lp
                       , n = 100
                       , m = n
                       , b = 1
                       , r = div(n, 10) + 1
                       , eps = 0.1 )

  costs = cost(n, m, reach = r)

  print("generating data (cpu)... ")
  dt = @elapsed begin
    out = rand(F, n, m, b)
    inp = rand(F, n, m, b)
  end
  println("done! ($dt seconds)")

  print("compiling operation (cpu)... ")
  dt = @elapsed begin
    op! = operator(Array{F}, (n, m, b), costs, eps)
  end
  println("done! ($dt seconds)")

  op!(out, inp)
  @time op!(out, inp)

  nothing
end

function test_expdomain_conv(args...; kwargs...)
  test_expdomain(expdomain_conv, args...; kwargs...)
end

function test_expdomain_dense(args...; kwargs...)
  test_expdomain(expdomain_dense, args...; kwargs...)
end

function test_expdomain_sparse(args...; kwargs...)
  test_expdomain(expdomain_sparse, args...; kwargs...)
end
