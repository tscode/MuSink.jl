
# ----- General interface and defaults --------------------------------------- #

"""
Ground cost objects for the unbalanced optimal transport problem.

Only costs that are functions of the differences `(ia - ja, ib - jb)`
of pixel coordinates `(ia, ja)` and `(ib, jb)` are supported.
"""
abstract type Cost end

Base.show(io, c :: Cost) = print(io, string(c))
Base.show(io, :: MIME"text/plain", c :: Cost) =  show(io, c)


"""
Costs with a separable kernel. This means that the total cost is the sum of two
marginal costs that depend on the row respectively column of a 2-d grid point
only.
"""
abstract type SeparableCost <: Cost end

max_reach(cost) = max(size(cost)...) - 1

axes(c :: Cost, args...) = axes(Float64, c, args...)
axes(:: DataType, c :: Cost, r = max_reach(c)) = axes(Float64, c, r, r)

window(c :: Cost, args...) = window(Float64, c, args...)
window(F :: DataType, c :: Cost, r = max_reach(c)) = window(F, c, r, r)

matrix(c :: Cost, args...) = matrix(Float64, c, args...)
matrix(F :: DataType, c :: Cost, r = max_reach(c)) = matrix(F, c, r, r)

matrix_axes(c :: Cost, args...) = matrix_axes(Float64, c, args...)
matrix_axes(F :: DataType, c :: Cost, r = max_reach(c)) = matrix_axes(F, c, r, r)

matrixCSC(c :: Cost, args...) = matrixCSC(Float64, c, args...)
matrixCSC(F :: DataType, c :: Cost, r = max_reach(c)) = matrixCSC(F, c, r, r)

matrixCSC_axes(c :: Cost, args...) = matrixCSC_axes(Float64, c, args...)
matrixCSC_axes(F :: DataType, c :: Cost, r = max_reach(c)) = matrixCSC_axes(F, c, r, r)

function window(F :: DataType, cost :: SeparableCost, ri, rj)
  ci, cj = axes(F, cost, ri, rj)
  ci .+ cj'
end

function matrix_axes(F :: DataType, cost :: SeparableCost, ri, rj)

  n, m = size(cost)
  ci, cj = axes(F, cost, ri, rj)
  mi = fill(typemax(F), n, n)
  mj = fill(typemax(F), m, m)

  for ia in 1:n, ir in -ri:ri
    ib = ia + ir
    if 1 <= ib <= n
      @inbounds mi[ia, ib] = ci[ia - ib + ri + 1]
    end
  end

  for ja in 1:m, jr in -rj:rj
    jb = ja + jr
    if 1 <= jb <= m
      @inbounds mj[ja, jb] = cj[ja - jb + rj + 1]
    end
  end

  mi, mj
end

function matrix(F :: DataType, cost :: Cost, ri, rj)
  n, m = size(cost)
  cc = window(F, cost, ri, rj)
  mat = fill(typemax(F), n, m, n, m)
  for ia in 1:n, ja in 1:m
    for jr in -rj:rj
      jb = ja + jr
      if 1 <= jb <= m
        for ir in -ri:ri
          ib = ia + ir
          if 1 <= ib <= n
            c = cc[ia - ib + ri + 1, ja - jb + rj + 1]
            @inbounds mat[ia, ja, ib, jb] = c
          end
        end
      end
    end
  end
  reshape(mat, n * m, n * m)
end

# TODO: Timings here can be optimized, at least by threading
function matrixCSC(F :: DataType, cost :: Cost, ri, rj)
  n, m = size(cost)
  cc = window(F, cost, ri, rj)

  A = Int64[]
  B = Int64[]
  V = F[]

  sizehint!(A, n * m * (2ri+1)*(2rj+1))
  sizehint!(B, n * m * (2ri+1)*(2rj+1))
  sizehint!(V, n * m * (2ri+1)*(2rj+1))

  for ia in 1:n
    for ja in 1:m
      for jr in -rj:rj
        jb = ja + jr
        if 1 <= jb <= m
          @inbounds for ir in -ri:ri
            ib = ia + ir
            if 1 <= ib <= n
              push!(A, ia + n * (ja-1))
              push!(B, ib + n * (jb-1))
              val = cc[ia - ib + ri + 1, ja - jb + rj + 1]
              push!(V, val)
            end
          end
        end
      end
    end
  end

  sparse(A, B, V, n*m, n*m)
end

function matrixCSC_axes(F :: DataType, cost :: SeparableCost, ri, rj)
  n, m = size(cost)
  ci, cj = axes(F, cost, ri, rj)

  A = Int64[]
  B = Int64[]
  V = F[]

  sizehint!(A, n * (2ri+1))
  sizehint!(B, n * (2ri+1))
  sizehint!(V, n * (2ri+1))

  for ia in 1:n, ir in -ri:ri
    ib = ia + ir
    if 1 <= ib <= n
      push!(A, ia)
      push!(B, ib)
      @inbounds val = ci[ia - ib + ri + 1]
      push!(V, val)
    end
  end

  mi = sparse(A, B, V, n, n)

  empty!(A)
  empty!(B)
  empty!(V)

  sizehint!(A, n * (2rj+1))
  sizehint!(B, n * (2rj+1))
  sizehint!(V, n * (2rj+1))

  for ja in 1:m, jr in -rj:rj
    jb = ja + jr
    if 1 <= jb <= m
      push!(A, ja)
      push!(B, jb)
      @inbounds val = cj[ja - jb + rj + 1]
      push!(V, val)
    end
  end

  mj = sparse(A, B, V, m, m)

  mi, mj
end

# ----- Convenient Scaling --------------------------------------------------- #

(Base.:*)(a :: Real, cost :: Cost) = scale(cost, Float64(a))

# ----- Implementations ------------------------------------------------------ #

"""
Generic costs defined by a fixed cost window.
"""
struct GenericCost <: Cost
  window :: Matrix{Float64}

  function GenericCost(window)
    li, lj = size(window)
    @assert li >= 1 && lj >= 1
    @assert isodd(li) && isodd(lj)
    new(window)
  end
end

function GenericCost(c :: Cost)
  n, m = size(c)
  GenericCost(window(c, n-1, m-1))
end

Base.size(cost :: GenericCost) = div.(size(cost.window), 2) .+ 1

function scale(cost :: GenericCost, a :: Float64)
  GenericCost(a .* cost.window)
end

function window(F :: DataType, cost :: GenericCost, ri, rj)

  rmax = max_reach(cost)
  @assert ri <= rmax "reach for this cost must not be larger than $rmax"
  @assert rj <= rmax "reach for this cost must not be larger than $rmax"

  n, m = size(cost)

  di = min(n-1, ri)
  dj = min(m-1, rj)

  cc = fill(typemax(F), 2ri + 1, 2rj + 1)

  for ir in -di:di, jr in -dj:dj
    c = cost.window[ir + n, jr + m]
    cc[ir + ri + 1, jr + rj + 1] = c
  end

  cc
end

function Base.show(io :: IO, cost :: GenericCost)
  sz = size(cost)
  println(io, "GenericCost$sz")
end

function Base.show(io :: IO, ::MIME"text/plain", cost :: GenericCost)
  sz = size(cost)
  println(io, "GenericCost$sz")
end


"""
Generic separable cost defined by fixed cost axes.
"""
struct GenericSeparableCost <: SeparableCost
  axis_i :: Vector{Float64}
  axis_j :: Vector{Float64}

  function GenericSeparableCost(axis_i, axis_j)
    li, lj = length(axis_i), length(axis_j)
    @assert li >= 1 && lj >= 1
    @assert isodd(li) && isodd(lj)
    new(axis_i, axis_j)
  end
end

function GenericSeparableCost(c :: SeparableCost)
  n, m = size(c)
  axis_i, axis_j = axes(c, n-1, m-1)
  GenericSeparableCost(axis_i, axis_j)
end

function Base.size(cost :: GenericSeparableCost)
  si = div(length(cost.axis_i), 2) + 1
  sj = div(length(cost.axis_j), 2) + 1
  (si, sj)
end

function scale(cost :: GenericSeparableCost, a :: Float64)
  GenericSeparableCost(a .* cost.axis_i, a .* cost.axis_j)
end

function axes(F :: DataType, cost :: GenericSeparableCost, ri, rj)

  rmax = max_reach(cost)
  @assert ri <= rmax "reach for this cost must not be larger than $rmax"
  @assert rj <= rmax "reach for this cost must not be larger than $rmax"

  n, m = size(cost)

  di = min(n-1, ri)
  dj = min(m-1, rj)

  ci = fill(typemax(F), 2ri + 1)
  cj = fill(typemax(F), 2rj + 1)
  
  for ir in -di:di
    ci[ir + ri + 1] = cost.axis_i[ir + n]
  end

  for jr in -dj:dj
    cj[jr + rj + 1] = cost.axis_j[jr + m]
  end

  ci, cj
end

function Base.show(io :: IO, cost :: GenericSeparableCost)
  sz = size(cost)
  println(io, "GenericSeparableCost$sz")
end

function Base.show(io :: IO, ::MIME"text/plain", cost :: GenericSeparableCost)
  sz = size(cost)
  println(io, "GenericSeparableCost$sz")
end


"""
Lp costs.
"""
struct Lp <: SeparableCost
  n :: Int
  m :: Int

  power :: Float64
  scale :: Float64
  max :: Float64
end

function Lp(n :: Integer , m :: Integer ; p = 2 , max = 1.)
  scale = max / ((n-1)^p + (m-1)^p)
  Lp(n, m, p, scale, max)
end

Base.size(cost :: Lp) = (cost.n, cost.m)

function scale(cost :: Lp, a :: Float64)
  Lp(cost.n, cost.m; p = cost.power, max = a * cost.max)
end

function axes(F :: DataType, cost :: Lp, ri, rj)

  rmax = max_reach(cost)
  @assert ri <= rmax "reach for this cost must not be larger than $rmax"
  @assert rj <= rmax "reach for this cost must not be larger than $rmax"

  ci = zeros(F, 2ri + 1)
  for ir in -ri:ri
    ci[ir + ri + 1] =  abs(ir)^cost.power
  end
  ci .= cost.scale .* ci

  cj = zeros(F, 2rj + 1)
  for jr in -rj:rj
    cj[jr + rj + 1] =  abs(jr)^cost.power
  end
  cj .= cost.scale .* cj

  ci, cj
end

function Base.string(c :: Lp)
  "Lp($(c.power), max = $(c.max))"
end

function Base.show(io :: IO, c :: Lp)
  println(io, string(c))
end

function Base.show(io :: IO, ::MIME"text/plain", c :: Lp)
  println(io, string(c))
end

"""
Lp costs that are implemented as if they were not separable. For debugging
purposes only.
"""
struct LpNotSep <: Cost
  n :: Int
  m :: Int
  power :: Float64
  scale :: Float64
  max :: Float64
end

function LpNotSep(n :: Integer , m :: Integer ; p = 2 , max = 1.)
  scale = max / ((n-1)^p + (m-1)^p)
  LpNotSep(n, m, p, scale, max)
end

Base.size(cost :: LpNotSep) = (cost.n, cost.m)

function scale(cost :: LpNotSep, a :: Float64)
  LpNotSep(cost.n, cost.m; p = cost.power, max = a * cost.max)
end

function window(F :: DataType, cost :: LpNotSep, ri, rj)

  rmax = max_reach(cost)
  @assert ri <= rmax "reach for this cost must not be larger than $rmax"
  @assert rj <= rmax "reach for this cost must not be larger than $rmax"

  cc = zeros(F, 2ri+1, 2rj+1)
  for ir in -ri:ri, jr in -rj:rj
    vi = abs(ir)^cost.power
    vj = abs(jr)^cost.power
    cc[ir + ri + 1, jr + rj + 1] = vi + vj
  end
  cc .= cost.scale .* cc
  cc
end

function Base.string(c :: LpNotSep)
  "LpNotSep($(c.power), max = $(c.max))"
end

function Base.show(io :: IO, c :: LpNotSep)
  println(io, string(c))
end

function Base.show(io :: IO, ::MIME"text/plain", c :: LpNotSep)
  println(io, string(c))
end
