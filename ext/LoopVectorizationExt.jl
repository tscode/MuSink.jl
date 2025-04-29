
module LoopVectorizationExt

__precompile__(false)

import MuSink
import MuSink.Ops

import LoopVectorization

"""
CPU kernel function implementing the inner sinkhorn loop
"""
function Ops.cpukernel_logdomain(ia, ja, k, db, cc, eps)

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
  LoopVectorization.@turbo for jb in jbmin:jbmax, ib in ibmin:ibmax
    c = cc[ia - ib + r + 1, ja - jb + r + 1]
    val = (db[ib, jb, k] - c) * eps_inv
    mx = max(mx, val)
  end

  # exponential sum
  s = zero(eps)
  if !isinf(mx)
    LoopVectorization.@turbo for jb in jbmin:jbmax, ib in ibmin:ibmax
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
function Ops.cpukernel_logdomain_sep!(out, k, db, ci, eps)
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
      LoopVectorization.@turbo for ib in ibmin:ibmax
        c = ci[ib - ia + r + 1]
        val = (db[ib, jb, k] - c) * eps_inv
        mx = max(mx, val)
      end

      s = zero(eps)
      if !isinf(mx)
        LoopVectorization.@turbo for ib in ibmin:ibmax
          c = ci[ib - ia + r + 1]
          val = (db[ib, jb, k] - c) * eps_inv
          s += exp(val - mx)
        end
      end

      @inbounds out[jb, ia, k] = eps * (log(s) + mx)
    end
  end
end

function Ops.cpukernel_logdomain_sept!(out, k, dat, cj, eps)
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
      LoopVectorization.@turbo for jb in jbmin:jbmax
        c = cj[jb - ja + r + 1]
        val = (dat[jb, ia, k] - c) * eps_inv
        mx = max(mx, val)
      end

      s = zero(eps)
      if !isinf(mx)
        LoopVectorization.@turbo for jb in jbmin:jbmax
          c = cj[jb - ja + r + 1]
          val = (dat[jb, ia, k] - c) * eps_inv
          s += exp(val - mx)
        end
      end

      @inbounds out[ia, ja, k] = eps * (log(s) + mx)
    end
  end
end

end
