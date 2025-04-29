
"""
Marginal penalties that implement relaxed constraints in the
unbalanced optimal transport problem.
"""
abstract type Penalty{P} end

struct TotalVariation <: Penalty{Float64} end

function (:: TotalVariation)(a, b, rho)
  vals = rho * sum(abs, a .- b, dims = (1,2))
  dropdims(vals, dims = (1,2))
end

function approx!(x, :: TotalVariation, rho, eps = 1.0)
  clamp!(x, -rho, rho)
end

function marginal_mass_bias(:: TotalVariation, rhos, masses)
  n = length(rhos)
  @assert n == length(masses)
  if any(isinf, rhos)
    ms = [masses[i] for i in 1:n if isinf(rhos[i])]
    median(ms)
  else
    # weighted median
    rho_total = sum(rhos)
    perm = sortperm(masses)
    rho_sorted = rhos[perm]
    masses_sorted = masses[perm]
    idx = findfirst(crho -> crho >= 0.5 * rho_total, cumsum(rho_sorted))
    # TODO: this should probably interpolate
    masses_sorted[idx]
  end
end

struct KullbackLeibler <: Penalty{Float64} end

function kl(a, b)
  if b == 0
    0
  else
    log.(a ./ b) .* b
  end
end

function (:: KullbackLeibler)(a, b, rho)
  vals = rho * sum(kl, a, b)
  dropdims(vals, dims = (1,2))
end

function approx!(x, :: KullbackLeibler, rho, eps)
  if rho == 0
    x .= 0
  elseif rho != Inf
    x .*= rho / (rho + eps)
  end
  # Do nothing if rho == Inf
end

function marginal_mass_bias(:: KullbackLeibler, rhos, masses)
  n = length(rhos)
  @assert n == length(masses)
  if any(isinf, rhos)
    ms = [masses[i] for i in 1:n if isinf(rhos[i])]
    exp(sum(log, ms) / length(ms)) # geometric mean
  else
    exponent = sum(rhos[i] * log(masses[i]) for i in 1:n if rhos[i] > 0)
    exp(exponent/sum(rhos))
  end
end
