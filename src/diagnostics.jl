
# ------ Diagnostics --------------------------------------------------------- #
# TODO: documentation!

# TODO
function report_wellposed(w :: Workspace)
  # if the masses of the targets are different but rho is set high, convergence
  # may be very slow

  # if eps is large compared to rho and the costs, the product of the reference
  # measures becomes a fixpoint for the optimal coupling. This can *greatly*
  # influence the mass of the solution
end

# TODO
struct WorkspaceStatus
  steps :: Int

  # Convergence
  objective_change :: Float64
  potential_change :: Float64
  plan_change :: Float64
  tmap_change :: Float64
  mass_instability :: Float64

  # Mass
  mass :: Float64
  target_mass :: Float64
  reference_mass :: Float64

  # Sharpness
  softness :: Float64
  std :: Float64
end

# TODO
function report_health(w :: Workspace; peek_steps = 1)
  if w.steps <= 1
    println("Not enough steps ($(w.steps)) have been performed for a health check")
  end

  println("Convergence:")
  change_potential = step_impact_potential(w)
  println("  maximal change in the potentials: $change_potential")

  change_plan = step_impact_plan(w)
  println("  change in the coupling (pessimistic heuristic): $change_plan")

  change_tmap = step_impact_map(w)
  println("  mean change of the projected transport map: $change_tmap")

  println("Mass:")

  println("Blur:")

  # CONVERGENCE
    # provide changes of potential + plan
    # provide changes of transport map per iteration
    # provide mass instability

  # MASS
end


"""
    status(w)

Print a status string that includes several indicators of health and
convergence.
"""
function status(w)

end


function steps(w :: Workspace)
  w.steps
end

function step_impact_potential(w :: Workspace)
  if w.logdomain
    maximum(keys(w.potentials)) do node
      # target = exp.(w.targets[node] .* w.eps_inv)
      # weights = target / sum(target)
      maximum(absfinite, w.potentials[node] .- w.potentials_prev[node])
      # sum(absfinite, diff)
    end
  else
    maximum(keys(w.potentials)) do node
      # weights = w.targets[node] / sum(w.targets[node])
      potential = w.eps .* log.(w.potentials[node])
      potential_prev = w.eps .* log.(w.potentials_prev[node])
      maximum(absfinite, potential .- potential_prev)
      # sum(absfinite, diff)
    end
  end
end

function step_impact_plan(w :: Workspace)
  exp(2 * step_impact_potential(w) * w.eps_inv) - 1
end

# TODO
function step_impact_objective(w :: Workspace)
  wc = copy(w)
  MuSink.step!(wc)
  MuSink.objective(wc) - MuSink.objective(w)
end

function step_impact_map(w :: Workspace; peek_steps = 1)
  wc = copy(w)
  MuSink.step!(wc, peek_steps)
  maximum(keys(w.targets)) do node
    neighbors = Tree.neighbors(node)
    maximum(neighbors) do neighbor
      ivals = Reductions.imap(w, node, neighbor)
      ivalsc = Reductions.imap(wc, node, neighbor)
      jvals = Reductions.jmap(w, node, neighbor)
      jvalsc = Reductions.jmap(wc, node, neighbor)
      diffsq = (ivals .- ivalsc).^2 .+ (jvals .- jvalsc).^2
      maximum(sqrt, diffsq)
    end
  end
end

function mean_marginal_deviation(w :: Workspace{T}) where {T}
  val = mean(keys(w.potentials)) do node
    target = convert(T, w.problem.targets[node])
    marg = marginal(w, node)
    sum(abs, target .- marg)
  end
  val / mass(w)
end



# ---- Sharpness metrics ----------------------------------------------------- #

# TODO: This currently takes averages over batches...
function count_heavy_points(plan :: Coupling, i :: Int, j :: Int; threshold = 0.5)
  @assert 0 <= threshold <= 1

  b = size(plan.buffer, 3)
  masses = transport(plan, i, j, conditional = true) 
  masses = reshape(masses, :)
  masses = convert(Array, masses)
  sort!(masses, rev = true)

  index = 1
  mass = 0
  while mass <= threshold * b
    mass += masses[index]
    index += 1
  end

  index / b
end

# TODO: This is a preliminary implementation that should be improved
# Idea: sample from a transport plan and determine the sharpness
# of the corresponding transport
function estimate_heavy_points(w :: Workspace; samples = 10, threshold = 0.5)
  n, m, _ = size(w.buffer)

  a = w.root
  b = w.root.children[1]
  plan = Coupling(w, a, b)

  C = CartesianIndices((n,m))
  data = reshape(plan.marginal_a[:,:,1:1], :)
  perm = sortperm(data, rev = true)

  idx = perm[1:min(samples, end)]
  mean(C[idx]) do cidx
    i, j = Tuple(cidx)
    count_heavy_points(plan, i, j; threshold)
  end
end
