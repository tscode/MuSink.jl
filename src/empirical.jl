

"""
MuSink empirical problem type that defines the fixed parts of a classical
entropically regularized optimal transport problem.

Contains two weight vectors and a cost specifier.
"""
struct EmpiricalProblem
  a :: AbstractVector{<: AbstractFloat}
  b :: AbstractVector{<: AbstractFloat}
  c :: AbstractMatrix{<: AbstractFloat}
  
  function EmpiricalProblem(a, b, c)
    @assert isapprox(sum(a), sum(b)) """
    Weight vectors a and b must have same total mass.
    """
    @assert size(c, 1) == length(a) && size(c, 2) == length(b) """
    Size of cost specifier $(size(c)) does not fit length of weight vectors.
    """
    new(a, b, c)
  end

end

function Base.show(io :: IO, p :: EmpiricalProblem)
  print(io, "EmpiricalProblem($(length(p.a))×$(length(p.b)))")
end

function Base.show(io :: IO, ::MIME"text/plain", p :: EmpiricalProblem)
  Base.show(io, p)
end

"""
    EmpiricalProblem(a, b, c)
    EmpiricalProblem(c)

Create an empirical entropically regularized optimal transport problem.

The arguments `a` and `b` are weight vectors and `c` is a cost matrix of
dimension `(length(a), length(b))`. If the weights are omitted, uniform
weights of total mass 1 are assumed.

!!! note
    Since this is vanilla empirical optimal transport, the weight vectors `a`
    and `b` must have the same total mass.
"""
function EmpiricalProblem(c :: AbstractMatrix{<: AbstractFloat})
  a = ones(Float64, size(c, 1)) ./ size(c, 1)
  b = ones(Float64, size(c, 2)) ./ size(c, 2)
  EmpiricalProblem(a, b, c)
end

function check_stepmode(stepmode)
  stepmode = stepmode |> string |> lowercase
  options = ["default", "symmetric"]
  if stepmode in options
    Symbol(stepmode)
  else
    @warn "stepmode :$stepmode is invalid. Falling back to :stable" 
    :default
  end
end

"""
MuSink empirical workspace type that implements efficient Sinkhorn iterations
to solve entropically regularized transport between two weight vectors with same
total masses.

Compared to [`MuSink.Workspace`](@ref), this workspace is not limited to 2D
grids but works with general cost matrices. On the other hand, it only works for
the special case of 2 marginals and same total masses.
"""
mutable struct EmpiricalWorkspace{T <: AbstractArray, F}
  problem :: EmpiricalProblem

  a :: T
  b :: T
  c :: T

  stepmode :: Symbol

  eps :: F
  eps_inv :: F

  # Dual solutions, i.e., potentials in the logdomain
  da :: T
  db :: T

  # Dual variables in the exp-domain (scaling variables),
  # which may be absorbed into the kernel periodically
  u :: T
  v :: T

  u_prev :: T
  v_prev :: T

  u_buf :: T
  v_buf :: T

  kernel :: T

  absorbed :: Bool
  absorptions :: Int
  steps :: Int
end

function EmpiricalWorkspace(args...; kwargs...)
  EmpiricalWorkspace(EmpiricalProblem(args...); kwargs...)
end


function EmpiricalWorkspace( p :: EmpiricalProblem
                           ; eps = 1
                           , stepmode = :default
                           , atype = :array64 )

  stepmode = check_stepmode(stepmode)

  T = MuSink.atype(atype)
  F = eltype(T)
  eps = F(eps)
  eps_inv = one(F) / eps

  a = convert(T, p.a)
  b = convert(T, p.b)
  c = convert(T, p.c)

  da = T(undef, length(a))
  db = T(undef, length(b))

  u = T(undef, length(a))
  v = T(undef, length(b))

  u_prev = T(undef, length(a))
  v_prev = T(undef, length(b))

  u_buf = T(undef, length(a))
  v_buf = T(undef, length(b))

  kernel = exp.(.- c .* eps_inv)

  da .= zero(F)
  db .= zero(F)

  u .= one(F)
  v .= one(F)

  EmpiricalWorkspace{T, eltype(T)}( p
                                  , a
                                  , b
                                  , c
                                  , stepmode
                                  , eps
                                  , eps_inv
                                  , da
                                  , db
                                  , u
                                  , v
                                  , u_prev
                                  , v_prev
                                  , u_buf
                                  , v_buf
                                  , kernel
                                  , false
                                  , 0
                                  , 0 )

end

MuSink.atype(:: EmpiricalWorkspace{T}) where {T} = T
MuSink.steps(w :: EmpiricalWorkspace) = w.steps

function Base.show(io :: IO, w :: EmpiricalWorkspace{T}) where {T}
  dims = size(w.kernel)
  print(io, "EmpiricalWorkspace{$T}($dims)")
end

function Base.show(io :: IO, ::MIME"text/plain", w :: EmpiricalWorkspace{T}) where {T}
  dims = join(size(w.kernel), "×")
  sz = "size: $dims"
  eps = "eps: $(w.eps)"
  mode = "stepmode: :$(w.stepmode)"
  steps = "steps: $(w.steps)"
  absorptions = "absorptions: $(w.absorptions)"

  println(io, "Workspace{$T}:")
  println(io, "  $sz")
  println(io, "  $eps")
  println(io, "  $log")
  println(io, "  $mode")
  println(io, "  $steps")
  print(io, "  $absorptions")
end

function Base.copy(w :: EmpiricalWorkspace{T}, S :: Type{<: AbstractArray} = T) where {T}
  ws = EmpiricalWorkspace(w.problem; w.eps, w.stepmode, atype = S)

  ws.steps = w.steps
  ws.absorptions = w.absorptions
  ws.absorbed = w.absorbed

  ws.u .= convert(S, w.u)
  ws.v .= convert(S, w.v)

  ws.u_prev .= convert(S, w.u_prev)
  ws.v_prev .= convert(S, w.v_prev)

  ws.kernel .= convert(S, w.kernel)

  ws
end

function Base.convert(:: Type{EmpiricalWorkspace{T}}, w :: EmpiricalWorkspace{S}) where {T, S}
  T == S ? w : copy(w, T)
end

function Base.convert(:: Type{EmpiricalWorkspace{T, F}}, w :: EmpiricalWorkspace{S}) where {T, F, S}
  T == S ? w : copy(w, T)
end

function absorbpotentials!(w :: EmpiricalWorkspace{T, F}) where {T, F}
  w.da .+= w.eps * log.(w.u)
  w.db .+= w.eps * log.(w.v)
  
  w.kernel .= w.u .* w.kernel .* w.v'
  w.u .= one(F)
  w.v .= one(F)

  w.absorptions += 1
  w.absorbed = true

  nothing
end

function step!(w :: EmpiricalWorkspace{T}; stepmode = w.stepmode) where {T}

  w.u_prev .= w.u
  w.v_prev .= w.v
  
  if stepmode == :symmetric

    mul!(w.u_buf, w.kernel, w.v)
    mul!(w.v_buf, w.kernel', w.u)

    w.u .= sqrt.(w.u) .* sqrt.(w.a ./ w.u_buf)
    w.v .= sqrt.(w.v) .* sqrt.(w.b ./ w.v_buf)

  else

    mul!(w.u, w.kernel, w.v)
    w.u .= w.a ./ w.u

    mul!(w.v, w.kernel', w.u)
    w.v .= w.b ./ w.v
  end

  w.steps += 1
  w.absorbed = false

  nothing
end

MuSink.get_eps(w :: EmpiricalWorkspace) = w.eps
MuSink.get_stepmode(w :: EmpiricalWorkspace) = w.stepmode

function MuSink.set_eps!(w :: EmpiricalWorkspace{T}, eps) where {T}
  w.eps = eps
  w.eps_inv = 1 / eps

  # This updates da and db and brings u and v back to 1, which makes them
  # independent of eps. Therefore, they do not have to be updated here.
  absorbpotentials!(w)

  w.kernel .= exp.(.- (w.c .- w.da .- w.db') .* w.eps_inv)

  w.absorbed = false

  nothing
end

function MuSink.set_stepmode!(w :: EmpiricalWorkspace, stepmode)
  w.stepmode = check_stepmode(stepmode)
  nothing
end

function MuSink.marginal(w :: EmpiricalWorkspace, index :: Int)
  @assert index in [1, 2] """
  Empirical workspaces only support two marginals with index 1 and 2.\
  Given: $index.
  """
  if !w.absorbed
    absorbpotentials!(w)
  end
  dims = index == 1 ? 2 : 1
  reshape(sum(w.kernel, dims = dims), :)
end

function MuSink.dense(w :: EmpiricalWorkspace{T}) where {T}
  if !w.absorbed
    absorbpotentials!(w)
  end
  w.kernel
end

function step_impact_potential(w :: EmpiricalWorkspace)
  w.u_buf .= (log.(w.u) .- log.(w.u_prev))
  w.v_buf .= (log.(w.v) .- log.(w.v_prev))
  w.eps * max(
    maximum(absfinite, w.u_buf),
    maximum(absfinite, w.v_buf),
  )
end

function step_impact_plan(w :: EmpiricalWorkspace)
  exp(2 * step_impact_potential(w) * w.eps_inv) - 1
end

function check_consistency(w :: EmpiricalWorkspace)
  # Update da and db
  if !w.absorbed
    absorbpotentials!(w)
  end
  plan = exp.(.- (w.c .- w.da .- w.db') .* w.eps_inv)
  error = maximum(abs, plan .- MuSink.dense(w))
  if error > 1e-8
    @warn "Accumulated transport plan error of up to $error detected"
    false
  else
    true
  end
end

function MuSink.converge!( w :: EmpiricalWorkspace
                         ; start_eps = nothing
                         , target_eps = nothing
                         , scaling = 0.85
                         , tolerance = 1e-2
                         , target_tolerance = tolerance
                         , absorption_threshold = 1e5
                         , max_time = 600
                         , max_steps = 10000
                         , callback = () -> nothing
                         , callback_scaling = () -> nothing
                         , verbose = false )

  log(msg) = verbose && println(msg)

  count = 0
  start_time = time()
  error = Inf

  # Set start value for epsilon
  if !isnothing(start_eps)
    log("Setting workspace eps = $start_eps")
    MuSink.set_eps!(w, start_eps)
  else
    log("No start_eps provided, using workspace value eps = $(get_eps(w))")
    start_eps = get_eps(w)
  end

  # Set target value for epsilon to the current value if no target is specified
  if isnothing(target_eps)
    log("No target_eps provided. Setting target_eps = $start_eps")
    target_eps = MuSink.get_eps(w)
  else
    log("Setting target_eps = $target_eps")
  end

  # Scaling loop
  # In each iteration, the workspace is meant to converge for a given epsilon.
  # If the target eps is not yet reached, epsilon is decreased
  # and the next iteration commences
  count_scaling = 0
  while true

    count_scaling += 1
    log("Running scaling iteration $count_scaling")
    callback_scaling()
    error = Inf

    steps_before = MuSink.steps(w)

    # Step loop
    # Iterate Sinkhorn steps until the error is below tolerance
    while error > tolerance
      # Conduct the Sinkhorn step and update the error value
      MuSink.step!(w, 1)
      error = abs(step_impact_plan(w))

      magnitude = max(
        maximum(w.u),
        maximum(abs, w.v),
      )
      if magnitude > absorption_threshold
        log("Absorption of potentials into kernel during step $(w.steps)")
        absorbpotentials!(w)
      end

      count += 1
      callback()

      # Set stop = true if time or step count is too large
      time_limit = time() - start_time > max_time
      step_limit = count > max_steps

      if time_limit
        log("Interrupting scaling iteration $count_scaling due to time limit ($max_time seconds)")
        return false
      elseif step_limit
        log("Interrupting scaling iteration $count_scaling due to step limit ($max_steps steps)")
        return false
      end
    end

    steps_scaling = MuSink.steps(w) - steps_before
    log("Finished scaling iteration $count_scaling after $steps_scaling steps")

    # Check if the target is reached

    eps = MuSink.get_eps(w)
    if eps <= target_eps
      log("Targeted value target_eps has been reached")
      if tolerance > target_tolerance
        tolerance = target_tolerance
        log("Setting final tolerance value to $target_tolerance")
      else
        return true
      end
    # Target eps is not yet reached. Decrease epsilon and continue
    else
      eps = max(target_eps, MuSink.get_eps(w) * scaling)
      MuSink.set_eps!(w, eps)
      log("Reducing workspace eps to $eps")
    end
  end
end
