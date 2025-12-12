
"""
MuSink empirical problem type that defines the fixed parts of a classical
entropically regularized optimal transport problem.

Contains two weight vectors and a cost specifier.
"""
struct EmpiricalProblem
  a :: Vector{Float64}
  b :: Vector{Float64}
  c :: Matrix{Float64}
  # x :: Vector   # TODO
  # y :: Vector
  
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

Base.show(io :: IO, ::MIME"text/plain", p :: EmpiricalProblem) = Base.show(io, p)

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
  logdomain :: Bool

  eps :: F
  eps_inv :: F

  potential_a :: T
  potential_b :: T

  potential_a_prev :: T
  potential_b_prev :: T

  potential_a_buf :: T
  potential_b_buf :: T

  kernel :: T

  absorbed :: Bool
  absorptions :: Int
  steps :: Int
end

function EmpiricalWorkspace( p :: EmpiricalProblem
                           ; eps = 1
                           , stepmode = :default
                           , logdomain = false
                           , atype = :array64 )

  @assert logdomain == false
  @assert stepmode in [:default, :symmetric] """
  Only stepmodes :default and :symmetric are supported.
  """

  T = MuSink.atype(atype)
  F = eltype(T)
  eps = F(eps)

  a = convert(T, p.a)
  b = convert(T, p.b)
  c = convert(T, p.c)

  potential_a = T(undef, length(a))
  potential_b = T(undef, length(b))

  potential_a_prev = T(undef, length(a))
  potential_b_prev = T(undef, length(b))

  potential_a_buf = T(undef, length(a))
  potential_b_buf = T(undef, length(b))


  if logdomain
    # TODO: logdomain does not work
    kernel = copy(c)
    potential_a .= zero(F)
    potential_b .= zero(F)
  else
    kernel = exp.(.- c ./ eps)
    potential_a .= one(F)
    potential_b .= one(F)
  end

  potential_a_prev .= potential_a
  potential_b_prev .= potential_b

  EmpiricalWorkspace{T, eltype(T)}( p
                                  , a
                                  , b
                                  , c
                                  , stepmode
                                  , logdomain
                                  , eps
                                  , one(F) / eps
                                  , potential_a
                                  , potential_b
                                  , potential_a_prev
                                  , potential_b_prev
                                  , potential_a_buf
                                  , potential_b_buf
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
  log = "logdomain: $(w.logdomain)"
  mode = "stepmode: :$(w.stepmode)"

  println(io, "Workspace{$T}:")
  println(io, "  $sz")
  println(io, "  $eps")
  println(io, "  $log")
  print(io, "  $mode")
end

function Base.copy(w :: EmpiricalWorkspace{T}, S :: Type{<: AbstractArray} = T) where {T}
  ws = Workspace(w.problem; w.eps, w.logdomain, w.stepmode, atype = S)
  ws.steps = w.steps
  ws.absorptions = w.absorptions
  ws.absorbed = w.absorbed

  ws.kernel .= convert(S, w.buffer)
  ws.potential_a .= convert(S, w.potential_a)
  ws.potential_b .= convert(S, w.potential_b)

  ws.potential_a_prev .= convert(S, w.potential_a_prev)
  ws.potential_b_prev .= convert(S, w.potential_b_prev)

  ws.potential_a_buf .= convert(S, w.potential_a_buf)
  ws.potential_b_buf .= convert(S, w.potential_b_buf)

  ws
end

function Base.convert(:: Type{EmpiricalWorkspace{T}}, w :: EmpiricalWorkspace{S}) where {T, S}
  T == S ? w : copy(w, T)
end

function Base.convert(:: Type{EmpiricalWorkspace{T, F}}, w :: EmpiricalWorkspace{S}) where {T, F, S}
  T == S ? w : copy(w, T)
end

function absorbpotentials!(w :: EmpiricalWorkspace{T, F}) where {T, F}
  if w.logdomain
    w.kernel .= w.c .- w.potential_a .- w.potential_b'
    w.potential_a .= zero(F)
    w.potential_b .= zero(F)
  else
    w.kernel .= w.potential_a .* w.kernel .* w.potential_b'
    w.potential_a .= one(F)
    w.potential_b .= one(F)
  end

  w.absorptions += 1
  w.absorbed = true

  nothing
end

function step!(w :: EmpiricalWorkspace{T}; stepmode = w.stepmode) where {T}

  w.potential_a_prev .= w.potential_a
  w.potential_b_prev .= w.potential_b
  
  if stepmode == :symmetric

    mul!(w.potential_a_buf, w.kernel, w.potential_b)
    mul!(w.potential_b_buf, w.kernel', w.potential_a)

    w.potential_a .= sqrt.(w.potential_a) .* sqrt.(w.a ./ w.potential_a_buf)
    w.potential_b .= sqrt.(w.potential_b) .* sqrt.(w.b ./ w.potential_b_buf)

  else

    mul!(w.potential_a, w.kernel, w.potential_b)
    w.potential_a .= w.a ./ w.potential_a

    mul!(w.potential_b, w.kernel', w.potential_a)
    w.potential_b .= w.b ./ w.potential_b
  end

  w.steps += 1
  w.absorbed = false

  nothing
end

MuSink.get_eps(w :: EmpiricalWorkspace) = w.eps
MuSink.get_domain(w :: EmpiricalWorkspace) = w.logdomain
MuSink.get_stepmode(w :: EmpiricalWorkspace) = w.stepmode

function MuSink.set_eps!(w :: EmpiricalWorkspace{T}, eps) where {T}
  eps_old = w.eps
  eps_inv_old = 1 / eps_old

  w.eps = eps
  w.eps_inv = 1 / eps

  # Potentials only change if stored in expdomain
  if !w.logdomain
    # Go to logdomain via old eps and back to expdomain via new eps
    w.potential_a .= eps_old .* log.(w.potential_a)
    w.potential_a .= exp.(w.potential_a .* w.eps_inv)
    w.potential_b .= eps_old .* log.(w.potential_b)
    w.potential_b .= exp.(w.potential_b .* w.eps_inv)

    w.kernel .= eps_old .* log.(w.kernel) 
    w.kernel .= exp.(w.kernel .* w.eps_inv)
  end

  w.absorbed = false

  nothing
end

function MuSink.set_domain!(w :: EmpiricalWorkspace, logdomain)
  error("Not implemented")
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
  if w.logdomain
    max(
      maximum(absfinite, w.potential_a .- w.potential_a_prev),
      maximum(absfinite, w.potential_b .- w.potential_b_prev),
    )
  else
    w.potential_a_buf .= (log.(w.potential_a) .- log.(w.potential_a_prev))
    w.potential_b_buf .= (log.(w.potential_b) .- log.(w.potential_b_prev))
    w.eps * max(
      maximum(absfinite, w.potential_a_buf),
      maximum(absfinite, w.potential_b_buf),
    )
  end
end

function step_impact_plan(w :: EmpiricalWorkspace)
  exp(2 * step_impact_potential(w) * w.eps_inv) - 1
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

      if !w.logdomain
        magnitude = max(
          maximum(w.potential_a),
          maximum(abs, w.potential_b),
        )
        if magnitude > absorption_threshold
          log("Absorption of potentials into kernel during step $(w.steps)")
          absorbpotentials!(w)
        end
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
