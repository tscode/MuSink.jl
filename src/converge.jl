
"""
    converge!(workspace; <keyword arguments>)

Auxiliary method that implements epsilon scaling to approximate the
unregularized problem.

## Arguments:
  - `start_eps = nothing`. Initial value of epsilon. By default, the current `epsilon` of `workspace` is used.
  - `target_eps = nothing`. Target value of epsilon. By default, determined indirectly via `target_blur`.
  - `target_blur = nothing`. Target value of the average blur of the transport plan. Either this value or `target_eps` should be passed explicitly.
  - `scaling = 0.9`. Factor by which epsilon is reduced per scaling step.
  - `tolerance = 1e-2`. Convergence tolerance to be achieved before `epsilon` is scaled down.
  - `target_tolerance = tolerance`. Convergence tolerance for the final scaling step.
  - `max_time = 600`. Maximal time in seconds that the function is allowed to run.
  - `max_steps = 10000`. Maximal number of steps that the function is allowed to conduct.
  - `callback`. Procedure that is called after each Sinkhorn step.
  - `callback_scaling`. Procedure that is called after each (partial) convergence.
  - `verbose = false`. If true, the function documents its progress.
"""
function converge!( w :: Workspace
                  ; start_eps = nothing
                  , target_eps = nothing
                  , target_blur = nothing
                  , scaling = 0.85
                  , tolerance = 1e-2
                  , target_tolerance = tolerance
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
  if isnothing(target_eps) && isnothing(target_blur)
    log("No target_eps or target_blur provided. Setting target_eps = $start_eps")
    target_eps = MuSink.get_eps(w)
  elseif !isnothing(target_eps) && isnothing(target_blur)
    log("Setting target_eps = $target_eps")
  elseif isnothing(target_eps) && !isnothing(target_blur)
    log("Setting target_blur = $target_blur")
  else
    log("Setting target_eps = $target_eps and target_blur = $target_blur")
  end

  # Scaling loop
  # In each iteration, the workspace is meant to converge for a given epsilon.
  # If the target (either eps or blur) is not yet reached, epsilon is decreased
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
    first_step = true
    while error > tolerance
      # Conduct the Sinkhorn step and update the error value
      # The first step after each scaling should be conducted in the stable
      # mode, since the alpha vectors are not updated in the workspace when
      # epsilon changes
      if first_step
        MuSink.step!(w, 1, stepmode = :stable)
        first_step = false
      else
        MuSink.step!(w, 1)
      end
      error = abs(MuSink.step_impact_plan(w))
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

    # Check if the target resolution is reached
    target_reached = true
    if !isnothing(target_eps)
      eps = MuSink.get_eps(w)
      target_eps_reached = (eps <= target_eps)
      if target_eps_reached
        log("Targeted value target_eps has been reached")
      end
      target_reached &= target_eps_reached
    end

    if target_reached && !isnothing(target_blur)
      blur = MuSink.blur(w)
      target_blur_reached = (blur <= target_blur)
      if target_blur_reached
        log("Targeted value target_blur has been reached")
      end
      target_reached &= target_blur_reached
    end

    # Even if epsilon does not have to be decreased further, more iterations may be
    # needed to reach the specified target tolerance
    if target_reached
      if tolerance > target_tolerance
        tolerance = target_tolerance
        log("Setting final tolerance value to $target_tolerance")
      else
        return true
      end

    # The target resolution is not yet reached. Decrease epsilon and continue
    else
      blur = MuSink.blur(w)
      eps = max(target_eps, MuSink.get_eps(w) * scaling)
      MuSink.set_eps!(w, eps)
      log("Reducing workspace eps to $eps (blur $blur)")
      
    end
  end
end
