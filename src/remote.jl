
const RChannel{T} = RemoteChannel{Channel{T}} where {T <: Any}
const AChannel{T} = Union{Channel{T}, RChannel{T}} where {T <: Any}

function init_channel(:: Type{Channel}, T :: Type, sz :: Int)
  Channel{T}(sz)
end

function init_channel(:: Type{RChannel}, T :: Type, sz :: Int)
  RemoteChannel(() -> Channel{T}(sz))
end

function set!(ac :: AChannel, val)
  isready(ac) && take!(ac)
  put!(ac, val)
end


"""
Structure that enables remote MuSink computations.

Remote workspaces can be constructed like regular ones (see [`Workspace`](@ref))
and can then be initalized in a separate task / thread / worker via
[`Remote.init`](@ref).
"""
struct RemoteWorkspace{C}
  upload :: C     # Channel to upload workspaces
  queries :: C    # Channel to communicate queries
  status :: C     # Channel reference to the current status
  activity :: C   # Channel that documents current activity

  function RemoteWorkspace(C :: Type{<: AChannel} = Channel)
    upload = init_channel(C, Tuple{Workspace, Any, C}, 1)
    queries = init_channel(C, Tuple{Symbol, Function, C}, 1000)
    status = init_channel(C, Symbol, 1)
    activity = init_channel(C, Vector{Symbol}, 1)

    put!(status, :uninitialized)
    new{C}(upload, queries, status, activity)
  end
end

function RemoteWorkspace(w::Workspace)
  rw = RemoteWorkspace(RChannel)
  @async upload!(rw, w)
  rw
end

function RemoteWorkspace(args...; kwargs...)
  rw = RemoteWorkspace(RChannel)
  w = Workspace(args...; kwargs...)
  @async upload!(rw, w)
  rw
end

function Base.show(io :: IO, rw :: RemoteWorkspace)
  s = MuSink.status(rw)
  print(io, "RemoteWorkspace(:$s)")
end

function Base.show(io :: IO, mime :: MIME"text/plain", rw :: RemoteWorkspace)
  s = status(rw)
  print(io, "RemoteWorkspace(:$s)")
end

function status(rw :: RemoteWorkspace)
  fetch(rw.status)
end

function is_initialized(rw :: RemoteWorkspace)
  status(rw) != :uninitialized
end

function is_idle(rw :: RemoteWorkspace)
  status(rw) == :idle
end

function is_running(rw :: RemoteWorkspace)
  status(rw) in [:idle, :query]
end

function is_closed(rw :: RemoteWorkspace)
  status(rw) in [:closed]
end

function upload!(rw :: RemoteWorkspace{C}, w :: Workspace{S}, T = S) where {C, S}
  @assert !is_closed(rw) "Cannot upload to closed remote workspace"

  ch = init_channel(C, Bool, 1)
  put!(rw.upload, (w, T, ch))
  if take!(ch)
    nothing
  else
    throw(ArgumentError("Workspace upload not accepted"))
  end
end

function download(rw :: RemoteWorkspace, T = Array{Float64})
  query(rw, :download) do w
    convert(Workspace{T}, w)
  end
end

function close_channels!(rw :: RemoteWorkspace)
  if isopen(rw.status)
    set!(rw.status, :closed)
    close(rw.status)
  end
  close(rw.upload)
  close(rw.queries)
end

function stop(rw :: RemoteWorkspace)
  @assert is_initialized(rw) "Cannot close uninitialized remote workspace"
  close_channels!(rw)
end

function signal(:: RemoteWorkspace{C}) where {C}
  init_channel(C, Bool, 1)
end

function signal()
  init_channel(RChannel, Bool, 1)
end

"""
    init(ws::RemoteWorkspace, signal = nothing)

Initialize the remote workspace `ws`.

A channel `signal` can be passed, which receives the value `true` if
initialization is successful.
"""
function init(rw :: RemoteWorkspace{C}, signal = nothing) where {C}

  w = Channel{Workspace}(1)
  if !isready(rw.queries)
    set!(rw.status, :idle)
  end

  # tell external task that the worker is initialized
  if !isnothing(signal)
    put!(signal, true)
  end

  # fetch new workspaces
  @async while true
    ws, T, ch = try
      take!(rw.upload)
    catch
      # if rw.workspace is closed, this is a shutdown signal
      close_channels!(rw)
      break
    end

    try
      T = MuSink.atype(T)
      set!(w, convert(Workspace{T}, ws))
      put!(ch, true)
    catch
      # this will fail if T is not a supported array type
      # in this case, reject the update
      put!(ch, false)
    end
  end
  
  # work on queries
  while true
    f, ch = try
      query = take!(rw.queries)
      set!(rw.status, :query)
      query[2], query[3]
    catch 
      # if rw.queries or rw.status is closed, this is a shutdown signal
      close_channels!(rw)
      break
    end
    try
      result = f(fetch(w))
      put!(ch, result)
    catch err
      # local errors that depend on f don't stop the worker
      # set!(rw.status, :error)
      put!(ch, err)
    end
    if !isready(rw.queries)
      set!(rw.status, :idle)
    end
    yield()
  end

  println("remote workspace closed")
end

function query( f :: Function
              , rw :: RemoteWorkspace{C}
              , name :: Symbol = :unnamed ) where {C}

  @assert is_running(rw) "Remote workspace is not running"
  # @assert hasmethod(f, Tuple{Workspace}) "Invalid query function"
  ch = init_channel(C, Any, 1)
  put!(rw.queries, (name, f, ch))
  res = take!(ch)
  if res isa Exception
    throw(res)
  else
    res
  end
end

# ----- Workspace interface -------------------------------------------------- #

convert_response(x) = x
convert_response(x :: AbstractArray) = convert(Array, x)

function query_body(fname)
  if fname isa Symbol
    sym = fname
  else
    sym = fname.args[2].value
  end
  quote
    function $fname(rw :: RemoteWorkspace, args...; kwargs...)
      query(rw, $(QuoteNode(sym))) do w
        val = $fname(w, args...; kwargs...)
        convert_response(val)
      end
    end
  end |> esc
end

macro remote_method(expr)
  if expr isa Symbol
    query_body(expr)
  elseif expr.head == :block
    names = filter(expr.args) do ex
      ex isa Symbol || ex isa Expr && ex.head == :.
    end
    bodies = map(query_body, names)
    Expr(:block, bodies...)
  else
    throw(ArgumentError("Expected symbol or block of symbols"))
  end
end

@remote_method begin

  # Node arrays
  MuSink.target
  MuSink.marginal
  MuSink.potential

  # Scalar quantities
  MuSink.mass
  MuSink.cost
  MuSink.objective

  # Sinkhorn primitives
  MuSink.update_alpha!
  MuSink.update_potential!
  MuSink.update_potential_symmetric!
  MuSink.update_operators!

  MuSink.backward_pass!
  MuSink.forward_pass!
  MuSink.forward_backward_pass!
  MuSink.forward_pass_symmetric!

  # Sinkhorn steps
  MuSink.step!
  MuSink.sync!

  # Obtain parameters
  MuSink.get_eps
  MuSink.get_rho
  MuSink.get_reach
  MuSink.get_weight
  MuSink.get_domain
  MuSink.get_stepmode

  # Modify parameters
  MuSink.set_eps!
  MuSink.set_rho!
  MuSink.set_reach!
  MuSink.set_weight!
  MuSink.set_domain!
  MuSink.set_stepmode!

  # Reductions
  MuSink.Reductions.reduce
end

