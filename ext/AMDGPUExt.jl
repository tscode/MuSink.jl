
module AMDGPUExt

using LinearAlgebra, SparseArrays

using AMDGPU

import MuSink
import MuSink: Workspace, Cost, SeparableCost
import MuSink: axes, window, matrix, matrix_axes, matrixCSC_axes
import MuSink.Ops
import MuSink.Remote

# Tell MuSink about new array types
function __init__()
  MuSink.register_atype!(:amd32, ROCArray{Float32})
  MuSink.register_atype!(:amd64, ROCArray{Float64})
end

# Specialize some MuSink functions
MuSink.drop_batchdim(x :: ROCVector) = AMDGPU.@allowscalar x[1]
MuSink.sync_arrays(:: Type{<: ROCArray}) = AMDGPU.synchronize()

end # AMDGPUExt

