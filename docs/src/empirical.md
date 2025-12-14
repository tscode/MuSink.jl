# Empirical Entropic Optimal Transport

For testing purposes and for convenience, MuSink also implements a robust and performant version of the classical Sinkhorn scaling for entropic optimal transport. It is restricted to the special case of **balanced** optimal transport between **two measures** only. The cost structure can be arbitrary, however.

This functionality is implemented via [`EmpiricalWorkspace`](@ref).
```julia
n, m = (10, 20)

a, b = rand(n), rand(m)
a ./= sum(a)
b ./= sum(b)

# Random cost matrix
c = rand(n, m)

# Create the empirical workspace (limited to the special sub-problem of
# two balanced measures and full transport)
w = EmpiricalWorkspace(a, b, c, eps = 1)
converge!(w, target_eps = 0.01)

marginal(w, 1) # should be similar to a
marginal(w, 2) # should be similar to b

dense(w) # Get the full plan
```

Often, the cost matrix will be derived from two point clouds to be matched. Below is an example where we calculate entropic optimal transport between two images where only a few pixels are active.

We now have two options to do so. First, we can operate on the full dense images. Alternatively, we construct a suitable cost matrix out of the locations of active pixels and employ the [`EmpiricalWorkspace`](@ref).

```julia
pixels_1 = [
  (1, 5),
  (10, 100),
  (55, 3)
]

pixels_2 = [
  (72, 42),
  (37, 66),
]

img_1 = zeros(100, 100)
img_2 = zeros(100, 100)

for (i, j) in pixels_1
  img_1[i, j] = 1
end

for (i, j) in pixels_2
  img_2[i, j] = 1
end

img_1 .= img_1 ./ sum(img_1)
img_2 .= img_2 ./ sum(img_2)


##
## Option 1: Solve the problem between the 2d-images
##

# This is convenience syntax for a workspace with a simple linear graph
cost = Lp(100, 100; p = 2, max = 1)
w1 = Chain([img_1, img_2], cost = cost, eps = 0.1)

converge!(w1, target_eps = 0.001, scaling = 0.5)

# Coloc mass a cost threshold 0.1
@show sum(Reductions.coloc(w1, 1, 2, threshold = 0.1))


##
## Option 2: Solve the problem by classical EOT with a given cost matrix
##

scale = 99^2 + 99^2 # scale maximal cost to 1

c = broadcast(pixels_1, reshape(pixels_2, 1, :)) do (i1, j1), (i2, j2)
  ((i1 - i2)^2 + (j1 - j2)^2) / scale 
end

w2 = EmpiricalWorkspace(c, eps = 0.1)

converge!(w2, target_eps = 0.001, scaling = 0.5)

# Coloc mass a cost threshold 0.1. Here, we can compute the full plan
plan = dense(w2)

# Coloc mass a cost threshold 0.1
@show sum(plan[c .<= 0.1])
```

As in the case of a usual [`Workspace`](@ref), you can let `w2` operate on a special array type to accelerate the sinkhorn steps. For example, to exploit a CUDA capable GPU, load the CUDA package and specify `atype = :cuda64`.

If the construction of the cost matrix `c` is a bottleneck in your application, it can be constructed on the GPU as well.
```julia
pixels_1 = convert(CuArray{Float64}, pixels_1)
pixels_2 = convert(CuArray{Float64}, pixels_2)

c = broadcast(pixels_1, reshape(pixels_2, 1, :)) do (i1, j1), (i2, j2)
  ((i1 - i2)^2 + (j1 - j2)^2) / scale
end

w2 = EmpiricalWorkspace(c, eps = 0.1, atype = :cuda64)
```
