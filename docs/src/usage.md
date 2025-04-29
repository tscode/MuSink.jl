# Usage

MuSink aims to implement fast and stable UMOT solvers for measures on uniform 2d grids with tree-based cost structures. See section 5 of [this publication](https:// arxiv.org/ abs/2103.10854) for the conceptual background.

## Basics
In the following, we want to solve a UMOT problem for a temporal sequence of three images. In a first step, we construct the cost tree:
```julia
using MuSink
root = Tree.Root(1)                   # root node with index 1
child_2 = Tree.new_child!(root, 2)    # child with index 2
child_3 = Tree.new_child!(child_2, 3) # child of child 2 with index 3
```
Since we only need the root node object in the following code (we usually reference nodes by their index), we could also have written `root = Tree.Sequence(3)` to accomplish the same goal. Next, we will attach images to each node. These images act as the marginal targets for the UMOT problem
```julia
img1 = zeros(4, 7)
img1[:, 1] .= 1

img2 = zeros(4, 7)
img2[:, 3] .= 1

img3 = zeros(4, 7)
img3[:, 7] .= 1

targets = Dict(
  1 => img1,
  2 => img2,
  3 => img3
)
```
In the last line, we assign one image to every node-index of the tree via a dictionary. Alternatively, one can also specify targets via a list / vector `targets = [img1, img2, img3]`, in wich case images are associated to nodes in order of increasing indices. It is important that all images have the same resolution (`4x7` in this case).

The targets in this example tell us that a vertical strip of mass is supposed to move from left (`img1`) to the middle (`img2`) to the right (`img3`) in three time steps. It should be easy to reconstruct this horizontal flow of mass via UMOT.

To do so, we construct a `MuSink.Problem`, which collects all of the aspects of a UMOT problem that are unlikely to change.
```julia
problem = Problem(root, targets; cost = Lp(4, 7, p = 2, max = 1), penalty = TotalVariation())
```
Here, we provide the `MuSink.Problem` with the tree topology, the target measures, the cost functional (L2 costs, scaled such that the diagonal corresponds to cost 1), and the penalty between the marginal measures of the solution and the targets (the total variation norm in this case). Additionally, we could also have provided optional reference measures via `references = ...` and `reference_mass = ...`. By default, the counting measure is used as reference for each target.

Now that we have collected the *fixed* parts of the UMOT problem, we are ready to combine them with the *moving* parts. This results in a `MuSink.Workspace`, which is the most important structure in `MuSink`, as it takes care of essentially all computational aspects. In its basic form, a workspace can be constructed via
```julia
ws = Workspace(problem, eps = 1, rho = Inf)
```
There are several other optional arguments that we will come back to later. Important are the arguments `eps`, which is the strength of the entropic regularization (usually denoted by an ε an mathematical literature), and the parameter `rho` that influences the strength of the marginal penalty. Rules of thumb:

1. larger values of `eps` will make the resulting coupling more blurry (and the
   limit of small `eps` corresponds to the unregularized OT problem),
2. larger values of `rho` will force the marginal measures to better coincide
  with the target measures. The choice `rho = Inf` basically takes the 'U' from
  'UMOT' away, as we now are in a setting with strict marginal constraints.

We are now set to conduct Sinkhorn steps on the object `ws`, each of which should bring us closer to the true solution of the UMOT problem. In this simple example, `10` steps should suffice:
```julia
step!(ws, 10)
```
Congratulations! You have just calculated the solution to a UMOT problem. Now, what do we do? First, we could check consistency. Since `rho = Inf` means harsh marginal restrictions, the marginal and the target measures should be very similar:
```julia
target(ws, 1)   # this is just img1 from above
marginal(ws, 1) # this is the first marginal of the coupling after 10 Sinkhorn steps
```
We can also calculate the total mass of the solution coupling:
```julia
mass(ws)
```
As expected, it amounts to 4, which is also the mass of all of the marginals.

However, we are far more interested in the actual transport that the workspace currently hides from us. For example, where does the first pixel of the first image go to?
The answer is near:
```julia
transport(ws, 1, 2, (1, 1)) # transport of pixel (1, 1) from first to second image
transport(ws, 1, 3, (1, 1)) # transport of pixel (1, 1) from first to third image
```
Curiously, this pixel seems to be smeared all over images 2 and 3. Why? This is the consequence of the choice `eps = 1` from above. If we repeat the Sinkhorn steps with a new workspace and `eps = 0.1`, the result will be sharper. To get a really crisp result, we can run (usually, smaller `eps` implies that more steps are needed for convergence)
```julia
set_eps!(ws, 0.01)
step!(ws, 100)
transport(ws, 1, 2, (1, 1))
```
Resetting a parameter in this way (which also works for `rho`, or any other parameter of a workspace), keeps the previously obtained dual solutions in tact and effectively uses them as new initial conditions.

## ε-Scaling

If we want to calculate sharp transport plans, the value of epsilon has to be quite small. However, this raises some problems. Besides issues of numerical stability in the expdomain (i.e., one usually has to use `logdomain = true` for stable computations), we find that the smaller the value of `eps`, the less a single Sinkhorn iteration will impact the potentials and thus the transport plan. Just naively setting `eps` to something small can therefor be a bad idea and may lead to excessive computational efforts.

The common strategy to counter this effect is called *ε-scaling*. The idea is simply to iteratively calculate dual solutions for increasingly smaller values of `eps`, using the previous solutions as initial values for the next round. This scheme is supported by MuSink.jl via the `converge!` function.
```julia
ws = Workspace(problem, eps = 1, rho = Inf)
# This will start with eps = 1 and successively scale it down by a factor of 0.85
MuSink.converge!(ws, target_eps = 0.01, scaling = 0.85)
```
Instead of a target value of `eps`, which can be unhelpfully abstract (what does `eps = 1e-3` mean for the transport plan in a specific problem?), one can also set a target blur value. This value provides a rough intuition into how many target pixels an average source pixel is "blurred" in a standard-deviation sense. You should be careful not to set the value too small, since the solution of the problem may require some amount of mass splitting.
```julia
ws = Workspace(problem, eps = 1, rho = Inf)
MuSink.converge!(ws, target_blur = 1.5)
```
In this trivial example, this does not lead to any ε-scalings. In larger images, `target_blur` is more useful. See the documentation of `blur` and `converge!` for more information.

## Reductions

Except for extracting marginals or finding out how single pixels are moved, what else can we do? For example, we can easily access the dual potentials via `potential(ws, index)`, where `index` indexes a node in the problem tree. These potentials can, at least in principle, be used to compute any UMOT-related quantity.

In practice, the most interesting quantities might involve operations on the transport plan between two nodes. Of course, this can be realized via the pixelwise transport functionalty introduced above (`transport(ws, ...)`), or even by calling the function `dense(ws, index_a, index_b)`, which returns the full transport plan. Unfortunately, it is quite expensive to evaluate the full transport plan like this, both in time and memory, and any attemps of this sort will fail miserably on larger images (say 256x256 or 512x512).

Fortunately, certain properties of the transport plan *between neighboring nodes* can be computed much faster. This includes the actual transport cost, as well as the 'barycentric projection' of the transport plan. The latter is fancy wording for the simple idea of finding the mean target pixel position that mass is transported to, given a source pixel position. Play around with the following methods:
```julia
Reductions.cost(ws, 1, 2) # the pixelwise cost of transport from image 1 to 2
Reductions.imap(ws, 1, 2) # the pixelwise average i-positions (first coordinate)
Reductions.jmap(ws, 1, 2) # the pixelwise average j-positions (first coordinate)
Reductions.ishift(ws, 1, 2) # the pixelwise average i-shift
Reductions.jshift(ws, 1, 2) # the pixelwise average j-shift
Reductions.ivar(ws, 1, 2) # the pixelwise variance of the i-coordinate
Reductions.jvar(ws, 1, 2) # the pixelwise variance of the j-coordinate
Reductions.std(ws, 1, 2) # the pixelwise standard deviation
```
For convenience, the aggregate function `cost(ws, index_a, index_b)` that sums all values from `Reductions.cost(ws, index_a, index_b)` to a scalar (the total transport cost), is also provided.

The prefix `Reductions` for these type of operations is used since they effectively amount to partial weighted summations of the transport plan. If the weights have a suitable structure, an efficient implementation is possible. If you want to define your own Reductions, you can easily do so. For example, the following code constructs and applies a *colocalization* reduction, which, for each pixel, adds up mass contributions from transport with a cost value `<= 0.1`.
```julia
colocalize = Reductions.Reduction((diff_i,diff_j,c) -> c <= 0.1)
colocalize(ws, 1, 2)
```
The arguments `diff_i` and `diff_j`, which we did not make use of in the anonymous function above, correspond to the shift in the first respectively second coordinate. Note that, for technical reasons, the reduction function has to be non-negative. If you want to integrate the plan over negative functions, you have to offset them first, and afterwards subtract the offset again. Alternatively, you can use two reductions, one for the positive / negative part each.

In case you are interested in the conditional colocalization, this is as simple as `colocalize(ws, 1, 2) ./ marginal(ws, 1)`, or alternatively `colocalize(ws, 1, 2, conditional = true)`. If you want to calculate the colocalization between non-adjacent nodes, this is, unfortunately, much harder (or even impossible) to implement efficiently. Currently, this is not supported and `colocalize(ws, 1, 3)` will error.

## Relaxing marginal constraints
Up to now, each example used hard marginal constraints (`rho = Inf`). We next see what happens when we relax them. As we will see, this can be much more tricky than it initially seems.

Let us just experiment a bit with `rho`. A first try:
```julia
ws = Workspace(problem, eps = 0.01, rho = 1)
step!(ws, 1000)
marginal(ws, 1)
```
Apparantly, this value of `rho` is still too large to create / destroy mass. Let us go smaller:
```julia
ws = Workspace(problem, eps = 0.01, rho = 0.1)
step!(ws, 1000)
marginal(ws, 1)
```
Okay, something has changed. However, it is not clear if this is reasonable. For some reason, the total mass has **increased** by a little bit! See yourself: `mass(ws)`.

Let's continue:
```julia
ws = Workspace(problem, eps = 0.01, rho = 0.01)
step!(ws, 1000)
marginal(ws, 1)
```
Now, that is a healthy mixture between reasonable and strange. On the one hand, the marginal now resembles more of a mix of the three marginal measures, just as expected. On the other hand, its total mass has increased greatly! Surely, this is a bug?

Well, no. This is a real-world consequence of the choice of the counting measure as a reference. The problem is that the
product of the three counting measures used as reference has a mass of `(4*7)^3 = 21952`. Therefore, for small values of `rho`, where the coupling has no interest in following the marginal constraints, this mass `21952` will be a point of attraction. In the extreme case of zero costs and `rho = 0`, we would expect exactly this amount of mass in the coupling.
Since the cost function is not zero here, we get a lower value in our example, but still something much larger than 4:
```julia
ws = Workspace(problem, eps = 0.01, rho = 0)
step!(ws, 1000)
mass(ws)
```
To further check this hypothesis, we can recalculate the mass under trivial costs:
```julia
problem2 = Problem(root, targets; cost = Lp(4, 7, max = 0))
ws2 = Workspace(problem2, eps = 0.01, rho = 0)
step!(ws2, 1000)
mass(ws2)
```
So, what can we then do if we want to decrease `rho` without letting the mass explode? Maybe it helps if we make `eps` smaller?. Then, the effect of an high product mass should set in much later.

Well, looking at the marginals for `eps = 0.001` and `rho = 0.01` shows that there is no effect yet (i.e., it works as if `rho = Inf`). However, if we now decrease `rho` further, we **lose** mass extremely quickly! Watch this:
```julia
ws = Workspace(problem, eps = 0.001, rho = 0.08) # back to the problem with non-trivial costs
step!(ws, 1000)
mass(ws) # ~ 0.00055...
```
This time, instead of exploding, the mass collapses! Reason: at this point, the actual transport (and not the entropic penalty) is the most costly term in the UMOT objective, so transporting nearly no mass is preferable.

This example should make clear that there is a delicate balance going on, with several competing effects determining how useful the actual solution is. Unfortunately, it is not trivial to predict which magnitudes of `eps` and `rho` are suitable and lead to the desired outcomes. However, one universal trick to to stabilize the situation is to set the product reference mass to a reasonable value via
```
problem = Problem(..., reference_mass = 4, ...)
```
At least, this prevents mass explosions if `eps` is too large while `rho` is too small.

## Barycenters

The previous section might have given the impression that reducing `rho` leads to many disadvantages and is not worth it. However, sometimes it is necessary and safe to do so. For example, if the total masses of the targets would be distinct, then
letting `rho = Inf` would lead to an indefinite oscillation of mass during the Sinkhorn update step (this can be checked
via the function `MuSink.mass_instability`). In this case, finding a suitable value of `rho` is key for the convergence speed.

Another scenario where we want to soften marginal constraints are *barycenter* calculations. Going back to our previous example, we want to set `rho = 0` **only** for the second node. This effectively means that the marginal for the second node becomes a UMOT-Barycenter of the first and third (see 5.1 in the [reference paper](https:// arxiv.org/ abs/2103.10854)).
Fortunately, this works well:
```julia
ws = Workspace(problem, eps = 0.01, rho = Inf)
set_rho!(ws, 2, 0)
step!(ws, 1000)
marginal(ws, 2)
```
Decreasing `eps` further will make the barycenter sharper. Since this is central functionality, calculating barycenters has a shortcut implementation. We can thus easily calculate the barycenter of all three images:
```julia
ws = Barycenter([img1, img2, img3], eps = 0.01)
step!(ws, 1000)
marginal(ws, 1) # in this tree, the barycenter can be found on node 1
```

## Advanced workspace options
When creating a workspace, there is some more flexibility that was previously not mentioned. First, besides the option to set the values of `rho` for each node seperately, one can also set a `weight` for any edge in the tree. This weight is multiplied to the default costs. Example:
```julia
ws = Barycenter([img1, img3], eps = 0.01)
set_weight!(ws, 1, 2, 0.2) # cost between the barycenter node and img1 is weakened
step!(ws, 1000)
marginal(ws, 1) # the barycenter has shifted to the right
```

Next, when creating a workspace, we can choose the domain that the dual potentials are stored and processed in. It either works in the exponential domain (with auxiliary potentials) or the logarithmic domain (with the actual dual potentials). The former leads to faster computations but becomes unstable for small `eps`. For this reason, the logdomain is set as default. When you want to change this, you can do so either by setting `logdomain = false` in the workspace constructor or calling `set_logdomain!(ws, false)` afterwards.

One other issue that can affect the stability and performance of the algorithm is the update order within a Sinkhorn step. In each step, all edges of the tree have to be updated in both directions. Different options can be selected via the keyword option `stepmode = ...` or the `set_stepmode!(ws, ...)` function. The update order proposed in the [reference paper](https:// arxiv.org/ abs/2103.10854) usually works fine, but has a crucial failure mode if the root node equals a node with small `rho`. For this reason, a stable alternative (`stepmode = :stable`) without this drawback is implemented as default. This method uses more updates than might be necessary for most problems, so a more aggressive mode (`stepmode = :alternate`), which should work decently in most settings, is implemented as well.

## Accelerating MuSink
MuSink supports different array types that workspace operations are based on. By default, `Array{Float64}` is used. To create a workspace that works with single-floating point precision instead, you can call either of
```julia
ws = Workspace(problem, kwargs..., atype = Array{Float32})
# or
ws = Workspace(problem, kwargs..., atype = :array32)
```
Depending on the hardware, this can increase the performance of the workspace. In particular, if the package `LoopVectorization` is installed and loaded in the same julia session, SIMD optimizations will lead to further performance gains (a factor of 2 or 3).

On cuda-capable hardware, loading the package `CUDA.jl` and calling `ws = Workspace(..., atype = :cuda32)` will create a cuda-accelerated workspace with considerable speedups (depending on the GPU, improvements of a factor of 2 to 20 are possible). Similar options exist for `Metal.jl` with `atype = :metal32` or `oneAPI.jl` with `atype = :one32`.
