
using MuSink, Test, Random

@testset "Tree" begin

  root = Tree.Root(0)
  child1 = Tree.new_child!(root, 1)
  child2 = Tree.new_child!(root, 2)
  child3 = Tree.new_child!(child2, 3)
  child4 = Tree.new_child!(child3, 4)
  child5 = Tree.new_child!(child2, 5)

  @test Tree.is_root(root)
  @test !Tree.is_leaf(root)
  @test Tree.is_leaf(child1) && Tree.is_leaf(child4) && Tree.is_leaf(child5)
  
  @test length(root) == length(Tree.descendants(root)) == 6
  @test length(Tree.descendants(child2, false)) == 3

  @test Tree.root(child1) == root
  @test Tree.root(child2) == root
  @test Tree.root(child5) == root

  @test Tree.step_towards(child3, root) == child2
  @test Tree.step_towards(child1, child2) == root

  @test Tree.descendant(root, 1) == child1
  @test Tree.descendant(root, 2) == child2
  @test Tree.descendant(root, 3) == child3
  @test Tree.descendant(root, 4) == child4
  @test Tree.descendant(root, 5) == child5
  @test Tree.descendant(root, 0) == root
  @test Tree.descendant(root, -1) |> isnothing
  @test Tree.descendant(root, 6) |> isnothing

  root = Tree.Sequence(5)
  node = Tree.descendant(root, 2)
  @test Tree.index(node) == 2
  @test length(Tree.neighbors(root)) == 1
  @test length(Tree.neighbors(node)) == 2
  @test Tree.has_neighbor(node, root)
  @test Tree.has_descendant(node, Tree.descendant(root, 3))

  root = Tree.Star(6)
  node = Tree.descendant(root, 2)
  @test Tree.index(node) == 2
  @test length(Tree.neighbors(root)) == 5
  @test length(Tree.neighbors(node)) == 1
  @test Tree.has_neighbor(node, root)
  @test !Tree.has_neighbor(node, Tree.descendant(root, 3))

end

@testset "Cost" begin
  cost = MuSink.Lp(5, 3)
  @test size(cost) == (5, 3)
  ax1, ax2 = MuSink.axes(cost, 3)
  @test all(ax1 .== [0.45, 0.2, 0.05, 0.0, 0.05, 0.2, 0.45])
  @test all(ax2 .== [0.45, 0.2, 0.05, 0.0, 0.05, 0.2, 0.45])
  window = MuSink.window(cost, 3)
  @test all(window .== ax1 .+ ax2')
  @test all(MuSink.window(MuSink.GenericSeparableCost(cost), 2) .== MuSink.window(cost, 2))

  cost = MuSink.LpNotSep(5, 3)
  window = MuSink.window(cost, 3)
  @test all(window .== ax1 .+ ax2')
  @test all(MuSink.window(MuSink.GenericCost(cost), 2) .== MuSink.window(cost, 2))
end

@testset "Penalty" begin
  tv = MuSink.TotalVariation()
  vec = Float64[5, 2, 1, -10]
  MuSink.approx!(vec, tv, 4, 1)
  @test all(vec .== [4, 2, 1, -4])

  kl = MuSink.KullbackLeibler()
  MuSink.approx!(vec, kl, 1, 2)
  @test all(vec .== [4, 2, 1, -4] / 3)
end

@testset "Workspace" begin

  w = MuSink.testspace(n = 5, m = 3, eps = 1, rho = Inf, reach = 2, ntargets = 10)
  @test MuSink.get_eps(w) == 1
  @test MuSink.get_reach(w) == 2
  @test MuSink.get_rho(w) == Inf
  @test MuSink.get_rho(w, 1) === Inf
  @test MuSink.get_weight(w, 1, 2) === 1.0

  obj = MuSink.objective(w)
  for i in 1:200
    MuSink.step!(w, 1)
    obj2 = MuSink.objective(w)
    @test obj <= obj2 || isapprox(obj, obj2, rtol=1e-6)
    obj = obj2
  end
  for i in 1:10
    @test maximum(abs, MuSink.marginal(w, i) .- MuSink.target(w, i)) <= 1e-6
  end

  @test isapprox(MuSink.mass(w), 1)

  w = MuSink.testspace(; atype = :array32)

  MuSink.set_eps!(w, 0.1)
  MuSink.set_reach!(w, 2)
  MuSink.set_rho!(w, 8)
  MuSink.set_rho!(w, 1, 7.5)
  MuSink.set_weight!(w, 1, 2, 0.1)

  @test MuSink.get_eps(w) == 0.1f0
  @test MuSink.get_reach(w) == 2f0
  @test MuSink.get_rho(w) == 8f0
  @test MuSink.get_rho(w, 1) === 7.5f0
  @test MuSink.get_weight(w, 1, 2) === 0.1f0

  # Check that mass of a workspace does not depend on the node over which
  # we first marginalize
  masses = map(MuSink.nodes(w)) do node
    marginal = MuSink.marginal(w, node)
    sum(marginal)
  end
  @test all(isapprox(masses[1], rtol = 1e-6), masses)

  # Check that the cost of transportation does not depend on the ordering
  # in the coupling
  @test isapprox(MuSink.cost(w, 1, 2), MuSink.cost(w, 2, 1), rtol = 1e-6)
end
