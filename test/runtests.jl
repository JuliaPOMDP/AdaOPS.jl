using AdaOPS
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters
using BeliefUpdaters
using StaticArrays

include("rocksample_test.jl")
include("independent_bounds.jl")

pomdp = BabyPOMDP()
pomdp.discount = 1.0
p = solve(AdaOPSSolver(), pomdp)

K = 10
b_0 = initialstate(pomdp)
o = false
tval = 7.0
tree = AdaOPSTree(p, b0)
tree.obs[1] = o
b = WPFBelief([rand(b_0)], [1.0], 1, 0, tree, o)

pol = FeedWhenCrying()
rng = MersenneTwister(2)

# AbstractParticleBelief interface
@testset "WPFBelief" begin
    @test n_particles(b) == 1
    s = particle(b,1)
    @test rand(rng, b) == s
    @test pdf(b, rand(rng, b)) == 1
    sup = support(b)
    @test length(sup) == 1
    @test first(sup) == s
    @test mode(b) == s
    @test mean(b) == s
    @test first(particles(b)) == s
    @test first(weights(b)) == 1.0
    @test first(weighted_particles(b)) == (s => 1.0)
    @test weight_sum(b) == 1.0
    @test weight(b, 1) == 1.0
    @test currentobs(b) == o
    @test history(b)[end].o == o
end

Base.convert(::Type{SVector{1,Float64}}, s::Bool) = SVector{1,Float64}(s)
grid = StateGrid([1.0])
# Type stability
pomdp = BabyPOMDP()
bds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), 0.0)
solver = AdaOPSSolver(bounds=bds,
                      rng=MersenneTwister(4),
                      grid=grid,
                      zeta=0.03,
                      sigma=3,
                      m_init=60,
                      tree_in_info=true
                     )
p = solve(solver, pomdp)

b0 = initialstate(pomdp)
D, Depth = @inferred AdaOPS.build_tree(p, b0)
@inferred action_info(p, b0)
a, info = action_info(p, b0)
info_analysis(info)
@inferred AdaOPS.explore!(D, 1, p)
Δu, Δl = @inferred AdaOPS.expand!(D, D.b, p)
@inferred AdaOPS.backup!(D, 1, p, Δu, Δl)
@inferred AdaOPS.next_best(D, 1, p)
@inferred AdaOPS.excess_uncertainty(D, 1, p)
@inferred action(p, b0)

pomdp = BabyPOMDP()

# constant bounds
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = AdaOPSSolver(bounds=bds, zeta=0.03, m_init=60, xi=0.1, grid=grid, sigma=3, tree_in_info=false, num_b=10_000)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=50)
@time hist = simulate(hr, pomdp, planner)
hist_analysis(hist)
println("Discounted reward is $(discounted_reward(hist))")

# FO policy lower bound
bds = IndependentBounds(SemiPORollout(FeedWhenCrying()), 0.0)
solver = AdaOPSSolver(bounds=bds, zeta=0.03, m_init=60, xi=0.1, grid=grid, sigma=3, tree_in_info=false)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=50)
@time hist = simulate(hr, pomdp, planner)
hist_analysis(hist)
println("Discounted reward is $(discounted_reward(hist))")

# PO policy lower bound
bds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), 0.0)
solver = AdaOPSSolver(bounds=bds, zeta=0.03, m_init=60, xi=0.1, tree_in_info=false)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=50)
@time hist = simulate(hr, pomdp, planner)
hist_analysis(hist)
println("Discounted reward is $(discounted_reward(hist))")

# from README:
using POMDPs, POMDPModels, POMDPSimulators, AdaOPS

pomdp = TigerPOMDP()

solver = AdaOPSSolver(bounds=IndependentBounds(-20.0, 0.0, check_terminal=true), zeta=0.04)
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
