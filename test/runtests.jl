using AdaOPS
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters
using BeliefUpdaters

include("independent_bounds.jl")

pomdp = BabyPOMDP()
pomdp.discount = 1.0

K = 10
b_0 = initialstate(pomdp)
o = false
tval = 7.0
b = WPFBelief([rand(b_0)], [1.0], o)
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
    @test_deprecated previous_obs(b)
    @test history(b)[end].o == o
end

# Type stability
pomdp = BabyPOMDP()
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = AdaOPSSolver(bounds=bds,
                      rng=MersenneTwister(4),
                      k_min=2,
                      zeta=0.04,
                      delta=0.1,
                      xi=0.1,
                      tree_in_info=true
                     )
p = solve(solver, pomdp)

b0 = initialstate(pomdp)
D = @inferred AdaOPS.build_tree(p, b0)
D, extra_info = build_tree_test(p, b0)
extra_info_analysis(extra_info)
@inferred AdaOPS.explore!(D, 1, p)
@inferred AdaOPS.expand!(D, D.b_len, p)
@inferred AdaOPS.backup!(D, 1, p)
@inferred AdaOPS.next_best(D, 1, p)
@inferred AdaOPS.excess_uncertainty(D, 1, p)
@inferred action(p, b0)

# visualization
show(stdout, MIME("text/plain"), D)
a, info = action_info(p, initialstate(pomdp))
show(stdout, MIME("text/plain"), info[:tree])

pomdp = BabyPOMDP()

# constant bounds
bds = (reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = AdaOPSSolver(bounds=bds, k_min=2, zeta=0.04, delta=0.1, xi=0.1)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# FO policy lower bound
bds = IndependentBounds(FORollout(FeedWhenCrying()), 0.0)
solver = AdaOPSSolver(bounds=bds, k_min=2, zeta=0.04, delta=0.1, xi=0.1)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# PO policy lower bound
bds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), 0.0)
solver = AdaOPSSolver(bounds=bds, k_min=2, zeta=0.04, delta=0.1, xi=0.1)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# from README:
using POMDPs, POMDPModels, POMDPSimulators, AdaOPS

pomdp = TigerPOMDP()

solver = AdaOPSSolver(bounds=(-20.0, 0.0), k_min=2, zeta=0.04)
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
