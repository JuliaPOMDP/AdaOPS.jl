using Revise

using POMDPs
using ARDESPOT
using ProfileView
using POMDPModels

pomdp = BabyPOMDP()
bounds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = DESPOTSolver(epsilon_0=0.0,
                      K=100,
                      lambda=0.01,
                      bounds=bounds,
                      max_trials=3,
                      rng=MersenneTwister(4),
                      random_source=FastMersenneSource(1000, 1)
                     )
@show solver.lambda
p = solve(solver, pomdp)
b0 = initial_state_distribution(pomdp)
@show b0
@time D = ARDESPOT.build_despot(p, b0)
@show action(p, b0)
println(TreeView(D, 1, 90))
