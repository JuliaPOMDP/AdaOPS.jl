using Revise
using TraceCalls

using POMDPs
using ARDESPOT
using ProfileView
using POMDPModels

pomdp = BabyPOMDP()
bounds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = DESPOTSolver(epsilon_0=0.0,
                      bounds=bounds,
                      rng=MersenneTwister(4),
                      random_source=MersenneSource(500, 1, MersenneTwister(4))
                     )
p = solve(solver, pomdp)
b0 = initial_state_distribution(pomdp)
@trace ARDESPOT.build_despot(p, b0)
