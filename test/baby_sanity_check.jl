using POMDPs
using AdaOPS
using POMDPToolbox
using POMDPModels
using ProgressMeter

T = 50
N = 50

pomdp = BabyPOMDP()

bounds = IndependentBounds(RolloutLB(PORollout(FeedWhenCrying(),PreviousObservationUpdater())), 0.0)
# bounds = IndependentBounds(reward(pomdp, false, true)/(1-discount(pomdp)), 0.0)

solver = AdaOPSSolver(epsilon_0=0.1,
                      m=100,
                      D=50,
                      bounds=bounds,
                      T_max=Inf,
                      max_trials=500,
                      rng=MersenneTwister(4)
                     )

rsum = 0.0
fwc_rsum = 0.0
@showprogress for i in 1:N
    planner = solve(solver, pomdp)
    sim = RolloutSimulator(max_steps=T, rng=MersenneTwister(i))
    fwc_sim = deepcopy(sim)
    rsum += simulate(sim, pomdp, planner)
    fwc_rsum += simulate(fwc_sim, pomdp, FeedWhenCrying())
end

@show rsum/N
@show fwc_rsum/N
