using ProgressMeter

T = 50
N = 50

pomdp = BabyPOMDP()

bds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), 0.0)
# bds = IndependentBounds(reward(pomdp, false, true)/(1-discount(pomdp)), 0.0)

solver = AdaOPSSolver(epsilon_0=0.1,
                      m_init=100,
                      max_depth=50,
                      bounds=bds,
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
    global rsum += simulate(sim, pomdp, planner)
    global fwc_rsum += simulate(fwc_sim, pomdp, FeedWhenCrying())
end

@show rsum/N
@show fwc_rsum/N
