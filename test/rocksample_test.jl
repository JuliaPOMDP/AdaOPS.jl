using POMDPs, POMDPModelTools, AdaOPS, RockSample, POMDPSimulators

function rsgen(map)
    possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
    selected = unique(rand(possible_ps, map[2]))
    while length(selected) != map[2]
        push!(selected, rand(possible_ps))
        selected = unique!(selected)
    end
    return RockSamplePOMDP(map_size=(map[1],map[1]), rocks_positions=selected)
end

struct MoveEast<:Policy end
POMDPs.action(p::MoveEast, b) = 2
move_east = MoveEast()

map = (11, 11)
m = rsgen(map)

b0 = initialstate(m)
s0 = rand(b0)

bound = AdaOPS.IndependentBounds(FORollout(move_east), map[2]*10.0, check_terminal=true, consistency_fix_thresh=1e-5)

solver = AdaOPSSolver(bounds=bound,
                        delta=0.3,
                        m_init=30,
                        sigma=3.0,
                        bounds_warnings=true,
                        default_action=move_east,
                        tree_in_info=true,
                        num_b=30_000
                        )

adaops = solve(solver, m)
@time action(adaops, b0)
a, info = action_info(adaops, b0)
info_analysis(info)

num_particles = 30000
@time hist = simulate(HistoryRecorder(max_steps=90), m, adaops, SIRParticleFilter(m, num_particles), b0, s0)
hist_analysis(hist)
@show undiscounted_reward(hist)