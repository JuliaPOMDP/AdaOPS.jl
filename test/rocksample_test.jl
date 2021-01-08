using POMDPPolicies
using RockSample

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

convert(s::RSState, pomdp::RockSamplePOMDP) = [sum(s.rocks)]
grid = StateGrid(convert, range(1, stop=map[2], length=map[2])[2:end])

bound = AdaOPS.IndependentBounds(FORollout(move_east), map[2]*10.0, check_terminal=true, consistency_fix_thresh=1e-5)

solver = AdaOPSSolver(bounds=bound,
                        grid=grid,
                        delta=0.02,
                        zeta=0.01,
                        m_init=30,
                        sigma=2.0,
                        bounds_warnings=true,
                        default_action=move_east,
                        tree_in_info=true
                        )

adaops = solve(solver, m)
@time action(adaops, b0)
# show(stdout, MIME("text/plain"), info[:tree])
D, extra_info = build_tree_test(adaops, b0)
# show(stdout, MIME("text/plain"), D)
extra_info_analysis(D, extra_info)

num_particles = 30000
@show r = simulate(RolloutSimulator(), m, adaops, SIRParticleFilter(m, num_particles), b0, s0)