struct AdaOPSTree{S,A,O}
    weights::Vector{Vector{Float64}} # stores weights for *belief node*
    children::Vector{Vector{Int}} # to children *ba nodes*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    u::Vector{Float64}
    l::Vector{Float64}
    obs::Vector{O}
    obs_prob::Vector{Float64}

    ba_particles::Vector{Vector{S}} # stores particles for *ba nodes*
    ba_children::Vector{Vector{Int}}
    ba_parent::Vector{Int} # maps to parent *belief node*
    ba_u::Vector{Float64}
    ba_l::Vector{Float64}
    ba_r::Vector{Float64} # needed for backup
    ba_action::Vector{A}

    root_belief::WPFBelief
end

function AdaOPSTree(p::AdaOPSPlanner, b_0)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    cnt = zeros_like(p.sol.grid)
    k = 0
    m = p.init_m
    curr_particle_num = 0
    particle_set = S[]

    while true
        for i in (curr_particle_num+1):m
            s = rand(p.rng, b_0)
            push!(particle_set, s)
            if access(p.sol.grid, cnt, s, p.pomdp)
                k += 1
            end
        end
        curr_particle_num = m
        MESS = k > p.sol.k_min ? p.sol.MESS(k, p.sol.zeta) : p.init_m
        if m < MESS
            m = ceil(Int64, MESS)
        else
            break
        end
    end

    root_belief = WPFBelief(particle_set, fill(1/length(particle_set), length(particle_set)), 1.0, 1, 0)
    l, u = bounds(p.bounds, p.pomdp, root_belief, p.sol.bounds_warnings)

    return AdaOPSTree{S,A,O}([root_belief.weights],
                         [Int[]],
                         [0],
                         [0],
                         [u],
                         [l],
                         Vector{O}(undef, 1),
                         [1.0],

                         [],
                         [],
                         Int[],
                         Float64[],
                         Float64[],
                         Float64[],
                         A[],

                         root_belief
                 )
end

function expand!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    b = get_wpfbelief(D, b)
    b_resample = resample(b, p.init_m, p.rng)

    for a in actions(p.pomdp, b)
        next_states = S[]
        Rsum = 0.0
        wdict = Dict{O, Array{Float64,1}}() # weights of child beliefs
        fdict = Dict{O, Int}() # frequency of observations

        likelihood_sum_dict = Dict{O, Float64}()
        likelihood_square_sum_dict = Dict{O, Float64}()

        cnt_dict = Dict()
        kdict = Dict{O, Int}()

        m = p.init_m
        curr_particle_num = 0

        while true
            for s in b_resample.particles[curr_particle_num+1:m]
                if !isterminal(p.pomdp, s)
                    sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
                    Rsum += r
                    push!(next_states, sp)
                    for o in keys(wdict)
                        likelihood = obs_weight(p.pomdp, s, a, sp, o)
                        push!(wdict[o], likelihood)
                        likelihood_sum_dict[o] += likelihood
                        likelihood_square_sum_dict[o] += likelihood * likelihood
                    end
                    if haskey(wdict, o)
                        fdict[o] += 1
                    else
                        w_temp = Float64[]
                        for i in 1:length(next_states)
                            push!(w_temp, obs_weight(p.pomdp, b_resample.particles[i], a, next_states[i], o))
                        end
                        for (o1, w1) in wdict
                            if norm(w1-w_temp, 1) <= p.sol.delta
                                o = o1
                                w = w1
                                break
                            end
                        end
                        if haskey(wdict, o)
                            fdict[o] += 1
                        else
                            fdict[o] = 1
                            wdict[o] = w_temp
                            cnt_dict[o] = zeros_like(p.sol.grid)
                            kdict[o] = 0
                            likelihood_sum_dict[o] = sum(w_temp)
                            likelihood_square_sum_dict[o] = dot(w_temp, w_temp)
                        end
                    end
                    if access(p.sol.grid, cnt_dict[o], s, p.pomdp)
                        kdict[o] += 1
                    end
                end
            end
            curr_particle_num = m
            satisfied = true
            for o in keys(wdict)
                ESS = likelihood_sum_dict[o]*likelihood_sum_dict[o]/likelihood_square_sum_dict[o]
                MESS = kdict[o] > p.sol.k_min ? p.sol.MESS(kdict[o], p.sol.zeta) : p.init_m
                # MESS = p.sol.MESS(max(kdict[o], p.sol.k_min), p.sol.zeta*curr_particle_num/(fdict[o]*length(wdict)))
                if ESS < MESS
                    satisfied = false
                    temp_m = curr_particle_num*MESS/ESS
                    if temp_m > m
                        m = temp_m
                    end
                end
            end
            if satisfied
                break
            end
            m = ceil(Int64, m)
            if m > n_particles(b_resample)
                resample!(b_resample, b, m - n_particles(b_resample), p.rng)
            end
        end

        nbps = length(wdict)
        nparticles = length(next_states)
        last_b = length(D.weights)
        push!(D.ba_particles, next_states)
        push!(D.ba_children, [last_b+1:last_b+nbps;])
        push!(D.ba_parent, b.belief)
        push!(D.ba_r, Rsum / nparticles)
        push!(D.ba_action, a)
        ba = length(D.ba_particles)
        push!(D.children[b.belief], ba)

        resize!(D, last_b+nbps)
        bp = last_b
        wpf_belief = WPFBelief(next_states, fill(1/nparticles, nparticles), 1.0, bp, b.depth + 1, D, first(keys(wdict)))
        bounds_dict = bounds(p.bounds, p.pomdp, wpf_belief, wdict, p.sol.bounds_warnings)
        for (o, w) in wdict
            bp += 1
            D.weights[bp] = w
            D.children[bp] = Int[]
            D.parent[bp] = ba
            D.Delta[bp] = b.depth + 1
            D.obs[bp] = o
            D.obs_prob[bp] = fdict[o] / nparticles
            D.l[bp], D.u[bp] = bounds_dict[o]
        end
        push!(D.ba_l, D.ba_r[ba] + sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba]))
        push!(D.ba_u, D.ba_r[ba] + sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba]))
    end
end

function Base.resize!(D::AdaOPSTree, n::Int)
    resize!(D.weights, n)
    resize!(D.children, n)
    resize!(D.parent, n)
    resize!(D.Delta, n)
    resize!(D.u, n)
    resize!(D.l, n)
    resize!(D.obs, n)
    resize!(D.obs_prob, n)
end

function get_wpfbelief(D::AdaOPSTree, b::Int)
    if b == 1
        return D.root_belief
    else
        return WPFBelief(D.ba_particles[D.parent[b]], D.weights[b], b, D.Delta[b], D, D.obs[b])
    end
end