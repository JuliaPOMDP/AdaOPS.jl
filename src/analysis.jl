function tree_analysis(D::AdaOPSTree)
    mean_particle_number = mean(length(particles) for particles in D.ba_particles)
    @show mean_particle_number
end

function find_min_k(p::AdaOPSPlanner, b_0, lower_quantile=0.05)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()
    k_list = []

    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        b, k = explore_k!(D, 1, p)
        k_list = [k_list; k]
        backup!(D, b, p)
        trial += 1
    end

    return D, quantile(k_list, lower_quantile)
end

function explore_k!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    k_list = []
    while D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0
        if isempty(D.children[b]) # a leaf
            k = expand_k!(D, b, p)
            k_list = [k_list; k]
        end
        b = next_best(D, b, p)
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
    end
    return b, k_list
end

function expand_k!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    b = get_wpfbelief(D, b)
    b_resample = resample(b, p.init_m, p.rng)
    k_list = []

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
                MESS = p.sol.MESS(max(kdict[o], p.sol.k_min), p.sol.zeta)
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
        @show m
        k_max = 0
        for o in keys(wdict)
            if kdict[o] > k_max
                k_max = kdict[o]
            end
        end
        push!(k_list, k_max)

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
    return k_list
end