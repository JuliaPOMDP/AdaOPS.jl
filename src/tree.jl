function AdaOPSTree(p::AdaOPSPlanner, b_0)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    if p.sol.grid !== nothing
        access_cnt = zeros_like(p.sol.grid)
        k = 0
    end
    m = p.init_m
    curr_particle_num = 0
    weight_dict = Dict{S, Float64}()

    while curr_particle_num < m
        for i in (curr_particle_num+1):m
            s = rand(p.rng, b_0)
            if haskey(weight_dict, s)
                weight_dict[s] += 1.0
            else
                weight_dict[s] = 1.0
                if p.sol.grid !== nothing && access(p.sol.grid, access_cnt, s, p.pomdp)
                    k += 1
                end
            end
        end
        curr_particle_num = m
        MESS = p.sol.grid !== nothing ? p.sol.MESS(k, p.sol.zeta) : p.init_m
        m = ceil(Int64, MESS)
    end

    root_belief = WPFBelief(collect(keys(weight_dict)), collect(values(weight_dict)), m, 1, 0)
    l, u = bounds(p.bounds, p.pomdp, root_belief, p.sol.bounds_warnings)

    tree = p.tree
    empty!(tree)
    push!(tree.weights, root_belief.weights)
    push!(tree.children, Int[])
    push!(tree.parent, 0)
    push!(tree.Delta, 0)
    push!(tree.u, u)
    push!(tree.l, l)
    resize!(tree.obs, 1)
    push!(tree.obs_prob, 1.0)
    tree.root_belief = root_belief

    return tree::AdaOPSTree{S,A,O}
end

function expand!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    all_states = p.all_states # all states generated (may have duplicates)
    state_ind_dict = p.state_ind_dict # the index of all generated distinct states
    wdict = p.wdict # weights of child beliefs
    obs_ind_dict = p.obs_ind_dict # the index of observation branches
    freqs = p.freqs # frequency of observations

    # store the likelihood sum and likelihood square sum for convenience of ESS computation
    likelihood_sums = p.likelihood_sums
    likelihood_square_sums = p.likelihood_square_sums

    if p.sol.grid !== nothing
        access_cnts = p.access_cnts # store the access_count grid for each observation branch
        ks = p.ks # track the dispersion of child beliefs
    end

    b = get_wpfbelief(D, b)
    b_resample = resample(b, p.init_m, p.rng)
    for a in actions(p.pomdp, b)
        empty!(all_states)
        empty!(state_ind_dict)
        empty!(wdict)
        empty!(freqs)
        empty!(obs_ind_dict)
        empty!(likelihood_sums)
        empty!(likelihood_square_sums)
        if p.sol.grid !== nothing
            empty!(access_cnts)
            empty!(ks)
        end

        Rsum = 0.0
        next_states = S[]
        m = p.init_m
        curr_particle_num = 0

        while m > curr_particle_num
            resize!(all_states, m)
            for (i, s) in enumerate(b_resample.particles[curr_particle_num+1:m])
                if !isterminal(p.pomdp, s)
                    sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
                    Rsum += r
                    all_states[curr_particle_num+i] = sp
                    if haskey(state_ind_dict, sp)
                        state_ind = state_ind_dict[sp]
                        for (o, w) in wdict
                            likelihood = obs_weight(p.pomdp, s, a, sp, o)
                            w[state_ind] += likelihood
                            obs_ind = obs_ind_dict[o]
                            likelihood_sums[obs_ind] += likelihood
                            likelihood_square_sums[obs_ind] += likelihood * likelihood
                        end
                    else
                        push!(next_states, sp)
                        state_ind_dict[sp] = length(next_states)
                        for (o, w) in wdict
                            likelihood = obs_weight(p.pomdp, s, a, sp, o)
                            push!(w, likelihood)
                            obs_ind = obs_ind_dict[o]
                            likelihood_sums[obs_ind] += likelihood
                            likelihood_square_sums[obs_ind] += likelihood * likelihood
                        end
                    end
                    if !haskey(obs_ind_dict, o)
                        w_temp = zeros(length(next_states))
                        likelihood_sum = 0.0
                        likelihood_square_sum = 0.0
                        for j in 1:(curr_particle_num+i)
                            likelihood = obs_weight(p.pomdp, b_resample.particles[j], a, all_states[j], o)
                            likelihood_sum += likelihood
                            likelihood_square_sum += likelihood * likelihood
                            state_ind = state_ind_dict[all_states[j]]
                            w_temp[state_ind] += likelihood
                        end
                        for (o1, w1) in wdict
                            if norm(w1-w_temp, 1) <= p.sol.delta
                                obs_ind_dict[o] = obs_ind_dict[o1]
                                o = o1
                                w = w1
                                break
                            end
                        end
                        if !haskey(wdict, o)
                            push!(freqs, 0)
                            push!(likelihood_sums, likelihood_sum)
                            push!(likelihood_square_sums, likelihood_square_sum)
                            wdict[o] = w_temp
                            obs_ind_dict[o] = length(freqs)
                            if p.sol.grid !== nothing
                                push!(access_cnts, zeros_like(p.sol.grid))
                                push!(ks, 0)
                            end
                        end
                    end
                    obs_ind = obs_ind_dict[o]
                    freqs[obs_ind] += 1
                    if p.sol.grid !== nothing && access(p.sol.grid, access_cnts[obs_ind], sp, p.pomdp)
                        ks[obs_ind] += 1
                    end
                end
            end
            curr_particle_num = m
            ESS = likelihood_sums .* likelihood_sums ./ likelihood_square_sums
            if p.sol.grid !== nothing
                MESS = ks .|> x->p.sol.MESS(x, p.sol.zeta)
            else
                MESS = p.init_m
            end
            m = ceil(Int64, maximum(curr_particle_num .* MESS ./ ESS))
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
        push!(D.ba_r, Rsum / curr_particle_num)
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
            D.obs_prob[bp] = freqs[obs_ind_dict[o]] / curr_particle_num
            D.l[bp], D.u[bp] = bounds_dict[o]
        end
        push!(D.ba_l, D.ba_r[ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba]))
        push!(D.ba_u, D.ba_r[ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba]))
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

function Base.empty!(D::AdaOPSTree)
    empty!(D.weights)
    empty!(D.children)
    empty!(D.parent)
    empty!(D.Delta)
    empty!(D.u)
    empty!(D.l)
    empty!(D.obs)
    empty!(D.obs_prob)

    empty!(D.ba_particles)
    empty!(D.ba_children)
    empty!(D.ba_parent)
    empty!(D.ba_u)
    empty!(D.ba_l)
    empty!(D.ba_r)
    empty!(D.ba_action)

    D.root_belief = missing
end

function get_wpfbelief(D::AdaOPSTree, b::Int)
    if b == 1
        return D.root_belief
    else
        return WPFBelief(D.ba_particles[D.parent[b]], D.weights[b], b, D.Delta[b], D, D.obs[b])
    end
end