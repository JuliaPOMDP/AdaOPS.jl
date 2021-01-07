function AdaOPSTree(p::AdaOPSPlanner, b_0)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    tree = p.tree
    tree.b_len = 1
    tree.ba_len = 0

    empty!(tree.children[1])
    if p.sol.grid !== nothing
        tree.k[1] = 2
    end
    tree.u[1] = 10000.0 # Should be at least V_max but not to big for numerical reasons
    tree.l[1] = 0.0
    tree.root_belief = b_0

    return tree::AdaOPSTree{S,A,O}
end

function expand!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    all_states = p.all_states # all states generated (may have duplicates)
    resampled = p.resampled # all states resampled
    state_ind_dict = p.state_ind_dict # the index of all generated distinct states
    wdict = p.wdict # weights of child beliefs
    norm_w = p.norm_w # normalized weights for computing distances
    obs_ind_dict = p.obs_ind_dict # the index of observation branches
    freqs = p.freqs # frequency of observations

    # store the likelihood sum and likelihood square sum for convenience of ESS computation
    likelihood_sums = p.likelihood_sums
    likelihood_square_sums = p.likelihood_square_sums

    if p.sol.grid !== nothing
        access_cnts = p.access_cnts # store the access_count grid for each observation branch
        ks = p.ks # track the dispersion of child beliefs
    end
    m_min = p.sol.m_init
    m_max = ceil(Int, p.sol.m_init * p.sol.sigma)

    belief = get_belief(D, b)
    resample!(resampled, belief, p.rng)

    acts = actions(p.pomdp, belief)
    resize_ba!(D, D.ba_len + length(acts))
    ba = D.ba_len

    for a in acts
        empty!(state_ind_dict)
        empty!(wdict)
        empty!(freqs)
        empty!(obs_ind_dict)
        if p.sol.grid !== nothing
            empty!(ks)
        end

        ba += 1
        Rsum = 0.0
        next_states = D.ba_particles[ba]
        empty!(next_states)
        curr_particle_num = 0
        nonterminal = 0

        # generate a initial packing
        while nonterminal < m_min && curr_particle_num < m_max
            curr_particle_num += 1
            if isterminal(p.pomdp, resampled[curr_particle_num])
                all_states[curr_particle_num] = missing
            else
                nonterminal += 1
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, resampled[curr_particle_num], a, p.rng)
                Rsum += r
                all_states[curr_particle_num] = sp
                if !haskey(state_ind_dict, sp)
                    push!(next_states, sp)
                    state_ind_dict[sp] = length(next_states)
                end
                if haskey(obs_ind_dict, o)
                    freqs[obs_ind_dict[o]] += 1
                    obs_ind = obs_ind_dict[o]
                else
                    push!(freqs, 1)
                    obs_ind_dict[o] = length(freqs)
                    obs_ind = obs_ind_dict[o]
                    if p.sol.grid !== nothing
                        fill!(access_cnts[obs_ind], 0)
                        push!(ks, 0)
                    end
                end
                if p.sol.grid !== nothing && access(p.sol.grid, access_cnts[obs_ind], sp, p.pomdp)
                    ks[obs_ind] += 1
                end
            end
        end
        if nonterminal < m_min
            return -D.u[b], -D.l[b]
        end
        m_for_packing = curr_particle_num
        m = curr_particle_num
        # Initialize likelihood_sums and likelihood_square_sums such that the default ESS is Inf
        resize!(likelihood_sums, length(freqs))
        fill!(likelihood_sums, Inf)
        resize!(likelihood_square_sums, length(freqs))
        fill!(likelihood_square_sums, 1.0)
        bp = D.b_len
        for (o, obs_ind) in obs_ind_dict
            w = length(D.weights) > bp ? D.weights[bp+1] : Float64[]
            resize!(w, length(next_states))
            fill!(w, 0.0)
            likelihood_sum = 0.0
            likelihood_square_sum = 0.0
            for j in 1:m
                if all_states[j] !== missing
                    # likelihood = obs_weight(p.pomdp, resampled[j], a, all_states[j], o)
                    likelihood = pdf(observation(p.pomdp, a, all_states[j]), o)
                    likelihood_sum += likelihood
                    likelihood_square_sum += likelihood * likelihood
                    w[state_ind_dict[all_states[j]]] += likelihood
                end
            end
            if p.sol.delta > 0.0
                resize!(norm_w[obs_ind], length(w))
                norm_w[obs_ind][:] = w ./ likelihood_sum
                for (o′, w′) in wdict
                    new_obs_ind = obs_ind_dict[o′]
                    if norm(norm_w[obs_ind] - norm_w[new_obs_ind], 1) <= p.sol.delta
                        freqs[new_obs_ind] += freqs[obs_ind]
                        obs_ind_dict[o] = new_obs_ind
                        o = o′
                        if p.sol.grid !== nothing
                            access_cnts[new_obs_ind] += access_cnts[obs_ind]
                            ks[new_obs_ind] = count(x->x>0, access_cnts[new_obs_ind])
                        end
                        break
                    end
                end
            end
            if !haskey(wdict, o)
                likelihood_sums[obs_ind] = likelihood_sum
                likelihood_square_sums[obs_ind] = likelihood_square_sum
                wdict[o] = w
                bp += 1
            end
        end

        while true
            curr_particle_num = m
            ESS = likelihood_sums .* likelihood_sums ./ likelihood_square_sums
            if p.sol.grid !== nothing
                MESS = ks .|> x->p.sol.MESS(x, p.sol.zeta)
            else
                MESS = m_min
            end
            m = ceil(Int, min(m_max, maximum(curr_particle_num .* MESS ./ ESS)))
            if curr_particle_num >= m
                break
            end

            for (i, s) in enumerate(resampled[curr_particle_num+1:m])
                if isterminal(p.pomdp, s)
                    all_states[curr_particle_num+i] = missing
                else
                    sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
                    Rsum += r
                    all_states[curr_particle_num+i] = sp
                    if haskey(state_ind_dict, sp)
                        state_ind = state_ind_dict[sp]
                        for (o, w) in wdict
                            # likelihood = obs_weight(p.pomdp, s, a, sp, o)
                            likelihood = pdf(observation(p.pomdp, a, sp), o)
                            w[state_ind] += likelihood
                            obs_ind = obs_ind_dict[o]
                            likelihood_sums[obs_ind] += likelihood
                            likelihood_square_sums[obs_ind] += likelihood * likelihood
                        end
                    else
                        push!(next_states, sp)
                        state_ind_dict[sp] = length(next_states)
                        for (o, w) in wdict
                            # likelihood = obs_weight(p.pomdp, s, a, sp, o)
                            likelihood = pdf(observation(p.pomdp, a, sp), o)
                            push!(w, likelihood)
                            obs_ind = obs_ind_dict[o]
                            likelihood_sums[obs_ind] += likelihood
                            likelihood_square_sums[obs_ind] += likelihood * likelihood
                        end
                    end
                    if !haskey(obs_ind_dict, o)
                        w = length(D.weights) > bp ? D.weights[bp+1] : Float64[]
                        resize!(w, length(next_states))
                        fill!(w, 0.0)
                        obs_ind = length(freqs) + 1
                        likelihood_sum = 0.0
                        likelihood_square_sum = 0.0
                        for j in 1:m_for_packing
                            if all_states[j] !== missing
                                # likelihood = obs_weight(p.pomdp, resampled[j], a, all_states[j], o)
                                likelihood = pdf(observation(p.pomdp, a, all_states[j]), o)
                                likelihood_sum += likelihood
                                likelihood_square_sum += likelihood * likelihood
                                state_ind = state_ind_dict[all_states[j]]
                                w[state_ind] += likelihood
                            end
                        end
                        if p.sol.delta > 0.0
                            normalized_w = norm_w[obs_ind]
                            resize!(normalized_w, length(first(norm_w)))
                            normalized_w[:] = w[1:length(normalized_w)] ./ likelihood_sum
                            for (o′, w′) in wdict
                                new_obs_ind = obs_ind_dict[o′]
                                if norm(normalized_w - norm_w[new_obs_ind], 1) <= p.sol.delta
                                    obs_ind_dict[o] = new_obs_ind
                                    o = o′
                                    break
                                end
                            end
                        end
                        if !haskey(wdict, o)
                            for j in (m_for_packing+1):(curr_particle_num+i)
                                if all_states[j] !== missing
                                    # likelihood = obs_weight(p.pomdp, resampled[j], a, all_states[j], o)
                                    likelihood = pdf(observation(p.pomdp, a, all_states[j]), o)
                                    likelihood_sum += likelihood
                                    likelihood_square_sum += likelihood * likelihood
                                    w[state_ind_dict[all_states[j]]] += likelihood
                                end
                            end
                            push!(freqs, 0)
                            push!(likelihood_sums, likelihood_sum)
                            push!(likelihood_square_sums, likelihood_square_sum)
                            wdict[o] = w
                            obs_ind_dict[o] = obs_ind
                            if p.sol.grid !== nothing
                                fill!(access_cnts[obs_ind], 0)
                                push!(ks, 0)
                            end
                            bp += 1
                        end
                    end
                    obs_ind = obs_ind_dict[o]
                    freqs[obs_ind] += 1
                    if p.sol.grid !== nothing && access(p.sol.grid, access_cnts[obs_ind], sp, p.pomdp)
                        ks[obs_ind] += 1
                    end
                end
            end
        end

        D.ba_children[ba] = [D.b_len+1:bp;]
        D.ba_parent[ba] = b
        D.ba_r[ba] = Rsum / curr_particle_num
        D.ba_action[ba] = a
        push!(D.children[b], ba)

        nbp = bp - D.b_len
        bp = D.b_len
        D.b_len += nbp
        resize_b!(D, D.b_len)
        wpf_belief = WPFBelief(next_states, fill(1/length(next_states), length(next_states)), 1.0, D.b_len, D.Delta[b] + 1, D, first(keys(wdict)))
        bounds_dict = bounds(p.bounds, p.pomdp, wpf_belief, wdict, p.sol.bounds_warnings)
        for (o, w) in wdict
            bp += 1
            D.weights[bp] = w
            empty!(D.children[bp])
            D.parent[bp] = ba
            D.Delta[bp] = D.Delta[b] + 1
            D.obs[bp] = o
            obs_ind = obs_ind_dict[o]
            D.obs_prob[bp] = freqs[obs_ind] / curr_particle_num
            D.l[bp], D.u[bp] = bounds_dict[o]
            if p.sol.grid !== nothing
                D.k[bp] = ks[obs_ind]
            end
        end
        D.ba_l[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
        D.ba_u[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
    end
    D.ba_len += length(acts)
    return maximum(D.ba_u[ba] for ba in D.children[b]) - D.u[b], maximum(D.ba_l[ba] for ba in D.children[b]) - D.l[b]
end

function resize_b!(D::AdaOPSTree, n::Int)
    if n > length(D.weights)
        for i in (length(D.weights)+1):n
            push!(D.children, Int[])
        end
        resize!(D.weights, n)
        resize!(D.parent, n)
        resize!(D.Delta, n)
        resize!(D.k, n)
        resize!(D.u, n)
        resize!(D.l, n)
        resize!(D.obs, n)
        resize!(D.obs_prob, n)
    end
end

function resize_ba!(D::AdaOPSTree{S}, n::Int) where S
    if n > length(D.ba_children)
        for i in length(D.ba_children):n
            push!(D.ba_particles, S[])
        end
        resize!(D.ba_children, n)
        resize!(D.ba_parent, n)
        resize!(D.ba_u, n)
        resize!(D.ba_l, n)
        resize!(D.ba_r, n)
        resize!(D.ba_action, n)
    end
end

function get_belief(D::AdaOPSTree, belief::Int)
    if belief == 1
        return D.root_belief
    else
        return WPFBelief(D.ba_particles[D.parent[belief]], D.weights[belief], belief, D.Delta[belief], D, D.obs[belief])
    end
end