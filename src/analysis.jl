function extra_info_analysis(D::AdaOPSTree, info::Dict)
    k = info[:k]
    if length(k) > 0
        println("k: min/mean/max = $(minimum(k))/$(mean(k))/$(maximum(k))")
        println("Confidence interval (0.1, 0.9) = $(quantile(k, (0.1, 0.9)))")
    end
    m = info[:m]
    println("m: min/mean/max = $(minimum(m))/$(mean(m))/$(maximum(m))")
    println("Confidence interval (0.1, 0.9) = $(quantile(m, (0.1, 0.9)))")
    println("Number of action node expanded: $(length(m))")
    deff = D.Deff[2:D.b_len]
    println("Design Effect: min/mean/max = $(minimum(deff))/$(mean(deff))/$(maximum(deff))")
    println("Confidence interval (0.1, 0.9) = $(quantile(deff, (0.1, 0.9)))")
    branch = info[:branch]
    println("Number of observation branchs: min/mean/max = $(minimum(branch))/$(mean(branch))/$(maximum(branch))")
    println("Confidence interval (0.1, 0.9) = $(quantile(branch, (0.1, 0.9)))")
    depth = info[:depth]
    println("Times of exploration: $(length(depth))")
    println("Depth of exploration: min/mean/max = $(minimum(depth))/$(mean(depth))/$(maximum(depth))")
    println("Confidence interval (0.1, 0.9) = $(quantile(depth, (0.1, 0.9)))")
end

function build_tree_test(p::AdaOPSPlanner, b_0)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[], :depth=>Int[])

    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        new_info = explore_test!(D, 1, p, start)
        extra_info[:k] = [extra_info[:k]; new_info[:k]]
        extra_info[:m] = [extra_info[:m]; new_info[:m]]
        extra_info[:branch] = [extra_info[:branch]; new_info[:branch]]
        extra_info[:depth] = [extra_info[:depth]; new_info[:depth]]
        trial += 1
    end
    println("The runtime is $((CPUtime_us()-start)*1e-6)s")

    return D::AdaOPSTree, extra_info
end

function explore_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner, start::UInt64)
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[], :depth=>Int[])
    while D.Delta[b] < p.sol.D &&
        CPUtime_us()-start < p.sol.T_max*1e6
        if isempty(D.children[b]) # a leaf
            Δu, Δl, new_info = expand_test!(D, b, p)
            extra_info[:k] = [extra_info[:k]; new_info[:k]]
            extra_info[:m] = [extra_info[:m]; new_info[:m]]
            extra_info[:branch] = [extra_info[:branch]; new_info[:branch]]
            if backup!(D, b, p, Δu, Δl) || excess_uncertainty(D, b, p) <= 0.0
                break
            end
        end
        b = next_best(D, b, p)
    end
    if D.Delta[b] == p.sol.D
        backup!(D, b, p, -D.u[b], -D.l[b])
    end
    push!(extra_info[:depth], D.Delta[b])
    return extra_info
end

function expand_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    if D.Deff[b] > p.sol.Deff_thres
        return expand_with_resample_test!(D, b, p)
    else
        return expand_without_resample_test!(D, b, p)
    end
end

function expand_without_resample_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[])

    all_states = p.all_states # all states generated (may have duplicates)
    wdict = p.wdict # weights of child beliefs
    norm_w = p.norm_w # normalized weights for computing distances
    obs_ind_dict = p.obs_ind_dict # the index of observation branches
    freqs = p.freqs # frequency of observations

    # store the likelihood sum and likelihood square sum for convenience of ESS computation
    likelihood_sums = p.likelihood_sums
    likelihood_square_sums = p.likelihood_square_sums

    belief = get_belief(D, b)

    m_min = p.sol.m_init
    m_max = n_particles(belief)

    acts = actions(p.pomdp, belief)
    resize_ba!(D, D.ba_len + length(acts))
    ba = D.ba_len

    for a in acts
        empty!(wdict)
        empty!(freqs)
        empty!(obs_ind_dict)

        ba += 1
        Rsum = 0.0
        next_states = D.ba_particles[ba]
        empty!(next_states)
        curr_particle_num = 0
        nonterminal = 0

        # generate a initial packing
        while nonterminal < m_min && curr_particle_num < m_max
            curr_particle_num += 1
            if isterminal(p.pomdp, belief.particles[curr_particle_num])
                all_states[curr_particle_num] = missing
            else
                nonterminal += 1
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, belief.particles[curr_particle_num], a, p.rng)
                Rsum += weight(belief, curr_particle_num) * r
                all_states[curr_particle_num] = sp
                push!(next_states, sp)
                if haskey(obs_ind_dict, o)
                    freqs[obs_ind_dict[o]] += weight(belief, curr_particle_num)
                    obs_ind = obs_ind_dict[o]
                else
                    push!(freqs, weight(belief, curr_particle_num))
                    obs_ind_dict[o] = length(freqs)
                    obs_ind = obs_ind_dict[o]
                end
            end
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
                    likelihood = weight(belief, j) * pdf(observation(p.pomdp, a, all_states[j]), o)
                    likelihood_sum += likelihood
                    likelihood_square_sum += likelihood * likelihood
                    w[j] += likelihood
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

        for (i, s) in enumerate(belief.particles[curr_particle_num+1:m_max])
            if isterminal(p.pomdp, s)
                all_states[curr_particle_num+i] = missing
            else
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
                Rsum += weight(belief, i) * r
                all_states[curr_particle_num+i] = sp
                push!(next_states, sp)
                for (o, w) in wdict
                    # likelihood = obs_weight(p.pomdp, s, a, sp, o)
                    likelihood = weight(belief, curr_particle_num+i) * pdf(observation(p.pomdp, a, sp), o)
                    push!(w, likelihood)
                    obs_ind = obs_ind_dict[o]
                    likelihood_sums[obs_ind] += likelihood
                    likelihood_square_sums[obs_ind] += likelihood * likelihood
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
                            likelihood = weight(belief, j) * pdf(observation(p.pomdp, a, all_states[j]), o)
                            likelihood_sum += likelihood
                            likelihood_square_sum += likelihood * likelihood
                            w[j] += likelihood
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
                                likelihood = weight(belief, j) * pdf(observation(p.pomdp, a, all_states[j]), o)
                                likelihood_sum += likelihood
                                likelihood_square_sum += likelihood * likelihood
                                w[j] += likelihood
                            end
                        end
                        push!(freqs, 0)
                        push!(likelihood_sums, likelihood_sum)
                        push!(likelihood_square_sums, likelihood_square_sum)
                        wdict[o] = w
                        obs_ind_dict[o] = obs_ind
                        bp += 1
                    end
                end
                obs_ind = obs_ind_dict[o]
                freqs[obs_ind] += weight(belief, curr_particle_num+i)
            end
        end
        curr_particle_num = m_max
        # Update extra_info
        push!(extra_info[:m], curr_particle_num)
        push!(extra_info[:branch], length(wdict))

        D.ba_children[ba] = [D.b_len+1:bp;]
        D.ba_parent[ba] = b
        D.ba_r[ba] = Rsum / weight_sum(belief)
        D.ba_action[ba] = a
        push!(D.children[b], ba)

        nbp = bp - D.b_len
        bp = D.b_len
        D.b_len += nbp
        resize_b!(D, D.b_len)
        wpf_belief = WPFBelief(next_states, fill(1/length(next_states), length(next_states)), 1.0, D.b_len, D.Delta[b] + 1, D, first(keys(wdict)))
        bounds_dict = bounds(p.bounds, p.pomdp, wpf_belief, wdict, p.sol.bounds_warnings)
        ESS = likelihood_sums .* likelihood_sums ./ likelihood_square_sums
        for (o, w) in wdict
            bp += 1
            D.weights[bp] = w
            empty!(D.children[bp])
            D.parent[bp] = ba
            D.Delta[bp] = D.Delta[b] + 1
            D.obs[bp] = o
            obs_ind = obs_ind_dict[o]
            D.obs_prob[bp] = freqs[obs_ind] / weight_sum(belief)
            D.Deff[bp] = curr_particle_num/ESS[obs_ind]
            D.l[bp], D.u[bp] = bounds_dict[o]
        end
        D.ba_l[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
        D.ba_u[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
    end
    D.ba_len += length(acts)
    return maximum(D.ba_u[ba] for ba in D.children[b]) - D.u[b], maximum(D.ba_l[ba] for ba in D.children[b]) - D.l[b], extra_info
end

function expand_with_resample_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[])

    all_states = p.all_states # all states generated (may have duplicates)
    resampled = p.resampled # all states resampled
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
                push!(next_states, sp)
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
                    w[j] += likelihood
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
                    push!(next_states, sp)
                    for (o, w) in wdict
                        # likelihood = obs_weight(p.pomdp, s, a, sp, o)
                        likelihood = pdf(observation(p.pomdp, a, sp), o)
                        push!(w, likelihood)
                        obs_ind = obs_ind_dict[o]
                        likelihood_sums[obs_ind] += likelihood
                        likelihood_square_sums[obs_ind] += likelihood * likelihood
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
                                w[j] += likelihood
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
                                    w[j] += likelihood
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
        # Update extra_info
        if p.sol.grid !== nothing
            push!(extra_info[:k], maximum(ks))
        end
        push!(extra_info[:m], curr_particle_num)
        push!(extra_info[:branch], length(wdict))

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
        ESS = likelihood_sums .* likelihood_sums ./ likelihood_square_sums
        for (o, w) in wdict
            bp += 1
            D.weights[bp] = w
            empty!(D.children[bp])
            D.parent[bp] = ba
            D.Delta[bp] = D.Delta[b] + 1
            D.obs[bp] = o
            obs_ind = obs_ind_dict[o]
            D.obs_prob[bp] = freqs[obs_ind] / curr_particle_num
            D.Deff[bp] = curr_particle_num/ESS[obs_ind]
            D.l[bp], D.u[bp] = bounds_dict[o]
            if p.sol.grid !== nothing
                D.k[bp] = ks[obs_ind]
            end
        end
        D.ba_l[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
        D.ba_u[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
    end
    D.ba_len += length(acts)
    return maximum(D.ba_u[ba] for ba in D.children[b]) - D.u[b], maximum(D.ba_l[ba] for ba in D.children[b]) - D.l[b], extra_info
end