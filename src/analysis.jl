function extra_info_analysis(info::Dict)
    k = info[:k]
    if length(k) > 0
        println("k: min/mean/max = $(minimum(k))/$(mean(k))/$(maximum(k))")
        println("Confidence interval (0.1, 0.9) = $(quantile(k, (0.1, 0.9)))")
    end
    m = info[:m]
    println("m: min/mean/max = $(minimum(m))/$(mean(m))/$(maximum(m))")
    println("Confidence interval (0.1, 0.9) = $(quantile(m, (0.1, 0.9)))")
    println("Number of action node expanded: $(length(m))")
    branch = info[:branch]
    println("Number of observation branchs: min/mean/max = $(minimum(branch))/$(mean(branch))/$(maximum(branch))")
    println("Confidence interval (0.1, 0.9) = $(quantile(branch, (0.1, 0.9)))")
end

function build_tree_test(p::AdaOPSPlanner, b_0)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[])

    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        b, new_info = explore_test!(D, 1, p)
        extra_info[:k] = [extra_info[:k]; new_info[:k]]
        extra_info[:m] = [extra_info[:m]; new_info[:m]]
        extra_info[:branch] = [extra_info[:branch]; new_info[:branch]]
        backup!(D, b, p)
        trial += 1
    end
    println("CPU Time is ", (CPUtime_us()-start)*1e-6)

    return D::AdaOPSTree, extra_info
end

function explore_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[])
    while D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0
        if isempty(D.children[b]) # a leaf
            new_info = expand_test!(D, b, p)
            extra_info[:k] = [extra_info[:k]; new_info[:k]]
            extra_info[:m] = [extra_info[:m]; new_info[:m]]
            extra_info[:branch] = [extra_info[:branch]; new_info[:branch]]
        end
        b = next_best(D, b, p)
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
    end
    return b::Int, extra_info
end

function expand_test!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    extra_info = Dict(:k=>Int[], :m=>Int[], :branch=>Int[])

    all_states = S[] # all states generated (may have duplicates)
    state_ind_dict = Dict{S, Int}() # the index of all generated distinct states
    wdict = Dict{O, Array{Float64,1}}() # weights of child beliefs
    obs_ind_dict = Dict{O, Int}() # the index of observation branches
    freqs = Float64[] # frequency of observations

    # store the likelihood sum and likelihood square sum for convenience of ESS computation
    likelihood_sums = Float64[]
    likelihood_square_sums = Float64[]

    if p.sol.grid !== nothing
        access_cnts = Array[] # store the access_count grid for each observation branch
        ks = Int[] # track the dispersion of child beliefs
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

        # Update extra_info
        if p.sol.grid !== nothing
            push!(extra_info[:k], maximum(ks))
        end
        push!(extra_info[:m], curr_particle_num)
        push!(extra_info[:branch], length(wdict))

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
    return extra_info
end