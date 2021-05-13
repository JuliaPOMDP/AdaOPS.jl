function AdaOPSTree(p::AdaOPSPlanner{S,A,O}, b0::RB) where {S,A,O,RB}
    sol = solver(p)
    b0, w_sum = strip_terminals(b0, p.pomdp)
    if w_sum == 0.0
        error("All states in the current belief are terminal.")
    end
    belief = resample!(b0, p)

    if sol.tree_in_info || p.tree === nothing
        p.tree = AdaOPSTree([Float64[]],
                        [1:0],
                        [0],
                        [0],
                        [10000.0],
                        [-10000.0],
                        Vector{O}(undef, 1),
                        [1.0],

                        Vector{S}[],
                        UnitRange{Int64}[],
                        Int[],
                        Float64[],
                        Float64[],
                        Float64[],
                        A[],

                        belief,
                        1,
                        0
                    )
        resize_b!(p.tree, sol.num_b)
        resize_ba!(p.tree, sol.num_b)
    else
        reset!(p.tree, belief)
    end
    return p.tree::AdaOPSTree{S,A,O}
end

function reset!(tree::AdaOPSTree, b0::WeightedParticleBelief)
    empty!.(tree.weights)
    fill!(tree.children, 1:0)
    empty!.(tree.ba_particles)
    tree.u[1] = 10000.0
    tree.l[1] = -10000.0
    tree.b = 1
    tree.ba = 0
    tree.root_belief = b0
    return nothing
end

function expand!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    belief = get_belief(D, b, p)
    if weight_sum(belief) == 0.0
        return -D.l[b], -D.u[b]
    end

    sol = solver(p)
    m_max = sol.m_max
    acts = actions(p.pomdp, belief)
    num_a = length(acts)
    resize_ba!(D, D.ba + num_a)
    resize_b!(D, D.b + m_max * num_a)
    D.children[b] = (D.ba+1):(D.ba+num_a)
    for a in acts
        empty_buffer!(p)
        S, O, R = propagate_particles(D, belief, a, p)
        gen_packing!(D, O, belief, p)

        D.ba += 1 # increase ba count
        n_obs = length(p.w) # number of new obs
        fbp = D.b + 1 # first bp
        lbp = D.b + n_obs # last bp

        # initialize the new action branch
        D.ba_children[D.ba] = fbp:lbp
        D.ba_parent[D.ba] = b
        D.ba_r[D.ba] = R
        D.ba_action[D.ba] = a

        # initialize bounds
        D.b += n_obs
        b′ = WPFBelief(S, first(p.w), 1.0, fbp, D.Delta[b] + 1, D, first(O))
        resize!(p.u, n_obs)
        resize!(p.l, n_obs)
        bounds!(p.l, p.u, p.bounds, p.pomdp, b′, p.w, O, sol.max_depth, sol.bounds_warnings)

        # initialize new obs branches
        view(D.weights, fbp:lbp) .= p.w
        view(D.parent, fbp:lbp) .= D.ba
        view(D.Delta, fbp:lbp) .= D.Delta[b] + 1
        view(D.obs, fbp:lbp) .= O
        view(D.obs_prob, fbp:lbp) .= p.obs_w ./ weight_sum(belief)
        view(D.l, fbp:lbp) .= p.l
        view(D.u, fbp:lbp) .= p.u

        # update upper and lower bounds for action selection
        D.ba_l[D.ba] = D.ba_r[D.ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
        D.ba_u[D.ba] = D.ba_r[D.ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
    end
    return maximum(D.ba_l[ba] for ba in D.children[b]) - D.l[b], maximum(D.ba_u[ba] for ba in D.children[b]) - D.u[b]
end

function get_belief(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    if b === 1
        return D.root_belief
    end
    belief, w_sum = strip_terminals(WeightedParticleBelief(D.ba_particles[D.parent[b]], D.weights[b]), p.pomdp)
    if w_sum != 0.0 && DesignEffect(D, b) > solver(p).Deff_thres
        return resample!(belief, p)
    else
        return belief
    end
end

function strip_terminals(b, m::POMDP)
    return b, 1.0
end

function strip_terminals(b::AbstractParticleBelief, m::POMDP)
    w_sum = 0.0
    P = particles(b)
    W = weights(b)
    @inbounds for (i, s) in enumerate(P)
        if isterminal(m, s)
            W[i] = 0.0
        else
            w_sum += W[i]
        end
    end
    return WeightedParticleBelief(P, W, w_sum), w_sum
end

function DesignEffect(D::AdaOPSTree, b::Int)
    w = D.weights[b]
    n = length(w)
    ESS = (sum(w)^2)/dot(w, w)
    return n/ESS
end

function empty_buffer!(p::AdaOPSPlanner)
    empty!(p.obs)
    empty!(p.obs_ind_dict)
    empty!(p.w)
    empty!(p.obs_w)
    empty!(p.u)
    empty!(p.l)
    return nothing
end

function resample!(b::B, p::AdaOPSPlanner{S,A,O,M,N}) where {B,S,A,O,M<:POMDP{S,A,O},N}
    if N == 0
        # the number of resampled particles is default to p.sol.m_max
        return resample!(p.resampled, b, p.pomdp, p.rng)
    else
        fill!(p.access_cnt, 0)
        return kld_resample!(b, p)
    end
end

function kld_resample!(b::AbstractParticleBelief, p::AdaOPSPlanner)
    sol = solver(p)
    k = 0
    for s in particles(b)
        k += access(sol.grid, p.access_cnt, s, p.pomdp)
    end
    m = clamp(ceil(Int, KLDSampleSize(k, sol.zeta)), sol.m_min, sol.m_max)
    resize!(p.resampled, m)
    return resample!(p.resampled, b, p.pomdp, p.rng)
end

function kld_resample!(b, p::AdaOPSPlanner)
    sol = solver(p)
    m_max = sol.m_max
    S_resampled = particles(p.resampled)
    resize!(S_resampled, m_max)
    rng = p.rng

    n = 0
    m = sol.m_min
    k = 0
    while n < m
        for i in (n+1):m
            s = rand(rng, b)
            while isterminal(p.pomdp, s)
                s = rand(rng, b)
            end
            S_resampled[i] = s
            k += access(sol.grid, p.access_cnt, s, p.pomdp)
        end
        n = m
        m = min(m_max, ceil(Int, KLDSampleSize(k, sol.zeta)))
    end
    resize!(p.resampled, n)
    return p.resampled
end

function propagate_particles(D::AdaOPSTree, belief::WeightedParticleBelief, a, p::AdaOPSPlanner)
    S = D.ba_particles[D.ba+1]
    O = p.obs

    Rsum = 0.0
    k = 0 # number of multidimensional bins occupied
    for (i, s) in enumerate(particles(belief))
        w = weight(belief, i)
        if w == 0.0
            push!(S, s)
        else
            sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
            Rsum += w * r
            p.obs_dists[i] = observation(p.pomdp, a, sp)
            push!(S, sp)
            obs_ind = get(p.obs_ind_dict, o, 0)
            if obs_ind !== 0
                p.obs_w[obs_ind] += w
            else
                push!(p.obs_w, w)
                push!(O, o)
                p.obs_ind_dict[o] = length(O)
            end
        end
    end
    return S, O, Rsum/weight_sum(belief)
end

function gen_packing!(D::AdaOPSTree, O, belief::WeightedParticleBelief, p::AdaOPSPlanner)
    sol = solver(p)
    m = n_particles(belief)
    w = weights(belief)

    next_obs = 1 # denote the index of the next observation branch
    for i in eachindex(O)
        w′ = resize!(D.weights[D.b+next_obs], m)
        o = O[i]
        reweight!(w′, w, o, p.obs_dists)
        # check if the observation is already covered by the packing
        w′ .= w′ ./ sum(w′)
        obs_ind = in_packing(w′, p.w, sol.delta)
        if obs_ind != 0
            # merge the new obs into an existing obs
            p.obs_w[obs_ind] += p.obs_w[i]
        else
            # add the new obs into the packing
            p.obs_w[next_obs] = p.obs_w[i]
            O[next_obs] = o
            push!(p.w, w′)
            next_obs += 1
        end
    end

    n_obs = length(p.w)
    resize!(O, n_obs)
    resize!(p.obs_w, n_obs)

    return nothing
end

function reweight!(w′::AbstractVector{Float64}, w::AbstractVector{Float64}, o, obs_dists)
    @inbounds for i in eachindex(w′)
        if w[i] == 0.0
            w′[i] = 0.0
        else
            # w′[i] = w[i] * obs_weight(m, Φ[i], a, S[i], o)
            w′[i] = w[i] * pdf(obs_dists[i], o)
        end
    end
end

function in_packing(w::Vector{Float64}, W::AbstractVector{Vector{Float64}}, δ::Float64)
    @inbounds for i in eachindex(W)
        if cityblock(W[i], w) <= δ
            return i
        end
    end
    return 0
end

function Base.resize!(b::WeightedParticleBelief, n::Int)
    resize!(particles(b), n)
    resize!(weights(b), n)
    b.weight_sum = n
end

function resize_b!(D::AdaOPSTree, n::Int)
    if n > length(D.weights)
        resize!(D.weights, n)
        resize!(D.children, n)
        @inbounds for i in (length(D.parent)+1):n
            D.weights[i] = Float64[]
            D.children[i] = 1:0
        end
        resize!(D.parent, n)
        resize!(D.Delta, n)
        resize!(D.u, n)
        resize!(D.l, n)
        resize!(D.obs, n)
        resize!(D.obs_prob, n)
    end
    return nothing
end

function resize_ba!(D::AdaOPSTree{S}, n::Int) where S
    if n > length(D.ba_children)
        resize!(D.ba_particles, n)
        resize!(D.ba_children, n)
        @inbounds for i in (length(D.ba_parent)+1):n
            D.ba_particles[i] = S[]
        end
        resize!(D.ba_parent, n)
        resize!(D.ba_u, n)
        resize!(D.ba_l, n)
        resize!(D.ba_r, n)
        resize!(D.ba_action, n)
    end
    return nothing
end