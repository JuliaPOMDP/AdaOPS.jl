function AdaOPSTree(p::AdaOPSPlanner{S,A,O,M}, b0::RB, num_b::Int = 50_000) where {S,A,O,M<:POMDP{S,A,O},RB}
    num_a = length(actions(p.pomdp))
    num_ba = div(num_b, num_a)

    return AdaOPSTree(sizehint!([Float64[]], num_b),
                    sizehint!([Int[]], num_b),
                    sizehint!([0], num_b),
                    sizehint!([0], num_b),
                    sizehint!([10000.0], num_b),
                    sizehint!([-10000.0], num_b),
                    sizehint!(Vector{O}(undef, 1), num_b),
                    sizehint!([1.0], num_b),

                    sizehint!(Vector{S}[], num_ba),
                    sizehint!(Vector{Int}[], num_ba),
                    sizehint!(Int[], num_ba),
                    sizehint!(Float64[], num_ba),
                    sizehint!(Float64[], num_ba),
                    sizehint!(Float64[], num_ba),
                    sizehint!(A[], num_ba),

                    b0,
                    1,
                    0
                 )
end

function expand!(D::AdaOPSTree{S,A,O,RB}, b::Int, p::AdaOPSPlanner{S,A,O,M}) where {S,A,O,M<:POMDP{S,A,O},RB}
    belief, resampled = get_belief(D, b, p)
    if weight_sum(belief) === 0.0
        return -D.l[b], -D.u[b]
    end

    m_max = ceil(Int, p.sol.sigma * p.sol.m_init)
    acts = actions(p.pomdp, belief)
    num_a = length(acts)
    resize_ba!(D, D.ba + num_a, m_max)
    resize_b!(D, D.b + m_max * num_a, m_max, num_a)
    for a in acts
        empty_buffer!(p)
        P, Rsum = propagate_particles(D, belief, a, resampled, p)
        gen_packing!(D, P, belief, a, p)

        D.ba += 1 # increase ba count
        m_max = length(P) # number of particles used
        n_obs = length(p.w) # number of new obs
        fbp = D.b + 1 # first bp
        lbp = D.b + n_obs # last bp
        w_sum = sum(weights(belief)[1:m_max]) # calculate the weight sum of particles used

        # initialize the new action branch
        resize!(D.ba_children[D.ba], n_obs)
        D.ba_children[D.ba] .= fbp:lbp
        D.ba_parent[D.ba] = b
        D.ba_r[D.ba] = Rsum / w_sum
        D.ba_action[D.ba] = a
        push!(D.children[b], D.ba)

        # initialize bounds
        D.b += n_obs
        b′ = WPFBelief(P, first(p.w), 1.0, fbp, D.Delta[b] + 1, D, first(p.obs))
        resize!(p.u, n_obs)
        resize!(p.l, n_obs)
        bounds!(p.l, p.u, p.bounds, p.pomdp, b′, p.w, p.obs, p.sol.D, p.sol.bounds_warnings)

        # initialize new obs branches
        view(D.weights, fbp:lbp) .= p.w
        view(D.parent, fbp:lbp) .= D.ba
        view(D.Delta, fbp:lbp) .= D.Delta[b] + 1
        view(D.obs, fbp:lbp) .= p.obs
        view(D.obs_prob, fbp:lbp) .= p.obs_w ./ w_sum
        view(D.l, fbp:lbp) .= p.l
        view(D.u, fbp:lbp) .= p.u

        # update upper and lower bounds for action selection
        D.ba_l[D.ba] = D.ba_r[D.ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
        D.ba_u[D.ba] = D.ba_r[D.ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
    end
    return maximum(D.ba_l[ba] for ba in D.children[b]) - D.l[b], maximum(D.ba_u[ba] for ba in D.children[b]) - D.u[b]
end

function DesignEffect(D::AdaOPSTree, b::Int)
    w = D.weights[b]
    n = length(w)
    ESS = (sum(w)^2)/(w'w)
    return n/ESS
end

function get_belief(D::AdaOPSTree{S,A,O,RB}, b::Int, p::AdaOPSPlanner{S,A,O,M}) where {S,A,O,M<:POMDP{S,A,O},RB}
    if b === 1
        belief = resample!(p.resampled, D.root_belief, p.pomdp, p.rng)
        resampled = true
    else
        P = D.ba_particles[D.parent[b]]
        W = D.weights[b]
        w_sum = 0.0
        @inbounds for i in eachindex(P)
            if isterminal(p.pomdp, P[i])
                W[i] = 0.0
            else
                w_sum += W[i]
            end
        end
        if w_sum === 0.0
            return WeightedParticleBelief(P, W, w_sum)::WeightedParticleBelief{S}, true
        end
        resampled = DesignEffect(D, b) > p.sol.Deff_thres
        if resampled
            belief = resample!(p.resampled, WeightedParticleBelief(P, W, w_sum), p.rng)
        else
            belief = WeightedParticleBelief(P, W, w_sum)
        end
    end
    return belief::WeightedParticleBelief{S}, resampled::Bool
end

function empty_buffer!(p::AdaOPSPlanner{S,A,O,P,N}) where {S,A,O,P<:POMDP{S,A,O},N}
    empty!(p.obs)
    empty!(p.obs_ind_dict)
    empty!(p.w)
    fill!(p.access_cnt, 0)
    empty!(p.obs_w)
    empty!(p.u)
    empty!(p.l)
    return nothing
end

function gen_packing!(D::AdaOPSTree{S,A,O}, P::Vector{S}, belief::WeightedParticleBelief{S}, a::A, p::AdaOPSPlanner{S,A,O,M}) where {S,A,O,M<:POMDP{S,A,O}}
    m_min = p.sol.m_init
    m_max = length(P)
    w = weights(belief)

    next_obs = 1 # denote the index of the next observation branch
    for i in eachindex(p.obs)
        w′ = resize!(D.weights[D.b+next_obs], m_min)
        o = p.obs[i]
        # reweight first m_min particles
        reweight!(w′, view(w, 1:m_min), view(P, 1:m_min), a, o, p.pomdp)
        # check if the observation is already covered by the packing
        norm_w = p.norm_w[next_obs]
        norm_w .= w′ ./ sum(w′)
        obs_ind = in_packing(norm_w, view(p.norm_w, 1:(next_obs-1)), p.sol.delta)
        if obs_ind !== 0
            # merge new obs into existing obs
            p.obs_w[obs_ind] += p.obs_w[i]
        else
            # add new obs into the packing
            p.obs_w[next_obs] = p.obs_w[i]
            p.obs[next_obs] = o
            push!(p.w, resize!(w′, m_max))
            next_obs += 1
        end
    end

    n_obs = length(p.w)
    resize!(p.obs, n_obs)
    resize!(p.obs_w, n_obs)

    for i in eachindex(p.w)
        reweight!(view(p.w[i], (m_min+1):m_max), view(w, (m_min+1):m_max), view(P, (m_min+1):m_max), a, p.obs[i], p.pomdp)
    end

    return nothing
end

function reweight!(w′::T, w::SubArray{Float64,1}, P::SubArray{S,1}, a::A, o::O, m::M) where {T<:AbstractVector{Float64},S,A,O,M<:POMDP{S,A,O}}
    @inbounds for i in eachindex(w′)
        if w[i] === 0.0
            w′[i] = 0.0
        else
            # w′[i] = w[i] * obs_weight(m, Φ[i], a, P[i], o)
            w′[i] = w[i] * pdf(observation(m, a, P[i]), o)
        end
    end
end

function in_packing(norm_w::Vector{Float64}, W::SubArray{Vector{Float64},1}, δ::Float64)
    @inbounds for i in eachindex(W)
        if norm(W[i] - norm_w, 1) <= δ
            return i
        end
    end
    return 0
end

function propagate_particles(D::AdaOPSTree{S,A,O}, belief::WeightedParticleBelief{S}, a::A, resampled::Bool, p::AdaOPSPlanner{S,A,O,M,N}) where {S,A,O,M<:POMDP{S,A,O},N}
    m_min = p.sol.m_init
    m_max = ceil(Int, p.sol.sigma * p.sol.m_init)

    Φ = particles(belief)
    P = D.ba_particles[D.ba+1]

    Rsum = 0.0
    k = 0 # number of multidimensional bins
    n = 0 # number of used particles
    m = N === 0 ? m_max : m_min # number of needed particles
    while n < m
        for i in (n+1):m
            w = weight(belief, i)
            if w === 0.0
                push!(P, Φ[i])
            else
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, Φ[i], a, p.rng)
                Rsum += w * r
                push!(P, sp)
                obs_ind = get(p.obs_ind_dict, o, 0)
                if obs_ind !== 0
                    p.obs_w[obs_ind] += w
                else
                    push!(p.obs_w, w)
                    push!(p.obs, o)
                    p.obs_ind_dict[o] = length(p.obs)
                end
                if resampled && N !== 0 && access(p.sol.grid, p.access_cnt, P[i], p.pomdp)
                    k += 1
                end
            end
        end
        n = m
        if N !== 0
            m = min(m_max, ceil(Int, p.sol.MESS(k, p.sol.zeta)::Float64))
        end
    end
    return P, Rsum
end

function resize_b!(D::AdaOPSTree, n::Int, m_max::Int, num_a::Int)
    if n > length(D.weights)
        resize!(D.weights, n)
        resize!(D.children, n)
        @inbounds for i in (length(D.parent)+1):n
            D.weights[i] = sizehint!(Float64[], m_max)
            D.children[i] = sizehint!(Int[], num_a)
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

function resize_ba!(D::AdaOPSTree{S}, n::Int, m_max::Int) where S
    if n > length(D.ba_children)
        resize!(D.ba_particles, n)
        resize!(D.ba_children, n)
        @inbounds for i in (length(D.ba_parent)+1):n
            D.ba_particles[i] = sizehint!(S[], m_max)
            D.ba_children[i] = sizehint!(Int[], m_max)
        end
        resize!(D.ba_parent, n)
        resize!(D.ba_u, n)
        resize!(D.ba_l, n)
        resize!(D.ba_r, n)
        resize!(D.ba_action, n)
    end
    return nothing
end