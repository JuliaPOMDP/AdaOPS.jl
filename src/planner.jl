function build_tree(p::AdaOPSPlanner, b_0)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    Depth = sizehint!(Int[], 2000)
    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        push!(Depth, explore!(D, 1, p))
        trial += 1
    end
    if (CPUtime_us()-start)*1e-6 > p.sol.T_max*p.sol.overtime_warning_threshold
        @warn(@sprintf("Surpass the time limit. The actual runtime is %3.1fs, 
        delta=%4.2f, zeta=%4.2f, m_init=%3d, sigma=%4.2f, grid=%s, bounds=%s",
        (CPUtime_us()-start)*1e-6, p.sol.delta, p.sol.zeta, p.sol.m_init, p.sol.sigma, typeof(p.sol.grid), typeof(p.sol.bounds)))
    end
    return D, Depth
end

function explore!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    while D.Delta[b] < p.max_depth
        if isempty(D.children[b]) # a leaf
            Δl, Δu = expand!(D, b, p)
            if backup!(D, b, p, Δl, Δu) || excess_uncertainty(D, b, p) <= 0.0
                break
            end
        end
        b = next_best(D, b, p)
    end
    if D.Delta[b] == p.max_depth
        backup!(D, b, p, -D.u[b], -D.l[b])
    end

    return D.Delta[b]
end

function backup!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner, Δl::Float64, Δu::Float64)
    D.u[b] += Δu
    D.l[b] += Δl
    best_a_change = false
    while b != 1
        bp = b
        ba = D.parent[bp]
        b = D.ba_parent[ba]

        # Update u
        D.ba_u[ba] += discount(p.pomdp) * D.obs_prob[bp] * Δu
        largest_u = maximum(D.ba_u[ba] for ba in D.children[b])
        if D.ba_u[ba] < largest_u
            best_a_change = true
        end
        Δu = largest_u - D.u[b]
        D.u[b] = largest_u

        # Update l
        if Δl != 0.0
            D.ba_l[ba] += discount(p.pomdp) * D.obs_prob[bp] * Δl
            if D.l[b] < D.ba_l[ba]
                largest_l = D.ba_l[ba]
                Δl = largest_l - D.l[b]
                D.l[b] = largest_l
            else
                Δl = 0.0
            end
        end
    end
    return best_a_change
end

function next_best(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    max_u = -Inf
    best_ba = first(D.children[b])
    for ba in D.children[b]
        @inbounds if D.ba_u[ba] > max_u
            max_u = D.ba_u[ba]
            best_ba = ba
        end
    end

    max_eu = -Inf
    best_bp = first(D.ba_children[best_ba])
    tolerated_gap = p.xi * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[best_bp]+1]
    for bp in D.ba_children[best_ba]
        eu = D.obs_prob[bp] * (D.u[bp]-D.l[bp] - tolerated_gap)
        if eu > max_eu
            max_eu = eu
            best_bp = bp
        end
    end

    return best_bp
end

function excess_uncertainty(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    return D.obs_prob[b] * (D.u[b]-D.l[b] - p.xi * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[b]+1])
end