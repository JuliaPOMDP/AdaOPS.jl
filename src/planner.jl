function build_tree(p::AdaOPSPlanner, b0)
    D = AdaOPSTree(p, b0)
    b = 1
    trial = 1

    Depth = sizehint!(Int[], 2000)
    sol = solver(p)
    start = sol.timer()
    while D.u[1]-D.l[1] > sol.epsilon_0 &&
          sol.timer()-start < sol.T_max &&
          trial <= sol.max_trials
        push!(Depth, explore!(D, 1, p))
        trial += 1
    end
    if sol.timer()-start > sol.timeout_warning_threshold
        @warn(@sprintf("Surpass the time limit. The actual runtime is %3.1fs.
        Hyperparameters: delta=%4.2f, m_min=%3d, m_max=%3d, zeta=%4.2f, grid=%s, bounds=%s",
        sol.timer()-start, sol.delta, sol.m_min, sol.m_max, sol.zeta, typeof(sol.grid), typeof(sol.bounds)))
        info_analysis(Dict(:tree=>D, :depth=>Depth))
    end
    return D, Depth
end

function explore!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    sol = solver(p)
    while D.Delta[b] < sol.max_depth
        if isempty(D.children[b]) # a leaf
            Δl, Δu = expand!(D, b, p)
            if backup!(D, b, p, Δl, Δu) || excess_uncertainty(D, b, sol.xi, p) <= 0.0
                break
            end
        end
        b = next_best(D, b, p)
    end
    if D.Delta[b] == sol.max_depth
        backup!(D, b, p, -D.l[b], -D.u[b])
    end

    return D.Delta[b]
end

function backup!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner, Δl::Float64, Δu::Float64)
    D.u[b] += Δu
    D.l[b] += Δl
    disc = discount(p.pomdp)
    best_a_change = false
    while b != 1
        bp = b
        ba = D.parent[bp]
        b = D.ba_parent[ba]

        # Update u
        D.ba_u[ba] += disc * D.obs_prob[bp] * Δu
        largest_u = -Inf
        largest_ba = 0
        for ba in D.children[b]
            if D.ba_u[ba] > largest_u
                largest_u = D.ba_u[ba]
                largest_ba = ba
            end
        end
        if largest_ba != ba
            best_a_change = true
        end
        Δu = largest_u - D.u[b]
        D.u[b] = largest_u

        # Update l
        if Δl != 0.0
            D.ba_l[ba] += disc * D.obs_prob[bp] * Δl
            if D.l[b] < D.ba_l[ba]
                Δl = D.ba_l[ba] - D.l[b]
                D.l[b] = D.ba_l[ba]
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
    @inbounds for ba in D.children[b]
        if D.ba_u[ba] > max_u
            max_u = D.ba_u[ba]
            best_ba = ba
        end
    end

    max_eu = -Inf
    best_bp = first(D.ba_children[best_ba])
    tolerated_gap = solver(p).xi * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[best_bp]+1]
    @inbounds for bp in D.ba_children[best_ba]
        eu = D.obs_prob[bp] * (D.u[bp]-D.l[bp] - tolerated_gap)
        if eu > max_eu
            max_eu = eu
            best_bp = bp
        end
    end

    return best_bp
end

function excess_uncertainty(D::AdaOPSTree, b::Int, ξ::Float64, p::AdaOPSPlanner)
    return D.obs_prob[b] * (D.u[b]-D.l[b] - ξ * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[b]+1])
end