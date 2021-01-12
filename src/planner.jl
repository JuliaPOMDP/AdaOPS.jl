function build_tree(p::AdaOPSPlanner, b_0)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        explore!(D, 1, p, start)
        trial += 1
    end
    if (CPUtime_us()-start)*1e-6 > p.sol.T_max*p.sol.overtime_warning_threshold
        @warn ```Surpass the time limit. The actual runtime is $((CPUtime_us()-start)*1e-6)s 
                 delta=$(p.sol.delta) 
                 zeta=$(p.sol.zeta) 
                 m_init=$(p.sol.m_init) 
                 sigma=$(p.sol.sigma) 
                 grid=$(typeof(p.sol.grid)) 
                 bounds=$(typeof(p.sol.bounds))
                 ```
    end
    return D
end

function explore!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner, start::UInt64)
    while D.Delta[b] < p.sol.D &&
        CPUtime_us()-start < p.sol.T_max*1e6
        if isempty(D.children[b]) # a leaf
            Δu, Δl = expand!(D, b, p)
            if backup!(D, b, p, Δu, Δl) || excess_uncertainty(D, b, p) <= 0.0
                break
            end
        end
        b = next_best(D, b, p)
    end
    if D.Delta[b] == p.sol.D
        backup!(D, b, p, -D.u[b], -D.l[b])
    end

    return nothing
end

function backup!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner, Δu::Float64, Δl::Float64)
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
    best_ba = D.children[b][argmax([D.ba_u[ba] for ba in D.children[b]])]
    return D.ba_children[best_ba][argmax([excess_uncertainty(D, bp, p) for bp in D.ba_children[best_ba]])]
end

function excess_uncertainty(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    return D.obs_prob[b] * (D.u[b]-D.l[b] - p.sol.xi * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[b]+1])
end