function build_tree(p::PMCPPlanner, b_0)
    D = PMCPTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.U[1]-D.L[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        b = explore!(D, 1, p)
        backup!(D, b, p)
        trial += 1
    end

    return D
end

function explore!(D::PMCPTree, b::Int, p::PMCPPlanner)
    while D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0
        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end
        b = next_best(D, b, p)
    end

    if D.Delta[b] > p.sol.D
        make_default!(D, b)
    end
    return b
end

function make_default!(D::PMCPTree, b::Int)
    D.U[b] = D.L[b]
end

function backup!(D::PMCPTree, b::Int, p::PMCPPlanner)
    # Note: maybe this could be sped up by just calculating the change in the one mu and l corresponding to bp, rather than summing up over all bp
    while b != 1
        ba = D.parent[b]
        b = D.ba_parent[ba]

        D.ba_U[ba] = (D.ba_Rsum[ba] + discount(p.pomdp) * sum(D.U[bp] * D.obs_freq[bp] for bp in D.ba_children[ba]))/p.sol.m
        D.ba_L[ba] = (D.ba_Rsum[ba] + discount(p.pomdp) * sum(D.L[bp] * D.obs_freq[bp] for bp in D.ba_children[ba]))/p.sol.m

        D.U[b] = maximum(D.ba_U[ba] for ba in D.children[b])
        D.L[b] = maximum(D.ba_L[ba] for ba in D.children[b])
    end
end

function next_best(D::PMCPTree, b::Int, p::PMCPPlanner)
    max_U = -Inf
    best_ba = first(D.children[b])
    for ba in D.children[b]
        U = D.ba_U[ba]
        if U > max_U
            max_U = U
            best_ba = ba
        end
    end

    max_eu = -Inf
    best_bp = first(D.ba_children[best_ba])
    for bp in D.ba_children[best_ba]
        eu = excess_uncertainty(D, bp, p)
        if eu > max_eu
            max_eu = eu
            best_bp = bp
        end
    end

    return best_bp

    # ai = indmax(D.ba_mu[ba] for ba in D.children[b])
    # ba = D.children[b][ai]
    # zi = indmax(excess_uncertainty(D, bp, p) for bp in D.ba_children[ba])
    # return D.ba_children[ba][zi]
end

function excess_uncertainty(D::PMCPTree, b::Int, p::PMCPPlanner)
    return D.U[b]-D.L[b] - p.sol.xi * (D.U[1]-D.L[1]) / p.discounts[D.Delta[b]+1]
end