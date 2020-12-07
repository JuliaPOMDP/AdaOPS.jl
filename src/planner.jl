function build_tree(p::AdaOPSPlanner, b_0)
    D = AdaOPSTree(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    while D.u[1]-D.l[1] > p.sol.epsilon_0 &&
          CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        b = explore!(D, 1, p)
        backup!(D, b, p)
        trial += 1
    end

    return D
end

function explore!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    while D.Delta[b] <= p.sol.D &&
        excess_uncertainty(D, b, p) > 0.0
        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end
        b = next_best(D, b, p)
    end

    # if D.Delta[b] > p.sol.D
    #     make_default!(D, b)
    # end
    return b
end

# function make_default!(D::AdaOPSTree, b::Int)
#     D.u[b] = D.l[b]
# end


function backup!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    Δu = 0.0
    Δl = 0.0
    if D.Delta[b] > p.sol.D
        Δu = D.l[b] - D.u[b]
        D.u[b] = D.l[b]
    end
    while b != 1
        bp = b
        ba = D.parent[bp]
        b = D.ba_parent[ba]

        D.ba_u[ba] += discount(p.pomdp) * D.obs_prob[bp] * Δu
        D.ba_l[ba] += discount(p.pomdp) * D.obs_prob[bp] * Δl

        new_u = maximum(D.ba_u[ba] for ba in D.children[b])
        new_l = maximum(D.ba_l[ba] for ba in D.children[b])
        Δu = new_u - D.u[b]
        Δl = new_l - D.l[b]
        D.u[b] = new_u
        D.l[b] = new_l
    end
end
# function backup!(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
#     # Note: maybe this could be sped up by just calculating the change in the one mu and l corresponding to bp, rather than summing up over all bp
#     while b != 1
#         ba = D.parent[b]
#         b = D.ba_parent[ba]

#         D.ba_u[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
#         D.ba_l[ba] = D.ba_r[ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])

#         D.u[b] = maximum(D.ba_u[ba] for ba in D.children[b])
#         D.l[b] = maximum(D.ba_l[ba] for ba in D.children[b])
#     end
# end

function next_best(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    max_u = -Inf
    best_ba = first(D.children[b])
    for ba in D.children[b]
        u = D.ba_u[ba]
        if u > max_u
            max_u = u
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

function excess_uncertainty(D::AdaOPSTree, b::Int, p::AdaOPSPlanner)
    return D.obs_prob[b] * (D.u[b]-D.l[b] - p.sol.xi * (D.u[1]-D.l[1]) / p.discounts[D.Delta[b]+1])
end