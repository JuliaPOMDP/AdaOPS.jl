POMDPs.solve(sol::AdaOPSSolver, p::POMDP) = AdaOPSPlanner(sol, p)

function POMDPModelTools.action_info(p::AdaOPSPlanner, b)
    info = Dict{Symbol, Any}()
    try
        D = build_tree(p, b)

        if p.sol.tree_in_info
            info[:tree] = D
        end

        best_l = -Inf
        best_as = actiontype(p.pomdp)[]
        for ba in D.children[1]
            l = D.ba_l[ba]
            if l > best_l
                best_l = l
                best_as = [D.ba_action[ba]]
            elseif l == best_l
                push!(best_as, D.ba_action[ba])
            end
        end

        return rand(p.rng, best_as)::actiontype(p.pomdp), info # best_as will usually only have one entry, but we want to break the tie randomly
    catch ex
        return default_action(p.sol.default_action, p.pomdp, b, ex)::actiontype(p.pomdp), info
    end
end

POMDPs.action(p::AdaOPSPlanner, b) = first(action_info(p, b))::actiontype(p.pomdp)
POMDPs.updater(p::AdaOPSPlanner) = SIRParticleFilter(p.pomdp, ceil(Int, p.sol.m_init*p.sol.sigma*5), rng=p.rng)

function Random.seed!(p::AdaOPSPlanner, seed)
    Random.seed!(p.rng, seed)
    return p
end