POMDPs.solve(sol::AdaOPSSolver, p::POMDP) = AdaOPSPlanner(sol, p)

function POMDPModelTools.action_info(p::AdaOPSPlanner{S,A}, b) where {S,A}
    info = Dict{Symbol, Any}()
    try
        D, Depth = build_tree(p, b)

        info[:depth] = Depth
        if p.sol.tree_in_info
            info[:tree] = D
        end

        best_l = -Inf
        best_as = A[]
        for ba in D.children[1]
            l = D.ba_l[ba]
            if l > best_l
                best_l = l
                best_as = [D.ba_action[ba]]
            elseif l == best_l
                push!(best_as, D.ba_action[ba])
            end
        end

        return rand(p.rng, best_as)::A, info # best_as will usually only have one entry, but we want to break the tie randomly
    catch ex
        return default_action(p.sol.default_action, p.pomdp, b, ex)::A, info
    end
end

POMDPs.action(p::AdaOPSPlanner{S,A}, b) where {S,A} = first(action_info(p, b))::A
POMDPs.updater(p::AdaOPSPlanner) = SIRParticleFilter(p.pomdp, ceil(Int, p.sol.m_init*p.sol.sigma*50), rng=p.rng)

function Random.seed!(p::AdaOPSPlanner, seed)
    Random.seed!(p.rng, seed)
    return p
end