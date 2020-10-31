POMDPs.solve(sol::PMCPSolver, p::POMDP) = PMCPPlanner(sol, p)

function POMDPModelTools.action_info(p::PMCPPlanner, b)
    info = Dict{Symbol, Any}()
    try
        D = build_tree(p, b)

        if p.sol.tree_in_info
            info[:tree] = D
        end

        if isempty(D.children[1]) && D.U[1] - D.L[1] <= p.sol.epsilon_0
            throw(NoGap(D.l_0[1]))
        end

        best_L = -Inf
        best_as = actiontype(p.pomdp)[]
        for ba in D.children[1]
            L = D.ba_L[ba]
            if L > best_L
                best_L = L
                best_as = [D.ba_action[ba]]
            elseif L == best_L
                push!(best_as, D.ba_action[ba])
            end
        end

        return rand(p.rng, best_as)::actiontype(p.pomdp), info # best_as will usually only have one entry, but we want to break the tie randomly
    catch ex
        return default_action(p.sol.default_action, p.pomdp, b, ex)::actiontype(p.pomdp), info
    end
end

POMDPs.action(p::PMCPPlanner, b) = first(action_info(p, b))::actiontype(p.pomdp)
POMDPs.updater(p::PMCPPlanner) = SIRParticleFilter(p.pomdp, p.sol.m * 10, rng=p.rng)

function Random.seed!(p::PMCPPlanner, seed)
    Random.seed!(p.rng, seed)
    return p
end