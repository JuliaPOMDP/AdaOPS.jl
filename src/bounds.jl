function bounds_sanity_check(pomdp::P, b::WPFBelief{S}, L::Float64, U::Float64) where {S,P<:POMDP{S}}
    if L > U
        @warn(@sprintf("L (%-4.1f) > U (%-4.1f)", L, U))
        @info("Try e.g. `IndependentBounds(l, u, consistency_fix_thresh=1e-5)`.", maxlog=1)
    end
    if (L !== 0.0 || U !== 0.0) && all(isterminal(pomdp, particle(b, i)) for i in 1:n_particles(b) if weight(b, i) > 0.0)
        error(@sprintf("If all states are terminal, lower and upper bounds should be zero (L=%-4.1g, U=%-4.1g). (try IndependentBounds(l, u, check_terminal=true))", L, U))
    end
    if isinf(L) || isnan(L)
        @warn(@sprintf("L = %-4.1f. Infinite bounds are not supported.", L))
    end
    if isinf(U) || isnan(U)
        @warn(@sprintf("U = %-4.1f. Infinite bounds are not supported.", U))
    end
    return nothing
end

"""
    Dependent Bounds

Specify lower and upper bounds that are not independent.
###
It can take as input a function, `f(pomdp, belief)`, that returns both lower and upper bounds.
It can take as input any object for which a `bounds` function is implemented

"""

init_bounds(bds, pomdp, sol, rng) = bds

# Used to initialize lower and upper bounds with a single function
function bounds(f::Function, pomdp::P, b::WPFBelief{S,A,O}, max_depth::Int, bounds_warning::Bool) where {S,A,O,P<:POMDP{S,A,O}}
    l, u = f(pomdp, b)
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return l, u
end

# Used to initialize both the lower and upper bound with an object for which a `bounds` function is implemented
function bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::B, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int, bounds_warning::Bool) where {S,A,O,P<:POMDP{S,A,O},B}
    @inbounds for i in eachindex(W)
        switch_to_sibling!(b, obs[i], W[i])
        L[i], U[i] = bounds(bd, pomdp, b, max_depth, bounds_warning)
    end
    return L, U
end


"""
    IndependentBounds(lower, upper, check_terminal=false, consistency_fix_thresh=0.0)

Specify lower and upper bounds that are independent of each other (the most common case).
A lower or upper bound can be a Number, a Function, `f(pomdp, belief)`, that returns a bound, an object for which a `bound` function 
is implemented. Specifically, for FOValue, POValue, FORollout, SemiPORollout, and PORollout, a `bound` function is already implemented.
You can also implement a `bound!` function to initialize sibling beliefs simultaneously, if it could provide a further performace gain.

# Keyword Arguments
- `check_terminal::Bool=false`: if true, then if all the states in the belief are terminal, the upper and lower bounds will be overridden and set to 0.
- `consistency_fix_thresh::Float64=0.0`: if `upper < lower` and `upper >= lower-consistency_fix_thresh`, then `upper` will be bumped up to `lower`.
"""

mutable struct IndependentBounds{L, U}
    lower::L
    upper::U
    check_terminal::Bool
    consistency_fix_thresh::Float64
end

function IndependentBounds(l, u;
                           check_terminal=false,
                           consistency_fix_thresh=0.0)
    return IndependentBounds(l, u, check_terminal, consistency_fix_thresh)
end

function init_bounds(bds::IndependentBounds, pomdp::POMDP, sol::AdaOPSSolver, rng::AbstractRNG)
    return IndependentBounds(convert_estimator(bds.lower, sol, pomdp),
                             convert_estimator(bds.upper, sol, pomdp),
                             bds.check_terminal,
                             bds.consistency_fix_thresh,
                            )
end

function bounds(bds::IndependentBounds, pomdp::P, b::WPFBelief{S,A,O}, max_depth::Int, bounds_warning::Bool) where {S,A,O,P<:POMDP{S,A,O}}
    if bds.check_terminal && all(isterminal(pomdp, particle(b, i)) for i in 1:n_particles(b) if weight(b, i) > 0.0)
        return (0.0, 0.0)
    end
    l = bound(bds.lower, pomdp, b, max_depth)
    u = bound(bds.upper, pomdp, b, max_depth)
    if u < l && u >= l-bds.consistency_fix_thresh
        u = l
    end
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return l, u
end

function bounds!(L::Vector{Float64}, U::Vector{Float64}, bds::IndependentBounds{LB,UB}, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int, bounds_warning::Bool) where {LB,UB,S,A,O,P<:POMDP{S,A,O}}
    bound!(L, bds.lower, pomdp, b, W, obs, max_depth)
    bound!(U, bds.upper, pomdp, b, W, obs, max_depth)
    if bds.check_terminal
        @inbounds for i in eachindex(W)
            w = W[i]
            if (L[i] !== 0.0 || U[i] !== 0.0) && all(isterminal(pomdp, particle(b, j)) for j in 1:n_particles(b) if w[j] > 0.0)
                L[i] = 0.0
                U[i] = 0.0
            end
        end
    end
    @inbounds for i in eachindex(W)
        if U[i] < L[i] && U[i] >= L[i]-bds.consistency_fix_thresh
            U[i] = L[i]
        end
    end
    if bounds_warning
        @inbounds for i in eachindex(W)
            switch_to_sibling!(b, obs[i], W[i])
            bounds_sanity_check(pomdp, b, L[i], U[i])
        end
    end
    return L, U
end

# Used when the lower or upper bound is a fixed number
bound(n::N, pomdp, b, max_depth) where N<:Real = convert(Float64, n)
bound!(V::Vector{Float64}, n::Float64, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int) where {S,A,O,P<:POMDP{S,A,O}} = fill!(V, n)
bound!(V::Vector{Float64}, n::Int, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int) where {S,A,O,P<:POMDP{S,A,O}} = fill!(V, convert(Float64, n))

# Used when lower or upper bound is a function
bound(f::Function, pomdp::P, b::WPFBelief{S,A,O}, max_depth::Int) where {S,A,O,P<:POMDP{S,A,O}} = f(pomdp, b)

# Used when the lower or upper bound is an object for which a `bound` function is implemented
function bound!(V::Vector{Float64}, bd::B, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int) where {S,A,O,P<:POMDP{S,A,O},B}
    @inbounds for i in eachindex(W)
        switch_to_sibling!(b, obs[i], W[i])
        V[i] = bound(bd, pomdp, b, max_depth)
    end
    return V
end

struct SolvedFOValue{P<:Policy}
    policy::P
    values::Vector{Float64}
end

struct SolvedFORollout{P<:Policy, RNG<:AbstractRNG}
    policy::P
    values::Vector{Float64}
    rng::RNG
end

struct SolvedPORollout{P<:Policy, U<:Updater, RNG<:AbstractRNG}
    policy::P
    values::Vector{Float64}
    updater::U
    rng::RNG
end

struct POValue
    solver::Union{Solver, Policy}
end

struct SolvedPOValue{P<:Policy}
    policy::P
end

struct SemiPORollout
    solver::Union{Solver, Policy}
end

mutable struct SolvedSemiPORollout{S, O, P<:Policy, RNG<:AbstractRNG}
    policy::P
    inner_ind::Int
    leaf_ind::Int
    obs_ind_dict::Vector{Dict{O, Int}}
    states::Vector{Vector{S}}
    weights::Vector{Vector{Float64}}
    probs::Vector{Vector{Float64}}
    rng::RNG
end

function bound(bd::SolvedFORollout, pomdp::POMDP, b::WPFBelief{S}, max_depth::Int) where S
    resize!(bd.values, n_particles(b))
    broadcast!((s)->estimate_value(bd, pomdp, s, b, max_depth-b.depth), bd.values, particles(b))
    return dot(weights(b), bd.values)/weight_sum(b)
end

function bound!(V::Vector{Float64}, bd::SolvedFORollout, pomdp::POMDP, b::WPFBelief{S}, W::Vector{Vector{Float64}}, obs::Vector, max_depth::Int) where S
    resize!(bd.values, n_particles(b))
    broadcast!((s)->estimate_value(bd, pomdp, s, b, max_depth-b.depth), bd.values, particles(b))
    @inbounds for i in eachindex(W)
        V[i] = dot(bd.values, W[i]) / sum(W[i])
    end
    return V
end

function bound(bd::SolvedPORollout{P}, pomdp::M, b::WPFBelief{S,A,O}, max_depth::Int) where {P,S,A,O,M<:POMDP{S,A,O}}
    resize!(bd.values, n_particles(b))
    broadcast!((s)->estimate_value(bd, pomdp, s, b, max_depth-b.depth), bd.values, particles(b))
    return dot(weights(b), bd.values)/weight_sum(b)
end

function bound(bd::SolvedSemiPORollout{S,O,P}, pomdp::M, b::WPFBelief{S,A,O}, max_depth::Int) where {P,S,A,O,M<:POMDP{S,A,O}}
    bd.inner_ind = 0
    bd.leaf_ind = 0
    return estimate_value(bd, pomdp, b, max_depth-b.depth)
end

function bound(bd::SolvedFOValue{P}, pomdp::M, b::WPFBelief{S}, max_depth::Int) where {P,S,M<:POMDP{S}}
    resize!(bd.values, n_particles(b))
    broadcast!((s)->value(bd.policy, s), bd.values, particles(b))
    return dot(weights(b), bd.values)/weight_sum(b)
end

function bound!(V::Vector{Float64}, bd::SolvedFOValue{P}, pomdp::M, b::WPFBelief{S}, W::Vector{Vector{Float64}}, obs::Vector, max_depth::Int) where {P,S,M<:POMDP{S}}
    resize!(bd.values, n_particles(b))
    broadcast!((s)->value(bd.policy, s), bd.values, particles(b))
    @inbounds for i in eachindex(W)
        V[i] = dot(bd.values, W[i]) / sum(W[i])
    end
    return V
end

bound(bd::SolvedPOValue, pomdp::POMDP, b::WPFBelief, max_depth::Int) = value(bd.policy, b)

# Convert an unsolved estimator to solved estimator
convert_estimator(ev, solver, mdp) = ev

function convert_estimator(est::FOValue, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, UnderlyingMDP(pomdp))
    m_max = ceil(Int, solver.sigma * solver.m_init)
    SolvedFOValue(policy, sizehint!(Float64[], m_max))
end

function convert_estimator(est::FORollout, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, UnderlyingMDP(pomdp))
    m_max = ceil(Int, solver.sigma * solver.m_init)
    SolvedFORollout(policy, sizehint!(Float64[], m_max), solver.rng)
end

function convert_estimator(est::POValue, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedPOValue(policy)
end

function convert_estimator(est::PORollout, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    m_max = ceil(Int, solver.sigma * solver.m_init)
    SolvedPORollout(policy, sizehint!(Float64[], m_max), est.updater, solver.rng)
end

function convert_estimator(est::RolloutEstimator, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    m_max = ceil(Int, solver.sigma * solver.m_init)
    SolvedPORollout(policy, sizehint!(Float64[], m_max), updater(policy), solver.rng)
end

function convert_estimator(est::SemiPORollout, solver::AdaOPSSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    # this can be further optimized by memory preallocation - To Do
    SolvedSemiPORollout(policy, 0, 0, Dict{O, Int}[], Vector{S}[], Vector{Float64}[], Vector{Float64}[], solver.rng)
end

# Estimate the value of state with estimator
function estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    rollout(estimator, pomdp, start_state, b, steps)
end

POMDPLinter.@POMDP_require estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    @subreq rollout(estimator, pomdp, start_state, b, steps)
end

# Perform a rollout simulation to estimate the value.
function rollout(est::SolvedPORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    sim = RolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

POMDPLinter.@POMDP_require rollout(est::SolvedPORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    sim = RolloutSimulator(est.rng, steps)
    @subreq simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

function rollout(est::SolvedFORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    sim = RolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, start_state)
end

POMDPLinter.@POMDP_require rollout(est::SolvedFORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    sim = RolloutSimulator(est.rng, steps)
    @subreq simulate(sim, pomdp, est.policy, start_state)
end

function estimate_value(est::SolvedSemiPORollout{S,O,P}, pomdp::M, b::WPFBelief{S,A,O}, steps::Int) where {P,S,A,O,M<:POMDP{S,A,O}}
    if steps <= 0 || weight_sum(b) == 0.0
        return 0.0
    end
    est.inner_ind += 1
    if length(est.probs) < est.inner_ind
        push!(est.obs_ind_dict, Dict{O, Int}())
        push!(est.probs, Float64[])
    end
    obs_ind_dict = est.obs_ind_dict[est.inner_ind]
    probs = est.probs[est.inner_ind]
    empty!(obs_ind_dict)
    empty!(probs)

    states = Vector{S}[]
    weights = Vector{Float64}[]

    a = action(est.policy, b)

    r_sum = 0.0
    for (k, s) in enumerate(particles(b))
        if !isterminal(pomdp, s)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, est.rng)

            if !haskey(obs_ind_dict, o)
                push!(probs, 0.0)
                obs_ind_dict[o] = length(probs)
                est.leaf_ind += 1
                if length(est.states) < est.leaf_ind
                    push!(est.states, Array{S,1}[])
                    push!(est.weights, Array{Float64,1}[])
                else
                    empty!(est.states[est.leaf_ind])
                    empty!(est.weights[est.leaf_ind])
                end
                push!(states, est.states[est.leaf_ind])
                push!(weights, est.weights[est.leaf_ind])
            end
            obs_ind = obs_ind_dict[o]
            push!(states[obs_ind], sp)
            push!(weights[obs_ind], weight(b, k) * obs_weight(pomdp, s, a, sp, o))
            probs[obs_ind] += weight(b, k)

            r_sum += r * weight(b, k)
        end
    end

    U = 0.0
    for (o, obs_ind) in obs_ind_dict
        if length(states[obs_ind]) == 1
            U += probs[obs_ind] * simulate(RolloutSimulator(est.rng, steps-1), pomdp, est.policy, states[obs_ind][1])
        else
            bp = WPFBelief(states[obs_ind], weights[obs_ind], 1, b.depth+1, b.tree, o)
            U += probs[obs_ind] * estimate_value(est, pomdp, bp, steps-1)
        end
    end
    return (r_sum + discount(pomdp)*U)/weight_sum(b)
end

POMDPLinter.@POMDP_require estimate_value(est::SolvedSemiPORollout, pomdp::POMDP, b::WPFBelief, steps::Int) begin
    @subreq action(est.policy, rand(b))
end