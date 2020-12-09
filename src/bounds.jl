function bounds_sanity_check(pomdp::POMDP, b::WPFBelief, L, U)
    if L > U
        @warn("L ($L) > U ($U)   |ϕ| = $(n_particles(b))")
        @info("Try e.g. `IndependentBounds(l, u, consistency_fix_thresh=1e-5)`.", maxlog=1)
    end
    if all(isterminal(pomdp, s) for s in particles(b))
        if L != 0.0 || U != 0.0
            error(@sprintf("If all states are terminal, lower and upper bounds should be zero (L=%-10.2g, U=%-10.2g). (try IndependentBounds(l, u, check_terminal=true))", L, U))
        end
    end
    if isinf(L) || isnan(L)
        @warn("L = $L. Infinite bounds are not supported.")
    end
    if isinf(U) || isnan(U)
        @warn("U = $U. Infinite bounds are not supported.")
    end
end

init_bound(bd, pomdp, sol, rng) = bd
init_bounds(bds, pomdp, sol, rng) = bds
init_bounds(t::Tuple, pomdp, sol, rng) = (init_bound(first(t), pomdp, sol, rng), init_bound(last(t), pomdp, sol, rng))

# Used when the lower or upper bound is a fixed number
bound(n::Number, pomdp::POMDP, b::WPFBelief, max_depth::Int) = convert(Float64, n)

function bound(n::Number, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    u = bound(n, pomdp, b, max_depth)
    Dict([o=>u for (o,w) in wdict])
end

# Used when both lower and upper bounds are fixed numbers
function bounds(t::Tuple, pomdp::POMDP, b::WPFBelief, bounds_warning::Bool = true)
    l, u = bound(t[1], pomdp, b, ceil(Int, 5/(1-discount(pomdp)))), bound(t[2], pomdp, b, ceil(Int, 5/(1-discount(pomdp))))
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return (l, u)
end

function bounds(t::Tuple, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, bounds_warning::Bool = true) where {S, O}
    bound_dict = Dict{O, Tuple{Float64, Float64}}()
    l, u = bounds(t, pomdp, b, bounds_warning)
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    for (o, w) in wdict
        bound_dict[o] = (l, u)
    end
    return bound_dict
end

# Used when the lower or upper bound is an object for which a `bound` function is implemented
function bound(bd, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        bound_dict[o] = bound(bd, pomdp, b, max_depth)
    end
    return bound_dict
end

# Used when lower or upper bound is a function
bound(f::Function, pomdp::POMDP, b::WPFBelief, max_depth::Int) = f(pomdp, b)

function bound(f::Function, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        bound_dict[o] = bound(f, pomdp, b, max_depth)
    end
    return bound_dict
end

# Used to initialize lower and upper bounds with a single function
function bounds(f::Function, pomdp::POMDP, b::WPFBelief, bounds_warning::Bool = true)
    l, u = f(pomdp, b)
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return (l, u)
end

function bounds(f::Function, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, bounds_warning::Bool) where S where O
    bound_dict = Dict{O, Tuple{Float64, Float64}}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        l, u = bounds(f, pomdp, b, bounds_warning)
        if bounds_warning
            bounds_sanity_check(pomdp, b, l, u)
        end
        bound_dict[o] = (l, u)
    end
    return bound_dict
end


"""
    IndependentBounds(lower, upper, check_terminal=false, consistency_fix_thresh=0.0)

Specify lower and upper bounds that are independent of each other (the most common case).

# Keyword Arguments
- `check_terminal::Bool=false`: if true, then if all the states in the belief are terminal, the upper and lower bounds will be overridden and set to 0.
- `consistency_fix_thresh::Float64=0.0`: if `upper < lower` and `upper >= lower-consistency_fix_thresh`, then `upper` will be bumped up to `lower`.
"""

mutable struct IndependentBounds{L, U}
    lower::L
    upper::U
    check_terminal::Bool
    consistency_fix_thresh::Float64
    max_depth::Union{Nothing, Int}
end

function IndependentBounds(l, u;
                           check_terminal=false,
                           consistency_fix_thresh=0.0,
                           max_depth=nothing)
    return IndependentBounds(l, u, check_terminal, consistency_fix_thresh, max_depth)
end

function init_bounds(bds::IndependentBounds, pomdp::POMDP, sol::AdaOPSSolver, rng::R) where R <: AbstractRNG
    return IndependentBounds(convert_estimator(bds.lower, sol, pomdp),
                             convert_estimator(bds.upper, sol, pomdp),
                             bds.check_terminal,
                             bds.consistency_fix_thresh,
                             something(bds.max_depth, sol.D)
                            )
end

function bounds(bds::IndependentBounds, pomdp::POMDP, b::WPFBelief, bounds_warning::Bool = true)
    if bds.check_terminal && all(isterminal(pomdp, s) for s in particles(b))
        return (0.0, 0.0)
    end
    l = bound(bds.lower, pomdp, b, bds.max_depth)
    u = bound(bds.upper, pomdp, b, bds.max_depth)
    if u < l && u >= l-bds.consistency_fix_thresh
        u = l
    end
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return (l,u)
end

function bounds(bds::IndependentBounds, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}, bounds_warning::Bool = true) where O
    if bds.check_terminal && all(isterminal(pomdp, s) for s in b.particles)
        return bounds((0.0, 0.0), pomdp, b, wdict, bounds_warning)
    end
    l = bound(bds.lower, pomdp, b, wdict, bds.max_depth)
    u = bound(bds.upper, pomdp, b, wdict, bds.max_depth)
    bound_dict = Dict{O, Tuple{Float64, Float64}}()
    for (o, w) in wdict
        if u[o] < l[o] && u[o] >= l[o]-bds.consistency_fix_thresh
            u[o] = l[o]
        end
        if bounds_warning
            bounds_sanity_check(pomdp, b, l[o], u[o])
        end
        bound_dict[o] = (l[o], u[o])
    end
    return bound_dict
end

struct POValue
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

struct SolvedPOValue{P<:POMDPs.Policy}
    policy::P
end

function bound(bd::Union{SolvedFORollout, SolvedPORollout}, pomdp::POMDP, b::WPFBelief, max_depth::Int)
    values = [estimate_value(bd, pomdp, s, b, max_depth-b.depth) for s in particles(b)]
    return dot(b.weights, values)/b.weight_sum
end

function bound(bd::SolvedFORollout, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    values = [estimate_value(bd, pomdp, s, b, max_depth-b.depth) for s in particles(b)]
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        bound_dict[o] = dot(w, values)/sum(w)
    end
    return bound_dict
end

function bound(bd::SolvedPORollout, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        values = [estimate_value(bd, pomdp, s, b, max_depth-b.depth) for s in particles(b)]
        bound_dict[o] = dot(w, values)/sum(w)
    end
    return bound_dict
end

function bound(bd::SolvedFOValue, pomdp::POMDP, b::WPFBelief, max_depth::Int)
    values = [value(bd.policy, s) for s in particles(b)]
    return dot(b.weights, values)/b.weight_sum
end

function bound(bd::SolvedFOValue, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where O
    values = [value(bd.policy, s) for s in particles(b)]
    return Dict([o=>(dot(w, values)/sum(w)) for (o, w) in wdict])
end

bound(bd::SolvedPOValue, pomdp::POMDP, b::WPFBelief, max_depth::Int) = value(bd.policy, b)

function bound(bd::SolvedPOValue, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        bound_dict[o] = value(bd.policy, b)
    end
    return bound_dict
end

# Convert an unsolved estimator to solved estimator
function convert_estimator(est::FOValue, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, UnderlyingMDP(pomdp))
    SolvedFOValue(policy)
end

function convert_estimator(est::POValue, solver::AdaOPSSolver, pomdp::POMDP)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedPOValue(policy)
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
    b = extract_belief(est.updater, b)
    sim = RolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

POMDPLinter.@POMDP_require rollout(est::SolvedPORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    @req extract_belief(::typeof(est.updater), ::typeof(b))
    b = extract_belief(est.updater, b)
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