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

init_bound(bound, pomdp, sol, rng) = bound
init_bounds(bounds, pomdp, sol, rng) = bounds
init_bounds(t::Tuple, pomdp, sol, rng) = (init_bound(first(t), pomdp, sol, rng), init_bound(last(t), pomdp, sol, rng))

# Used when lower or upper bound is a fixed number
ubound(n::Number, pomdp, b) = convert(Float64, n)
lbound(n::Number, pomdp, b) = convert(Float64, n)

function ubound(n::Number, pomdp, b, wdict)
    u = ubound(n, pomdp, b)
    Dict([o=>u for (o,w) in wdict])
end
function lbound(n::Number, pomdp, b, wdict)
    l = lbound(n, pomdp, b)
    Dict([o=>l for (o,w) in wdict])
end

# Used when both lower and upper are fixed numbers
function bounds(t::Tuple, pomdp::POMDP, b::WPFBelief, bounds_warning::Bool = true)
    l, u = lbound(t[1], pomdp, b), ubound(t[2], pomdp, b)
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return (l, u)
end

function bounds(t::Tuple, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, bounds_warning::Bool = true) where S where O
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

# Used when lower or upper bound is a function
ubound(f::Function, pomdp, b) = f(pomdp, b)
lbound(f::Function, pomdp, b) = f(pomdp, b)

function ubound(f::Function, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        reweight!(b, w)
        bound_dict[o] = ubound(f, pomdp, b)
    end
    return bound_dict
end
function lbound(f::Function, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        reweight!(b, w)
        bound_dict[o] = lbound(f, pomdp, b)
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
        reweight!(b, w)
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

struct IndependentBounds{L, U}
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

function init_bounds(bounds::IndependentBounds, pomdp::POMDP, sol::OPSSolver, rng::R) where R <: AbstractRNG
    return IndependentBounds(init_bound(bounds.lower, pomdp, sol, rng),
                             init_bound(bounds.upper, pomdp, sol, rng),
                             bounds.check_terminal,
                             bounds.consistency_fix_thresh
                            )
end

function bounds(bounds::IndependentBounds, pomdp::POMDP, b::WPFBelief, bounds_warning::Bool = true)
    if bounds.check_terminal && all(isterminal(pomdp, s) for s in particles(b))
        return (0.0, 0.0)
    end
    l = lbound(bounds.lower, pomdp, b)
    u = ubound(bounds.upper, pomdp, b)
    if u < l && u >= l-bounds.consistency_fix_thresh
        u = l
    end
    if bounds_warning
        bounds_sanity_check(pomdp, b, l, u)
    end
    return (l,u)
end

function bounds(bounds::IndependentBounds, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}, bounds_warning::Bool = true) where O
    if bounds.check_terminal && all(isterminal(pomdp, s) for s in b.particles)
        return (0.0, 0.0)
    end
    l = lbound(bounds.lower, pomdp, b, wdict)
    u = ubound(bounds.upper, pomdp, b, wdict)
    bound_dict = Dict{O, Tuple{Float64, Float64}}()
    for (o, w) in wdict
        if u[o] < l[o] && u[o] >= l[o]-bounds.consistency_fix_thresh
            u[o] = l[o]
        end
        if bounds_warning
            bounds_sanity_check(pomdp, b, l[o], u[o])
        end
        bound_dict[o] = (l[o], u[o])
    end
    return bound_dict
end

# Value if Fully Observed Under a Policy
struct FOValueBound{P<:Union{Solver, Policy}}
    p::P
end

function init_bound(bd::FOValueBound{S}, pomdp::POMDP, sol::OPSSolver, rng::R) where S <: Solver where R <: AbstractRNG
    return FOValueBound(solve(bd.p, pomdp))
end

function ubound(ub::FOValueBound, pomdp::POMDP, b::WPFBelief)
    values = [value(ub.p, s) for s in particles(b)]
    return dot(b.weights, values)/b.weight_sum
end
function lbound(lb::FOValueBound, pomdp::POMDP, b::WPFBelief)
    values = [value(lb.p, s) for s in particles(b)]
    return dot(b.weights, values)/b.weight_sum
end

function ubound(ub::FOValueBound, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}) where O
    values = [value(ub.p, s) for s in particles(b)]
    return Dict([o=>(dot(w, values)/sum(w)) for (o, w) in wdict])
end
function lbound(lb::FOValueBound, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}) where O
    values = [value(lb.p, s) for s in particles(b)]
    return Dict([o=>(dot(w, values)/sum(w)) for (o, w) in wdict])
end

# Value if Partially Observed Under a Policy
struct POValueBound{P<:Union{Solver, Policy}}
    p::P
end

function init_bound(bd::POValueBound{S}, pomdp::POMDP, sol::OPSSolver, rng::R) where S <: Solver where R <: AbstractRNG
    return POValueBound(solve(bd.p, pomdp))
end

ubound(ub::POValueBound, pomdp::POMDP, b::WPFBelief) = value(ub.p, b)
lbound(lb::POValueBound, pomdp::POMDP, b::WPFBelief) = value(lb.p, b)

function ubound(ub::POValueBound, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}) where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        reweight!(b, w)
        bound_dict[o] = value(ub.p, b)
    end
    return bound_dict
end

function lbound(lb::POValueBound, pomdp::POMDP, b::WPFBelief, wdict::Dict{O, Array{Float64,1}}) where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        reweight!(b, w)
        bound_dict[o] = value(lb.p, b)
    end
    return bound_dict
end

# Default Policy Lower Bound

"""
    RolloutLB(policy; max_depth=nothing)
    RolloutLB(solver; max_depth=nothing)

A lower bound calculated by running a default policy on the scenarios in a belief.

# Keyword Arguments
- `rollout_estimator::Any`: used to estimate the value of belief
- `max_depth::Union{Nothing,Int}=nothing`: max depth to run the simulation. The depth of the belief will be automatically subtracted so simulations for the bound will be run for `max_depth-b.depth` steps. If `nothing`, the solver's max depth will be used.
"""

mutable struct RolloutLB{T}
    rollout_estimator::T
    max_depth::Union{Int, Nothing}
end

function RolloutLB(rollout_estimator::Any; max_depth=nothing)
    return RolloutLB(rollout_estimator, max_depth)
end

function init_bound(lb::RolloutLB, pomdp::POMDP, sol::OPSSolver, rng::R) where R <: AbstractRNG
    if typeof(lb.rollout_estimator) <: PORollout
        if typeof(lb.rollout_estimator.updater) <: BasicParticleFilter
            lb.rollout_estimator = PORollout(lb.rollout_estimator.solver,
                                             BasicParticleFilter(pomdp,
                                                                lb.rollout_estimator.updater.resampler,
                                                                lb.rollout_estimator.updater.n_init,
                                                                lb.rollout_estimator.updater.rng))
        end
    end
    solved_estimator = convert_estimator(lb.rollout_estimator, pomdp, rng)
    max_depth = something(lb.max_depth, sol.D)
    return RolloutLB(solved_estimator, max_depth)
end

function lbound(lb::RolloutLB, pomdp::POMDP, b::WPFBelief)
    values = [estimate_value(lb.rollout_estimator, pomdp, s, b, lb.max_depth-b.depth) for s in particles(b)]
    return dot(b.weights, values)/b.weight_sum
end

function lbound(lb::RolloutLB{R}, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}) where S where O where R <: SolvedFORollout
    values = [estimate_value(lb.rollout_estimator, pomdp, s, b, lb.max_depth-b.depth) for s in particles(b)]
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        bound_dict[o] = dot(w, values)/sum(w)
    end
    return bound_dict
end

function lbound(lb::RolloutLB{R}, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}) where S where O where R <: SolvedPORollout
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        values = [estimate_value(lb.rollout_estimator, pomdp, s, b, lb.max_depth-b.depth) for s in particles(b)]
        bound_dict[o] = dot(w, values)/sum(w)
    end
    return bound_dict
end

# Convert an unsolved estimator to solved estimator
function convert_estimator(ev::RolloutEstimator, pomdp::POMDP, rng::R) where R <: AbstractRNG
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, updater(policy), rng)
end

function convert_estimator(ev::PORollout, pomdp::POMDP, rng::R) where R <: AbstractRNG
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, ev.updater, rng)
end

function convert_estimator(est::FORollout, pomdp::POMDP, rng::R) where R <: AbstractRNG
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedFORollout(policy, rng)
end

# Estimate the value of state with rollout_estimator
function estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    rollout(estimator, pomdp, start_state, b, steps)
end

@POMDP_require estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    @subreq rollout(estimator, pomdp, start_state, b, steps)
end

# Perform a rollout simulation to estimate the value.
function rollout(est::SolvedPORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    b = extract_belief(est.updater, b)
    sim = RolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

@POMDP_require rollout(est::SolvedPORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    @req extract_belief(::typeof(est.updater), ::typeof(b))
    b = extract_belief(est.updater, b)
    sim = RolloutSimulator(est.rng, steps)
    @subreq simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

function rollout(est::SolvedFORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int)
    sim = RolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, start_state)
end

@POMDP_require rollout(est::SolvedFORollout, pomdp::POMDP, start_state, b::WPFBelief, steps::Int) begin
    sim = RolloutSimulator(est.rng, steps)
    @subreq simulate(sim, pomdp, est.policy, start_state)
end

# Extract Specific type of belief from WPFBelief
extract_belief(::NothingUpdater, b::WPFBelief) = nothing

function extract_belief(::PreviousObservationUpdater, b::WPFBelief)
    b._obs
end

function extract_belief(up::KMarkovUpdater, b::WPFBelief)
    hist = history(b)
    if length(hist) > up.k
        [tuple[:o] for tuple in hist[end-up.k+1:end]]
    else
        [tuple[:o] for tuple in hist]
    end
end