function bounds_sanity_check(pomdp::POMDP, b::WPFBelief, L, U)
    if L > U
        @warn("L ($L) > U ($U)   |ϕ| = $(n_particles(b))")
        @info("Try e.g. `IndependentBounds(l, u, consistency_fix_thresh=1e-5)`.", maxlog=1)
    end
    if all(isterminal(pomdp, b.particles[i]) for i in 1:n_particles(b) if weight(b, i) > 0)
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
    if bds.check_terminal && all(isterminal(pomdp, b.particles[i]) for i in 1:n_particles(b) if weight(b, i) > 0)
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
    l = bound(bds.lower, pomdp, b, wdict, bds.max_depth)
    u = bound(bds.upper, pomdp, b, wdict, bds.max_depth)
    bound_dict = Dict{O, Tuple{Float64, Float64}}()
    for (o, w) in wdict
        if bds.check_terminal && all(isterminal(pomdp, b.particles[i]) for i in 1:length(w) if w[i] > 0)
            bound_dict[o] = (0.0, 0.0)
        else
            if u[o] < l[o] && u[o] >= l[o]-bds.consistency_fix_thresh
                u[o] = l[o]
            end
            if bounds_warning
                switch_to_sibling!(b, o, w)
                bounds_sanity_check(pomdp, b, l[o], u[o])
            end
            bound_dict[o] = (l[o], u[o])
        end
    end
    return bound_dict
end

struct POValue
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

struct SolvedPOValue{P<:POMDPs.Policy}
    policy::P
end

struct SemiPORollout
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

mutable struct SolvedSemiPORollout{P<:POMDPs.Policy, S, O, RNG<:AbstractRNG}
    policy::P
    inner_ind::Int
    leaf_ind::Int
    obs_ind_dict::Vector{Dict{O, Int}}
    states::Vector{Vector{S}}
    weights::Vector{Vector{Float64}}
    probs::Vector{Vector{Float64}}
    rng::RNG
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

function bound(bd::SolvedSemiPORollout, pomdp::POMDP, b::WPFBelief, max_depth::Int)
    bd.inner_ind = 0
    bd.leaf_ind = 0
    estimate_value(bd, pomdp, b, max_depth-b.depth)
end

function bound(bd::SolvedSemiPORollout, pomdp::POMDP, b::WPFBelief{S, O}, wdict::Dict{O, Array{Float64,1}}, max_depth::Int) where S where O
    bound_dict = Dict{O, Float64}()
    for (o, w) in wdict
        switch_to_sibling!(b, o, w)
        bound_dict[o] = bound(bd, pomdp, b, max_depth)
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

function convert_estimator(est::SemiPORollout, solver::AdaOPSSolver, pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    O = obstype(pomdp)
    S = statetype(pomdp)
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
    if typeof(est.updater) <: BasicParticleFilter
        return pf_simulate(sim, pomdp, est.policy, est.updater, b, start_state)
    else
        return simulate(sim, pomdp, est.policy, est.updater, b, start_state)
    end
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

function estimate_value(est::SolvedSemiPORollout, pomdp::POMDP, b::WPFBelief, steps::Integer)
    if steps <= 0 || weight_sum(b) == 0.0
        return 0.0
    end
    est.inner_ind += 1
    S = statetype(pomdp)
    O = obstype(pomdp)
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
            if hasmethod(action, (typeof(est.policy), S))
                U += probs[obs_ind] * simulate(RolloutSimulator(est.rng, steps-1), pomdp, est.policy, states[obs_ind][1])
            else
                U += probs[obs_ind] * simulate(RolloutSimulator(est.rng, steps-1), pomdp, POtoFO(est.policy), states[obs_ind][1])
            end
        else
            bp = WPFBelief(states[obs_ind], weights[obs_ind], o, depth=b.depth+1)
            U += probs[obs_ind] * estimate_value(est, pomdp, bp, steps-1)
        end
    end
    return (r_sum + discount(pomdp)*U)/weight_sum(b)
end

# For the partially observable simulation
function pf_simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::BasicParticleFilter, initial_belief, s)
    eps = sim.eps === nothing ? 0.0 : sim.eps
    max_steps = sim.max_steps === nothing ? typemax(Int) : sim.max_steps

    b = initialize_belief(updater, initial_belief)

    disc = 1.0
    r_total = 0.0
    step = 1
    while disc > eps && !isterminal(pomdp, s) && step <= max_steps
        s, o, r, b = update(updater, s, b, action(policy, b))
        r_total += disc*r
        disc *= discount(pomdp)
        step += 1
    end
    return r_total
end

function update(up::BasicParticleFilter, s, b::ParticleCollection, a)
    pm = up._particle_memory
    wm = up._weight_memory
    resize!(pm, n_particles(b))
    resize!(wm, n_particles(b)+1)
    sp, o, r = @gen(:sp, :o, :r)(up.predict_model, s, a, up.rng)
    predict!(pm, up.predict_model, b, a, o, up.rng)
    push!(b.particles, s)
    push!(pm, sp)
    reweight!(wm, up.reweight_model, b, a, pm, o, up.rng)
    bp = ParticleFilters.resample(up.resampler,
                                    WeightedParticleBelief(pm, wm, sum(wm), nothing),
                                    up.predict_model,
                                    up.reweight_model,
                                    b, a, o,
                                    up.rng)
    return sp, o, r, bp
end

struct POtoFO{P<:POMDPs.Policy} <: POMDPs.Policy
    policy::P
end

POMDPs.action(p::POtoFO, s) = action(p.policy, ParticleCollection([s]))