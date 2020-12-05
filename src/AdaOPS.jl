module AdaOPS

using POMDPs
using BeliefUpdaters
using Parameters
using CPUTime
using ParticleFilters
using D3Trees
using Random
using Printf
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using POMDPLinter
using LinearAlgebra
using Distributions

using MCTS
import MCTS: convert_estimator, convert_to_policy
using BasicPOMCP
import BasicPOMCP: SolvedFORollout, SolvedPORollout, SolvedFOValue, convert_estimator

import Random.rand

export
    AdaOPSSolver,
    AdaOPSPlanner,
    AdaOPSTree,

    WPFBelief,
    previous_obs,

    default_action,
    NoGap,

    IndependentBounds,
    bounds,
    init_bounds,
    bound,
    init_bound,


    FOValue,
    POValue,
    PORollout,
    FORollout,
    RolloutEstimator,

    StateGrid,
    KLDSampleSize,

    extra_info_analysis,
    build_tree_test

include("grid.jl")
include("Sampling.jl")

"""
    AdaOPSSolver(<keyword arguments>)

Each field may be set via keyword argument. The fields that correspond to algorithm
parameters match the definitions in the paper exactly.

# Fields
- `epsilon_0`
- `xi`
- `K`
- `D`
- `lambda`
- `T_max`
- `max_trials`
- `bounds`
- `default_action`
- `rng`
- `random_source`
- `bounds_warnings`
- `tree_in_info`

Further information can be found in the field docstrings (e.g.
`?AdaOPSSolver.xi`)
"""
@with_kw mutable struct AdaOPSSolver{R<:AbstractRNG} <: Solver
    "The target gap between the upper and the lower bound at the root of the AdaOPS tree."
    epsilon_0::Float64                      = 0.0

    "The δ-packing of beliefs will be generated."
    delta::Float64                          = 0.2

    "The target error for belief estimation."
    zeta::Float64                           = 0.1

    "The minimum relative gap required for a branch to be expanded."
    xi::Float64                             = 0.95

    "State grid for adaptive particle filters"
    grid::Union{Nothing, StateGrid}         = nothing

    "The minimum number of different tiles in the grid occupied by a belief."
    k_min::Int                              = 5

    "Return the minimum effective sample size needed for accurate estimation"
    MESS::Function                          = KLDSampleSize

    "The maximum depth of the DESPOT."
    D::Int                                  = 90

    "The maximum online planning time per step."
    T_max::Float64                          = 1.0

    "The maximum number of trials of the planner."
    max_trials::Int                         = typemax(Int)

    "A representation for the upper and lower bound on the discounted value (e.g. `IndependentBounds`)."
    bounds::Any                             = IndependentBounds(-1e6, 1e6)

    """A default action to use if algorithm fails to provide an action because of an error.
   
    This can either be an action object, i.e. `default_action=1` if `actiontype(pomdp)==Int` or a function `f(pomdp, b, ex)` where b is the belief and ex is the exception that caused the planner to fail.
    """
    default_action::Any                     = ExceptionRethrow()

    "A random number generator for the internal sampling processes."
    rng::R                                  = MersenneTwister(rand(UInt32))

    "If true, sanity checks on the provided bounds are performed."
    bounds_warnings::Bool                   = true

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPModelTools.action_info."
    tree_in_info::Bool                      = false
end

struct AdaOPSPlanner{P<:POMDP, B, RNG<:AbstractRNG} <: Policy
    sol::AdaOPSSolver
    pomdp::P
    bounds::B
    init_m::Int
    discounts::Array{Float64,1}
    rng::RNG
end

function AdaOPSPlanner(sol::AdaOPSSolver, pomdp::POMDP)
    rng = deepcopy(sol.rng)
    bounds = init_bounds(sol.bounds, pomdp, sol, rng)
    init_m = ceil(Int64, sol.MESS(sol.k_min, sol.zeta))
    discounts = discount(pomdp) .^[0:(sol.D+1);]
    return AdaOPSPlanner(deepcopy(sol), pomdp, bounds, init_m, discounts, rng)
end

include("wpf_belief.jl")
include("bounds.jl")

include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")

include("visualization.jl")
include("exceptions.jl")

include("analysis.jl")

end # module
