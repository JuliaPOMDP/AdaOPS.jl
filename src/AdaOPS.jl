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
    FORollout,
    PORollout,
    SemiPORollout,
    RolloutEstimator,

    StateGrid,
    KLDSampleSize,

    extra_info_analysis,
    build_tree_test

include("grid.jl")
include("Sampling.jl")
include("wpf_belief.jl")

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

    "The initial number of particles at root. (no need to adjust)"
    m_init::Int                             = 50

    "Whether to use effective sampling size"
    ESS::Bool                               = true

    "At least m_min times of MESS(k_parent, zeta) particles are needed for estimating a belief."
    m_min::Float64                          = 0.6

    "At most m_max times of m_init particles are allowed for estimating a belief."
    m_max::Float64                          = 10.0

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

mutable struct AdaOPSTree{S,A,O}
    weights::Vector{Vector{Float64}} # stores weights for *belief node*
    children::Vector{Vector{Int}} # to children *ba nodes*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    k::Vector{Int}
    u::Vector{Float64}
    l::Vector{Float64}
    obs::Vector{O}
    obs_prob::Vector{Float64}

    ba_particles::Vector{Vector{S}} # stores particles for *ba nodes*
    ba_children::Vector{Vector{Int}}
    ba_parent::Vector{Int} # maps to parent *belief node*
    ba_u::Vector{Float64}
    ba_l::Vector{Float64}
    ba_r::Vector{Float64} # needed for backup
    ba_action::Vector{A}

    root_belief::Union{WPFBelief, Missing}
    b_len::Int
    ba_len::Int
end

struct AdaOPSPlanner{P<:POMDP, B, RNG<:AbstractRNG, S, O} <: Policy
    sol::AdaOPSSolver
    pomdp::P
    bounds::B
    discounts::Array{Float64,1}
    rng::RNG
    # The following attributes are used to avoid reallocating memory
    tree::AdaOPSTree
    all_states::Array{S, 1}
    state_ind_dict::Dict{S, Int}
    wdict::Dict{O, Array{Float64, 1}}
    obs_ind_dict::Dict{O, Int}
    freqs::Array{Float64, 1}
    likelihood_sums::Array{Float64, 1}
    likelihood_square_sums::Array{Float64, 1}
    access_cnts::Union{Array, Nothing}
    ks::Union{Array{Int, 1}, Nothing}
end

function AdaOPSPlanner(sol::AdaOPSSolver, pomdp::POMDP)
    S = statetype(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)

    rng = deepcopy(sol.rng)
    bounds = init_bounds(sol.bounds, pomdp, sol, rng)
    discounts = discount(pomdp) .^[0:(sol.D+1);]

    tree = AdaOPSTree{S,A,O}([Float64[]],
                         [Int[]],
                         [0],
                         [0],
                         [0],
                         Vector{Float64}(undef, 1),
                         Vector{Float64}(undef, 1),
                         Vector{O}(undef, 1),
                         [1.0],

                         Array{S,1}[],
                         Array{Int,1}[],
                         Int[],
                         Float64[],
                         Float64[],
                         Float64[],
                         A[],

                         missing,
                         1,
                         0
                 )

    if sol.grid !== nothing
        access_cnts = Array{Int, length(sol.grid)}[] # store the access_count grid for each observation branch
        ks = Int[] # track the dispersion of child beliefs
    else
        access_cnts = nothing
        ks = nothing
    end
    return AdaOPSPlanner(deepcopy(sol), pomdp, bounds, discounts, rng,
                        tree, S[], Dict{S, Int}(), Dict{O, Array{Float64,1}}(),
                        Dict{O, Int}(), Float64[], Float64[], Float64[], access_cnts, ks)
end

include("bounds.jl")
include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")
include("visualization.jl")
include("exceptions.jl")
include("analysis.jl")

end # module
