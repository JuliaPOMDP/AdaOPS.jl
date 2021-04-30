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
using POMDPLinter
using LinearAlgebra
using Distances
using Statistics
using StaticArrays
using Distributions
using Plots

using MCTS
import MCTS: convert_to_policy
using BasicPOMCP

export
    AdaOPSSolver,
    AdaOPSPlanner,
    AdaOPSTree,

    WPFBelief,

    default_action,

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
    zeros_like,
    access,

    info_analysis,
    hist_analysis

include("grid.jl")
include("sampling.jl")

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
@with_kw struct AdaOPSSolver{N, R<:AbstractRNG} <: Solver
    "The target gap between the upper and the lower bound at the root of the AdaOPS tree."
    epsilon_0::Float64                      = 1e-3

    "The minimum relative gap required for a branch to be expanded."
    xi::Float64                             = 0.95

    "The δ-packing of beliefs will be generated."
    delta::Float64                          = 0.1

    "State grid for adaptive particle filters"
    grid::StateGrid{N}                      = StateGrid()

    "The minimum number of bins a belief occupies (at least 2)."
    min_occupied_bins::Int                  = 2

    "The maximum number of bins a belief occupies (default to the grid size)."
    max_occupied_bins::Int                  = prod(size(grid))

    "The number of particles used for generating belief packing."
    m_min::Int                              = 30

    "The target error for belief estimation."
    zeta::Float64                           = KLDSampleSize(min_occupied_bins, m_min)

    "The maximum number of particles for belief estimation"
    m_max::Int                              = max(ceil(Int, KLDSampleSize(max_occupied_bins, zeta)), m_min)

    "Resample when the design effect of a belief node exceed Deff_thres"
    Deff_thres::Float64                     = 2.0

    "The maximum depth of the belief tree."
    max_depth::Int                          = 90

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
    bounds_warnings::Bool                   = false

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPModelTools.action_info."
    tree_in_info::Bool                      = false

    "Issue an warning when the planning time surpass the time limit by `timeout_warning_threshold` times"
    timeout_warning_threshold::Float64     = T_max * 2.0

    "Number of pre-allocated belief nodes"
    num_b::Int                              = 50_000
end

mutable struct AdaOPSTree{S,A,O}
    # belief nodes
    weights::Vector{Vector{Float64}} # stores weights for *belief node*
    children::Vector{UnitRange{Int}} # to children *ba nodes*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    u::Vector{Float64}
    l::Vector{Float64}
    obs::Vector{O}
    obs_prob::Vector{Float64}

    # action nodes
    ba_particles::Vector{Vector{S}} # stores particles for *ba nodes*
    ba_children::Vector{UnitRange{Int}}
    ba_parent::Vector{Int} # maps to parent *belief node*
    ba_u::Vector{Float64}
    ba_l::Vector{Float64}
    ba_r::Vector{Float64} # needed for backup
    ba_action::Vector{A}

    root_belief::WeightedParticleBelief
    b::Int
    ba::Int
end

mutable struct AdaOPSPlanner{S, A, O, P<:POMDP{S,A,O}, N, B, RNG<:AbstractRNG} <: Policy
    sol::AdaOPSSolver{N, RNG}
    pomdp::P
    bounds::B
    discounts::Vector{Float64}
    rng::RNG
    # The following attributes are used to avoid reallocating memory
    resampled::WeightedParticleBelief{S}
    obs::Vector{O}
    obs_ind_dict::Dict{O, Int}
    w::Vector{Vector{Float64}}
    norm_w::Vector{Vector{Float64}}
    access_cnt::Array{Int, N}
    obs_w::Vector{Float64}
    u::Vector{Float64}
    l::Vector{Float64}
    tree::Union{Nothing, AdaOPSTree{S,A,O}}
end

function AdaOPSPlanner(sol::AdaOPSSolver{N}, pomdp::POMDP{S,A,O}) where {S,A,O,N}
    rng = deepcopy(sol.rng)
    bounds = init_bounds(sol.bounds, pomdp, sol, rng)
    discounts = discount(pomdp) .^[0:(sol.max_depth+1);]
    @assert sol.m_max >= sol.m_min

    m_min = sol.m_min
    m_max = sol.m_max
    access_cnt = zeros_like(sol.grid)
    norm_w = Vector{Float64}[Vector{Float64}(undef, m_min) for i in 1:m_max]
    return AdaOPSPlanner(deepcopy(sol), pomdp, bounds, discounts, rng, 
                        WeightedParticleBelief(Vector{S}(undef, m_max), ones(m_max), m_max), sizehint!(O[], m_max),
                        Dict{O, Int}(), sizehint!(Vector{Float64}[], m_max), norm_w, access_cnt,
                        sizehint!(Float64[], m_max), sizehint!(Float64[], m_max), sizehint!(Float64[], m_max),
                        nothing)
end

solver(p::AdaOPSPlanner) = p.sol

include("wpf_belief.jl")
include("bounds.jl")
include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")
include("visualization.jl")
include("analysis.jl")

end # module
