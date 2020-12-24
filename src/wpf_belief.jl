mutable struct WPFBelief{S, O} <: AbstractParticleBelief{S}
    particles::Vector{S}
    weights::Vector{Float64}
    weight_sum::Float64
    belief::Union{Int, Nothing}
    depth::Int
    tree::Any
    _obs::O
    _probs::Union{Nothing, Dict{S,Float64}}
    _hist::Union{Nothing, Array{NamedTuple,1}}
end
WPFBelief(particles::Array, weights::Array{Float64,1}, weight_sum::Number, belief::Union{Int,Nothing}, depth::Int, tree, obs) = WPFBelief(particles, weights, convert(Float64, weight_sum), belief, depth, tree, obs, nothing, nothing)
WPFBelief(particles::Array, weights::Array{Float64,1}, belief::Union{Int,Nothing}, depth::Int, tree, obs) = WPFBelief(particles, weights, sum(weights), belief, depth, tree, obs, nothing, nothing)
WPFBelief(particles::Array, weights::Array{Float64,1}, weight_sum::Number, belief::Union{Int,Nothing}, depth::Int) = WPFBelief(particles, weights, weight_sum, belief, depth, nothing, nothing) 
WPFBelief(particles::Array, weights::Array{Float64,1}, belief::Union{Int,Nothing}, depth::Int) = WPFBelief(particles, weights, belief, depth, nothing, nothing) 
WPFBelief(particles::Array, weights::Array{Float64,1}, obs; depth::Int = 0) = WPFBelief(particles, weights, nothing, depth, nothing, obs)

ParticleFilters.rand(rng::R, b::WPFBelief) where R<:AbstractRNG = rand(rng, b.particles)

ParticleFilters.particles(b::WPFBelief) = b.particles
ParticleFilters.n_particles(b::WPFBelief) = length(b.particles)
ParticleFilters.weight(b::WPFBelief, i::Int) = b.weights[i]
ParticleFilters.particle(b::WPFBelief, i::Int) = b.particles[i]
ParticleFilters.weight_sum(b::WPFBelief) = b.weight_sum
ParticleFilters.weights(b::WPFBelief) = b.weights
ParticleFilters.weighted_particles(b::WPFBelief) = (b.particles[i]=>b.weights[i] for i in 1:length(b.weights))
POMDPs.mean(b::WPFBelief) = dot(b.weights, b.particles)/weight_sum(b)
POMDPs.currentobs(b::WPFBelief) = b._obs
@deprecate previous_obs POMDPs.currentobs

function POMDPs.history(belief::WPFBelief)
    if belief._hist === nothing
        belief._hist = Vector{NamedTuple}(undef, belief.depth+1)
        if belief.tree !== nothing && belief.belief != 1
            tree = belief.tree
            b = belief.belief
            depth = belief.depth + 1
            while depth != 1
                ba = tree.parent[b]
                belief._hist[depth] = (o=tree.obs[b], a=tree.ba_action[ba])
                b = tree.parent_b[ba]
                depth -= 1
            end
        end
        belief._hist[1] = (o=belief._obs,)
    end
    return belief._hist
end

function resample!(particle_set::Vector{S}, b::WPFBelief{S}, rng::AbstractRNG) where S
    m = length(particle_set)
    step = weight_sum(b)/m
    U = rand(rng)*step
    c = weight(b,1) # accumulate sum of weights
    i = 1
    for particle in 1:m
        while U > c
            i += 1
            c += weight(b, i)
        end
        U += step
        particle_set[particle] = particles(b)[i]
    end
    return nothing
end

function resample!(particle_set::Vector{S}, b, rng::AbstractRNG) where S
    for i in 1:length(particle_set)
        particle_set[i] = rand(rng, b)
    end
    return nothing
end

function switch_to_sibling!(b::WPFBelief, obs, weights::Array{Float64,1})
    b.weights = weights
    b.weight_sum = sum(weights)
    b._obs = obs
    b._probs = nothing
    if b._hist !== nothing
        if length(b._hist) > 1
            b._hist[end] = (a=b._hist[end].a, o=obs)
        else
            b._hist[end] = (o=obs,)
        end
    end
end

function ParticleFilters.probdict(b::WPFBelief{S, O}) where S where O
    if b._probs === nothing
        # update the cache
        probs = Dict{S, Float64}()
        for (i,p) in enumerate(particles(b))
            if haskey(probs, p)
                probs[p] += weight(b, i)/weight_sum(b)
            else
                probs[p] = weight(b, i)/weight_sum(b)
            end
        end
        b._probs = probs
    end
    return b._probs
end

# Extract Specific type of belief from WPFBelief
initialize_belief(::NothingUpdater, b::WPFBelief) = nothing
initialize_belief(::PreviousObservationUpdater, b::WPFBelief) = b._obs

function initialize_belief(up::KMarkovUpdater, b::WPFBelief)
    hist = history(b)
    if length(hist) > up.k
        [tuple[:o] for tuple in hist[end-up.k+1:end]]
    else
        [tuple[:o] for tuple in hist]
    end
end

function initialize_belief(up::BasicParticleFilter, b::WPFBelief)
    ParticleFilters.resample(up.resampler, b, up.rng)
end