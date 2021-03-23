mutable struct WPFBelief{S, A, O} <: AbstractParticleBelief{S}
    particles::Vector{S}
    weights::Vector{Float64}
    weight_sum::Float64
    belief::Int
    depth::Int
    tree::AdaOPSTree{S,A,O}
    _obs::O
    _probs::Union{Nothing, Dict{S,Float64}}
    _hist::Union{Nothing, Array{NamedTuple,1}}
end

WPFBelief(particles::Vector{S}, weights::Vector{Float64}, weight_sum::N, belief::Int, depth::Int, tree::AdaOPSTree{S,A,O}, obs::O) where {S,A,O,N<:Real} = WPFBelief{S,A,O}(particles, weights, convert(Float64, weight_sum), belief, depth, tree, obs, nothing, nothing)
WPFBelief(particles, weights, belief, depth, tree, obs) = WPFBelief(particles, weights, sum(weights), belief, depth, tree, obs, nothing, nothing)

function ParticleFilters.rand(rng::RNG, b::WPFBelief{S}) where {S,RNG<:AbstractRNG}
    t = rand(rng) * b.weight_sum
    i = 1
    cum_weight = b.weights[1]
    while cum_weight < t
        i += 1
        cum_weight += b.weights[i]
    end
    return b.particles[i]
end

ParticleFilters.particles(b::WPFBelief{S}) where S = b.particles
ParticleFilters.n_particles(b::WPFBelief) = length(b.particles)
ParticleFilters.weight(b::WPFBelief, i::Int) = b.weights[i]
ParticleFilters.particle(b::WPFBelief{S}, i::Int) where S = b.particles[i]
ParticleFilters.weight_sum(b::WPFBelief) = b.weight_sum
ParticleFilters.weights(b::WPFBelief) = b.weights
ParticleFilters.weighted_particles(b::WPFBelief{S}) where S = (b.particles[i]=>b.weights[i] for i in 1:length(b.weights))

function POMDPs.mean(b::WPFBelief{S}) where S
    mean_s = zero(eltype(particles(b)))
    for (w, s) in weighted_particles(b)
        mean_s += w * s
    end
    return mean_s / weight_sum(b)
end
POMDPs.currentobs(b::WPFBelief{S,A,O}) where {S,A,O} = b._obs

function POMDPs.history(belief::WPFBelief{S,A,O}) where {S,A,O}
    if belief._hist === nothing
        belief._hist = Vector{NamedTuple}(undef, belief.depth+1)
        D = belief.tree
        if belief.belief !== 1
            b = belief.belief
            depth = belief.depth + 1
            while depth !== 1
                ba = D.parent[b]
                belief._hist[depth] = (o=D.obs[b], a=D.ba_action[ba])
                b = D.parent_b[ba]
                depth -= 1
            end
        end
        belief._hist[1] = (o=D.obs[1],)
    end
    return belief._hist
end

function resample!(resampled::WeightedParticleBelief{S}, b::WeightedParticleBelief{S}, rng::AbstractRNG) where S
    m = n_particles(resampled)
    step = weight_sum(b)/m
    U = rand(rng)*step
    c = weight(b,1) # accumulate sum of weights
    i = 1
    P = particles(b)
    P_resampled = particles(resampled)
    @inbounds for j in 1:m
        while U > c
            i += 1
            c += weight(b, i)
        end
        U += step
        P_resampled[j] = P[i]
    end
    return resampled
end

function resample!(resampled::WeightedParticleBelief{S}, b::B, pomdp::POMDP{S}, rng::AbstractRNG) where {S,B}
    m = n_particles(resampled)
    i = 0
    P_resampled = particles(resampled)
    @inbounds for i in 1:m
        s = rand(rng, b)
        while isterminal(pomdp, s)
            s = rand(rng, b)
        end
        P_resampled[i] = s
    end
    return resampled
end

function switch_to_sibling!(b::WPFBelief{S,A,O}, obs::O, weights::Array{Float64,1}) where {S,A,O}
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

function ParticleFilters.probdict(b::WPFBelief{S,A,O}) where {S,A,O}
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
POMDPs.initialize_belief(::NothingUpdater, b::WPFBelief) = nothing
POMDPs.initialize_belief(::PreviousObservationUpdater, b::WPFBelief{S,A,O}) where {S,A,O} = b._obs

function POMDPs.initialize_belief(up::KMarkovUpdater, b::WPFBelief{S,A,O}) where {S,A,O}
    hist = history(b)
    if length(hist) > up.k
        [tuple[:o] for tuple in hist[end-up.k+1:end]]
    else
        [tuple[:o] for tuple in hist]
    end
end

function POMDPs.initialize_belief(up::BasicParticleFilter, b::WPFBelief{S}) where S
    ParticleFilters.resample(up.resampler, b, up.rng)
end