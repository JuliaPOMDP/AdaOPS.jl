struct OPSTree{S,A,O}
    weights::Vector{Vector{Float64}} # stores weights for *belief node*
    children::Vector{Vector{Int}} # to children *ba nodes*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    U::Vector{Float64}
    L::Vector{Float64}
    obs::Vector{O}
    obs_freq::Vector{Int}

    ba_particles::Vector{Vector{S}} # stores particles for *ba nodes*
    ba_children::Vector{Vector{Int}}
    ba_parent::Vector{Int} # maps to parent *belief node*
    ba_U::Vector{Float64}
    ba_L::Vector{Float64}
    ba_Rsum::Vector{Float64} # needed for backup
    ba_action::Vector{A}

    root_belief::WPFBelief
    # _discount::Float64 # for inferring L in visualization
end

function OPSTree(p::OPSPlanner, b_0)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    root_belief = WPFBelief([rand(p.rng, b_0) for i in 1:p.sol.m], fill(1/p.sol.m, p.sol.m), 1.0, 1, 0)
    L, U = bounds(p.bounds, p.pomdp, root_belief, p.sol.bounds_warnings)

    return OPSTree{S,A,O}([root_belief.weights],
                         [Int[]],
                         [0],
                         [0],
                         [U],
                         [L],
                         Vector{O}(undef, 1),
                         [p.sol.m],

                         [],
                         [],
                         Int[],
                         Float64[],
                         Float64[],
                         Float64[],
                         A[],

                         root_belief
                 )
end

function expand!(D::OPSTree, b::Int, p::OPSPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    b = get_wpfbelief(D, b)
    b = resample(b, p)

    for a in actions(p.pomdp, b)
        next_states = S[]
        Rsum = 0.0
        wdict = Dict{O, Array{Float64,1}}() # weights of child beliefs
        fdict = Dict{O, Int}() # frequency of observations

        for s in b.particles
            if !isterminal(p.pomdp, s)
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
                Rsum += r
                if haskey(wdict, o)
                    fdict[o] += 1
                else
                    wdict[o] = Float64[]
                    fdict[o] = 1
                end
                push!(next_states, sp)
            end
        end
        for o in keys(wdict)
            for i in 1:p.sol.m
                push!(wdict[o], obs_weight(p.pomdp, b.particles[i], a, next_states[i], o))
            end
        end
        packing, fdict = gen_packing(wdict, fdict, p.sol.delta)

        nbps = length(packing)
        last_b = length(D.weights)
        push!(D.ba_particles, next_states)
        push!(D.ba_children, [last_b+1:last_b+nbps;])
        push!(D.ba_parent, b.belief)
        push!(D.ba_Rsum, Rsum)
        push!(D.ba_action, a)
        ba = length(D.ba_particles)
        push!(D.children[b.belief], ba)

        resize!(D, last_b+nbps)
        bp = last_b
        for (o, w) in packing
            bp += 1
            D.weights[bp] = w
            D.children[bp] = Int[]
            D.parent[bp] = ba
            D.Delta[bp] = b.depth + 1
            D.obs[bp] = o
        end
        bp = WPFBelief(next_states, fill(1/p.sol.m, p.sol.m), 1.0, bp, D.Delta[bp], D, D.obs[bp])
        bounds_dict = bounds(p.bounds, p.pomdp, bp, packing, p.sol.bounds_warnings)
        Usum = 0.0
        Lsum = 0.0
        for bp in D.ba_children[ba]
            o = D.obs[bp]
            D.L[bp], D.U[bp] = bounds_dict[o]
            D.obs_freq[bp] = fdict[o]
            Usum += D.U[bp] * fdict[o]
            Lsum += D.L[bp] * fdict[o]
        end
        push!(D.ba_U, (D.ba_Rsum[ba] + discount(p.pomdp) * Usum)/p.sol.m)
        push!(D.ba_L, (D.ba_Rsum[ba] + discount(p.pomdp) * Lsum)/p.sol.m)
    end
end

function Base.resize!(D::OPSTree, n::Int)
    resize!(D.weights, n)
    resize!(D.children, n)
    resize!(D.parent, n)
    resize!(D.Delta, n)
    resize!(D.U, n)
    resize!(D.L, n)
    resize!(D.obs, n)
    resize!(D.obs_freq, n)
end

function get_wpfbelief(D::OPSTree, b::Int)
    if b == 1
        return D.root_belief
    else
        ba = D.parent[b]
        return WPFBelief(D.ba_particles[ba], D.weights[b], b, D.Delta[b], D, D.obs[b])
    end
end

function gen_packing(wdict::Dict{O, Array{Float64,1}}, fdict::Dict{O, Int}, delta::Float64) where O
    packing = Dict{O, Array{Float64,1}}()
    new_fdict = Dict{O, Int}()
    for (o1,w1) in wdict
        adjacent_particle = nothing
        for (o2,w2) in packing
            if norm(w1-w2, 1) <= delta
                adjacent_particle = o2
                break
            end
        end
        if adjacent_particle === nothing
            packing[o1] = w1
            new_fdict[o1] = fdict[o1]
        else
            new_fdict[adjacent_particle] += fdict[o1]
        end
    end
    return packing::Dict{O, Array{Float64,1}}, new_fdict::Dict{O, Int}
end