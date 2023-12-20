# AdaOPS

[![Build Status](https://travis-ci.com/LAMDA-POMDP/AdaOPS.jl.svg?branch=main)](https://travis-ci.com/LAMDA-POMDP/AdaOPS.jl)

[![Coverage Status](https://coveralls.io/repos/LAMDA-POMDP/AdaOPS.jl/badge.svg?branch=main&service=github)](https://coveralls.io/github/LAMDA-POMDP/AdaOPS.jl?branch=main)

[![codecov.io](http://codecov.io/github/LAMDA-POMDP/AdaOPS.jl/coverage.svg?branch=main)](http://codecov.io/github/LAMDA-POMDP/AdaOPS.jl?branch=main)


An implementation of the AdaOPS (Adaptive Online Packing-guided Search), which is an online POMDP Solver used to solve problems defined with the [POMDPs.jl generative interface](https://github.com/JuliaPOMDP/POMDPs.jl). The [paper](https://openreview.net/forum?id=0zvTBoQb5PA) of AdaOPS was published on NeurIPS'2021.

If you are trying to use this package and require more documentation, please file an issue!

## Installation
Press `]` key to enter the package management mode of Julia. Then, execute the following code.

```julia
pkg> add "POMDPs"
pkg> registry add "https://github.com/JuliaPOMDP/Registry.git"
pkg> add AdaOPS
```

## Usage
```julia
using POMDPs, POMDPModels, POMDPTools, AdaOPS

pomdp = TigerPOMDP()

solver = AdaOPSSolver(bounds=IndependentBounds(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
```

For minimal examples of problem implementations, see [this notebook](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/notebooks/Minimal_Example.ipynb) and [the POMDPs.jl generative docs](http://juliapomdp.github.io/POMDPs.jl/latest/generative/).

## Solver Options
Solver options can be found in the `AdaOPSSolver` docstring and accessed using [Julia's built in documentation system](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation-1) (or directly in the [Solver source code](/src/AdaOPS.jl)). Each option has its own docstring and can be set with a keyword argument in the `AdaOPSSolver` constructor.

### Belief Packing
#### delta
A δ-packing of observation branches will be generated, i.e., the belief nodes with L1 distance less than delta are merged.

### Adaptive Particle Filter
The core idea of the adaptive particle filter is that it can change the number of particles adaptively and use more particles to estimate the belief when needed.
#### grid
`grid` is used to split the state space into multidimensional bins, so that KLD-Sampling can estimate the particle numbers according to the number of bins occupied.
First, a function for converting a state to a multidimensional vector should be implemented, i.e., `Base.convert(::Type{SVector{D, Float64}},::S)`, where `D` is the dimension of the resulted vector.
Then, we define a StateGrid to discretize or split the state space.
A StateGrid is consist of a vector of cutpoints in each dimension. These cutpoints divide the whole space into small tiles. In each dimension, a number of intervals constitute the grid, and each of these intervals is left-closed and right-open with the endpoints be cutpoints with the exception of the left-most interval.
For example, a StateGrid can be defined as `StateGrid([dim1_cutpoints], [dim2_cutpoints], [dim3_cutpoints])`.
All states lie in one tile will be taken as the same.
With the number of tiles (bins) occupied, we can estimate the number of particles using KLD-Sampling.
##### max_occupied_bins
`max_occupied_bins` is the maximum number of bins occupied by a belief. Normally, it is exactly the grid size. However, in some domains, such as Roomba, only states within the room is accessible, and the corresponding bins will never be occupied.
##### min_occupied_bins
`min_occupied_bins` is the minimum number of bins occupied by a belief. Normally, it default to 2. A belief occupying `min_occupied_bins` tiles will be estimated with `m_min` particles. Increasing `min_occupied_bins` indicates that a belief need to occupy more bins so as to be estimated by the same amount of particles.
#### m_min
`m_min` is the minimum number of particles used for approximating beliefs.
#### m_max
`m_max` is the maximum number of particles used for approximating a belief. Normally, `m_max` is set to be big enough so that KLD-Sampling determines the number of particles used. When the KLD-Sampling is disabled, i.e. `grid=StateGrid()`, `m_max` will be sampled during the resampling.
#### zeta
`zeta` is the target error when estimating a belief. Spcifically, we use KLD Sampling to calculate the number of particles needed, where `zeta` is the targe Kullback-Leibler divergence between the estimated belief and the true belief. In AdaOPS, `zeta` is automatically adjusted according to the minimum number of bins occupied such that the minimum number of particles KLD-Sampling method suggests is exactly `m_min`.

### Bounds
#### Dependent bounds
The bound passed into `AdaOPSSolver` can be a function in the form of `lower_bound, upper_bound = f(pomdp, wpf_belief)`, or any other objects for which a `AdaOPS.bounds(obj::OBJECT, pomdp::POMDP, b::WPFBelief, max_depth::Int, bounds_warning::Bool)` function is implemented.

#### Independent bounds
In most cases, the recommended way to specify bounds is with an `IndependentBounds` object, i.e.
```julia
AdaOPSSolver(bounds=IndependentBounds(lower, upper))
```
where `lower` and `upper` are either a number, a function or some other objects (see below).

Often, the lower bound is calculated with a default policy, this can be accomplished using a `PORollout`, `FORollout` or `RolloutEstimator`. For the in-depth details, please refer to [BasicPOMCP](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/src/rollout.jl). Note that when mixing the `Rollout` structs from this package with those from `BasicPOMCP`, you should prefix the struct name with module name.

Both the lower and upper bounds can be initialized with value estimations using a `FOValue` or `POValue`.
`FOValue` support any `offline MDP` `Solver` or `Policy`. `POValue` support any `offline POMDP` `Solver` or `Policy`.

If `lower` or `upper` is a function, it should handle two arguments. The first is the `POMDP` object and the second is the `WPFBelief`. To access the state particles in a `WPFBelief` `b`, use `particles(b)`. To access the corresponding weights of particles in a `WPFBelief` `b`, use `weights(b)`. All `AbstractParticleBelief` APIs are supported for `WPFBelief`. More details can be found in the [solver source code](/src/wpf_belief.jl).

If an object `o` is passed in, `AdaOPS.bound(o, pomdp::POMDP, b::WPFBelief, max_depth::Int)` will be called.

In most cases, the `check_terminal` and `consistency_fix_thresh` keyword arguments of `IndependentBounds` should be used to add robustness (see the `IndependentBounds` docstring for more info).
When using rollout-base bounds, you can specify `max_depth` keyword argument to set the max depth of rollout.

##### Example

For the `BabyPOMDP` from `POMDPModels`, bounds setup might look like this:
```julia
using POMDPModels
using POMDPTools
using BasicPOMCP

always_feed = FunctionPolicy(b->true)
lower = FORollout(always_feed)

function upper(pomdp::BabyPOMDP, b::WPFBelief)
    if all(s==true for s in particles(b)) # all particles are hungry
        return pomdp.r_hungry # the baby is hungry this time, but then becomes full magically and stays that way forever
    else
        return 0.0 # the baby magically stays full forever
    end
end

solver = AdaOPSSolver(bounds=IndependentBounds(lower, upper))
```

## Visualization

[D3Trees.jl](https://github.com/sisl/D3Trees.jl) can be used to visualize the search tree, for example

```julia
using POMDPs, POMDPModels, POMDPTools, D3Trees, AdaOPS

pomdp = TigerPOMDP()

solver = AdaOPSSolver(bounds=(-20.0, 0.0), tree_in_info=true)
planner = solve(solver, pomdp)
b0 = initialstate(pomdp)

a, info = action_info(planner, b0)
inchrome(D3Tree(info[:tree], init_expand=5))
```
will create an interactive tree.

## Analysis
Two utilities, namely `info_analysis` and `hist_analysis`, are provided for getting a sense of how the algorithm is working.
`info_analysis` takes the infomation returned from `action_info(planner, b0)`. It will first visualize the tree if the `tree_in_info` option is turned on. Then it will show stats such as number nodes expanded, total explorations, average observation branches, and so on. `hist_analysis` takes the `hist` from `HistoryRecorder` simulator. It will show similar stats as `info_analysis` but in the form of figures. It should be noted that `HistoryRecoder` will store the tree of each single step, which makes it memory-intensive. An example is shown as follows.
```julia
using POMDPs, AdaOPS, RockSample,ParticleFilters, POMDPTools

m = RockSamplePOMDP(11, 11)

b0 = initialstate(m)
s0 = rand(b0)

bound = AdaOPS.IndependentBounds(FORollout(RSExitSolver()), FOValue(RSMDPSolver()), check_terminal=true, consistency_fix_thresh=1e-5)

solver = AdaOPSSolver(bounds=bound,
                        delta=0.3,
                        m_min=30,
                        m_max=200,
                        tree_in_info=true,
                        num_b=10_000
                        )

adaops = solve(solver, m)
a, info = action_info(adaops, b0)
info_analysis(info)

num_particles = 30000
@time hist = simulate(HistoryRecorder(max_steps=90), m, adaops, BootstrapFilter(m, num_particles), b0, s0)
hist_analysis(hist)
@show undiscounted_reward(hist)
```

## Reference
```
@inproceedings{wu2021adaptive,
  title={Adaptive Online Packing-guided Search for POMDPs},
  author={Wu, Chenyang and Yang, Guoyu and Zhang, Zongzhang and Yu, Yang and Li, Dong and Liu, Wulong and others},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
