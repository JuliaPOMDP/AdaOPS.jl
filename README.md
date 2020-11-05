# OPS

An implementation of the OPS (Online Packing-based Search) online POMDP Solver.

Problems use the [POMDPs.jl generative interface](https://github.com/JuliaPOMDP/POMDPs.jl).

If you are trying to use this package and require more documentation, please file an issue!

## Installation

```julia
using Pkg
pkg> add "POMDPs"
pkg> registry add "https://github.com/JuliaPOMDP/Registry.git"
pkg> add "git@github.com:AutomobilePOMDP/OPS.jl.git"
```

## Usage

```julia
using POMDPs, POMDPModels, POMDPSimulators, OPS

pomdp = TigerPOMDP()

solver = OPSSolver(bounds=(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
```

For minimal examples of problem implementations, see [this notebook](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/notebooks/Minimal_Example.ipynb) and [the POMDPs.jl generative docs](http://juliapomdp.github.io/POMDPs.jl/latest/generative/).

## Solver Options

Solver options can be found in the `OPSSolver` docstring and accessed using [Julia's built in documentation system](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation-1) (or directly in the [Solver source code](/src/OPS.jl)). Each option has its own docstring and can be set with a keyword argument in the `OPSSolver` constructor.

### Bounds

#### Tuple bounds
The bound passed into `OPSSolver` can be a tuple of form `(lower_bound, upper_bound)`.
#### Function bounds
The bound passed into `OPSSolver` can be a function in the form of `lower_bound, upper_bound = f(pomdp, wpf_belief)`.

#### Independent bounds

In most cases, the recommended way to specify bounds is with an `IndependentBounds` object, i.e.
```julia
OPSSolver(bounds=IndependentBounds(lower, upper))
```
where `lower` and `upper` are either a number, a function or some other objects (see below).

Often, the lower bound is calculated with a default policy, this can be accomplished using a `RolloutLB` which takes in a `PORollout`, `SolvedPORollout`, `FORollout` or `SolvedFORollout`. For the in-depth details, please refer to [BasicPOMCP](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/src/rollout.jl).

Both the lower and upper bounds can be initialized with value estimations using a `FOValueBound` or `POValueBound`.
`FOValueBound` support any `offline MDP Solver` or `Policy`. `POValueBound` support any `offline POMDP Solver` or `Policy`.

If `lower` or `upper` is a function, it should handle two arguments. The first is the `POMDP` object and the second is the `WPFBelief`. To access the state particles in a `WPFBelief` `b`, use `particles(b)`. To access the corresponding weights of particles in a `WPFBelief` `b`, use `weights(b)`. All `AbstractParticleBelief` APIs are supported for `WPFBelief`. More details can be found in the [solver source code](/src/wpf_belief.jl).

In most cases, the `check_terminal` and `consistency_fix_thresh` keyword arguments of `IndependentBounds` should be used to add robustness (see the `IndependentBounds` docstring for more info).

##### Example

For the `BabyPOMDP` from `POMDPModels`, bounds setup might look like this:
```julia
using POMDPModels
using POMDPPolicies
using BasicPOMCP

always_feed = FunctionPolicy(b->true)
lower = RolloutLB(FORollout(always_feed))

function upper(pomdp::BabyPOMDP, b::WPFBelief)
    if all(s==true for s in particles(b)) # all particles are hungry
        return pomdp.r_hungry # the baby is hungry this time, but then becomes full magically and stays that way forever
    else
        return 0.0 # the baby magically stays full forever
    end
end

solver = OPSSolver(bounds=IndependentBounds(lower, upper))
```

## Visualization

[D3Trees.jl](https://github.com/sisl/D3Trees.jl) can be used to visualize the search tree, for example

```julia
using POMDPs, POMDPModels, POMDPModelTools, D3Trees, OPS

pomdp = TigerPOMDP()

solver = OPSSolver(bounds=(-20.0, 0.0), tree_in_info=true)
planner = solve(solver, pomdp)
b0 = initialstate(pomdp)

a, info = action_info(planner, b0)
inchrome(D3Tree(info[:tree], init_expand=5))
```
will create an interactive tree that looks like this:
