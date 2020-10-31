# ARDESPOT

[![Build Status](https://travis-ci.org/JuliaPOMDP/ARDESPOT.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/ARDESPOT.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/ARDESPOT.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/ARDESPOT.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/ARDESPOT.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/ARDESPOT.jl?branch=master)

An implementation of the AR-DESPOT (Anytime Regularized DEterminized Sparse Partially Observable Tree) online POMDP Solver.

Tried to match the pseudocode from this paper: http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf as closely as possible. Look there for definitions of all symbols.

Problems use the [POMDPs.jl generative interface](https://github.com/JuliaPOMDP/POMDPs.jl).

If you are trying to use this package and require more documentation, please file an issue!

## Installation

On Julia v0.7 or later, ARDESPOT is in the JuliaPOMDP registry

```julia
Pkg.add("POMDPs")
using POMDPs
POMDPs.add_registry()
Pkg.add("ARDESPOT")
```

## Usage

```julia
using POMDPs, POMDPModels, POMDPSimulators, ARDESPOT

pomdp = TigerPOMDP()

solver = DESPOTSolver(bounds=(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
```

For minimal examples of problem implementations, see [this notebook](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/notebooks/Minimal_Example.ipynb) and [the POMDPs.jl generative docs](http://juliapomdp.github.io/POMDPs.jl/latest/generative/).

## Solver Options

Solver options can be found in the `DESPOTSolver` docstring and accessed using [Julia's built in documentation system](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation-1) (or directly in the [Solver source code](/src/ARDESPOT.jl)). Each option has its own docstring and can be set with a keyword argument in the `DESPOTSolver` constructor. The definitions of the parameters match as closely as possible to the corresponding definition in the pseudocode of [this paper](http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf).

### Bounds

#### Independent bounds

In most cases, the recommended way to specify bounds is with an `IndependentBounds` object, i.e.
```julia
DESPOTSolver(bounds=IndependentBounds(lower, upper))
```
where `lower` and `upper` are either a number or a function (see below).

Often, the lower bound is calculated with a default policy, this can be accomplished using a `DefaultPolicyLB` with any `Solver` or `Policy`.

If `lower` or `upper` is a function, it should handle two arguments. The first is the `POMDP` object and the second is the `ScenarioBelief`. To access the state particles in a `ScenairoBelief` `b`, use `particles(b)` (or `collect(particles(b))` to get a vector).

In most cases, the `check_terminal` and `consistency_fix_thresh` keyword arguments of `IndependentBounds` should be used to add robustness (see the `IndependentBounds` docstring for more info).

##### Example

For the `BabyPOMDP` from `POMDPModels`, bounds setup might look like this:
```julia
using POMDPModels
using POMDPPolicies

always_feed = FunctionPolicy(b->true)
lower = DefaultPolicyLB(always_feed)

function upper(pomdp::BabyPOMDP, b::ScenarioBelief)
    if all(s==true for s in particles(b)) # all particles are hungry
        return pomdp.r_hungry # the baby is hungry this time, but then becomes full magically and stays that way forever
    else
        return 0.0 # the baby magically stays full forever
    end
end

solver = DESPOTSolver(bounds=IndependentBounds(lower, upper))
```

#### Non-Independent bounds

Bounds need not be calculated independently; a single function that takes in the `POMDP` and `ScenarioBelief` and returns a tuple containing the lower and upper bounds can be passed to the `bounds` argument.

## Visualization

[D3Trees.jl](https://github.com/sisl/D3Trees.jl) can be used to visualize the search tree, for example

```julia
using POMDPs, POMDPModels, POMDPModelTools, D3Trees, ARDESPOT

pomdp = TigerPOMDP()

solver = DESPOTSolver(bounds=(-20.0, 0.0), tree_in_info=true)
planner = solve(solver, pomdp)
b0 = initialstate_distribution(pomdp)

a, info = action_info(planner, b0)
inchrome(D3Tree(info[:tree], init_expand=5))
```
will create an interactive tree that looks like this:

![DESPOT tree](img/tree.png)

## Relationship to DESPOT.jl

[DESPOT.jl](https://github.com/JuliaPOMDP/DESPOT.jl) was designed to exactly emulate the [C++ code](https://github.com/AdaCompNUS/despot) released by the original DESPOT developers. This implementation was designed to be as close to the pseudocode from the journal paper as possible for the sake of readability. ARDESPOT has a few more features (for example DESPOT.jl does not implement regularization and pruning), and has more compatibility with a wider range of POMDPs.jl problems because it does not emulate the C++ code.
