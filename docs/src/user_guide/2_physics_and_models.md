# Physics and models
*super brief summary*

## Physics model definition
---

The `Physics` object is part of the highest level API in `XCALibre.jl`. `Physics` objects are a means for users to set up the physics and models that are relevant to their particular CFD simulation. Internally, the `Physics` model created is pass to solvers and it is used for dispatch (solvers, algorithms and models). Thus, it is important to ensure that the information provided to this object is correct and representative of the user's intentions. `Physics` models consist of a struct with the  fields shown below. All fields must be provided to the solvers otherwise the construction of the object will fail.

```julia
struct Physics{T,F,M,Tu,E,D,BI}
    time::T
    fluid::F
    momentum::M 
    turbulence::Tu 
    energy::E
    domain::D
    boundary_info::BI
end 
```

 For convenience, `XCALibre.jl` provides a more user-friendly constructor that will automatically take care of the construction of derived fields. This constructor uses keyword arguments and has the following signature (follow the link for more information)

[`Physics(; time, fluid, turbulence, energy, domain)`](@ref)

At this point, it is worth reminding users that one of the benefits of working with `Julia` is its dynamic nature. This means that objects can be dynamically interrogated. For example:

```@repl 
using XCALibre
fieldnames(Physics)
```

This will provide users with all the fields that make up the `Physics` object. This is a nice way to explore the makeup of any object in `Julia` (and by extension `XCALibre.jl`). The rest of this page will provide details of the physics models available in XCALibre.jl, including:

* [Time models](@ref)
* [Fluid types](@ref)
* [Turbulence models](@ref)
* [Energy models](@ref)

!!! warning

    The mesh provided to the `domain` field must already be adapted to the required backend device. See [Backend selection](@ref) for detail. 

!!! note

    In the next version of XCALibre.jl the field `boundary_info` will be likely removed outside of the Physics object in order to improve on the current performance (avoiding unnecessary memory movement between GPU backends and the host devices).

## Time models
---

Earlier in this section, the dynamic nature of Julia was mentioned in the context of extracting fields for the `Physics` model used in XCALibre.jl. In the following sections this benefit of using Julia will be exploited further. XCALibre.jl takes advantage of Julia's rich type system and we define `Abstract` types to organise major functionality. For example, time models are subtypes of the abstract type `AbstractTimeModel`. Therefore, out-of-the-box we get for free a means to explore implemented features in XCALibre.jl. For example, to identify the time models implemented, we need simply need to type the following in the REPL:

```@repl 
using XCALibre
import Main.subtypes as subtypes # hide
subtypes(AbstractTimeModel)
```

From the output it can be seen that there are two time models in XCALibre.jl for `Steady` or `Transient` simulations. These are singleton types and contain (at present) not internal fields or data. They are largely used by XCALibre.jl to dispatch either steady or transient solvers. We starting to get a picture of how the `Physics` object is constructed. For example, to specify a Steady simulation

```julia
Physics(
    time = Steady()
    ...
)
```


## Fluid types
---

Following from the idea of using Julia's dynamic features to explore the types available in XCALibre.jl, in this section we will explore the fluid types available. This time, we will use a helper package `AbstractTrees` (which can be installed in the usual way, by entering into package mode in the REPL and typing `add AbstractTrees`). In XCALibre.jl all fluid types are subtypes of `AbstractFluid`. The types available are shown in the example below:

```@example
using XCALibre
# using Pkg; Pkg.add("AbstractTrees") # uncomment this line to install AbstractTrees
using AbstractTrees
import Main.subtypes as subtypes # hide
AbstractTrees.children(d::DataType) = subtypes(d)
print_tree(AbstractFluid)

# Note: this code snippet will not be shown later for succinctness
```

## Turbulence models
---

```@example
using XCALibre # hide
using AbstractTrees # hide
import Main.subtypes as subtypes # hide
AbstractTrees.children(d::DataType) = subtypes(d) # hide
print_tree(AbstractTurbulenceModel) # hide
```

## Energy models
---

```@example
using XCALibre # hide
using AbstractTrees # hide
import Main.subtypes as subtypes # hide
AbstractTrees.children(d::DataType) = subtypes(d) # hide
print_tree(AbstractEnergyModel) # hide
```

## Boundary conditions
---

```@example
using XCALibre # hide
using AbstractTrees # hide
import Main.subtypes as subtypes # hide
AbstractTrees.children(d::DataType) = subtypes(d) # hide
print_tree(AbstractBoundary) # hide
```