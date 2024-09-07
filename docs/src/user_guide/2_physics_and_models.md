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
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
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

```@repl
begin
    # Note: this code snippet will not be shown later for succinctness
    using XCALibre
    # using Pkg; Pkg.add("AbstractTrees") # run to install AbstractTrees
    using AbstractTrees 
    # import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
    AbstractTrees.children(d::DataType) = Main.subtypes(d)
    print_tree(AbstractFluid)
end
```

From the subtype tree above, we can see that XCALibre.jl offers 2 major abstract fluid types, `AbstractIncompressible` and `AbstractCompressible`. The concrete fluid types are 3:

* `Incompressible` - for simulations were the fluid density does not change with pressure
* `WeaklyCompressible` - for simulation were the fluid density is allowed to change (no shockwaves)
* `Compressible` - for simulations were discontinuities may appear (not available for general use yet)

To specify a given fluid type, the `Fluid` wrapper type is used as a general constructor which is specialised depending depending on the fluid type from the list above provided by the user. The constructors require the following inputs:

For incompressible fluid flow
```julia
Fluid{Incompressible}(; nu, rho=1.0) 
```

For compressible fluids (weak formulation)
```julia
Fluid{WeaklyCompressible}(; nu, cp, gamma, Pr)
```
where the input variable represent the following:

* `nu` - viscosity
* `rho` - fluid density
* `gamma` - specific heat ratio
* `Pr` - Prandlt number
* `cp` - specific heat at constant pressure


## Turbulence models
---

Below is a representation the `AbstractTurbulenceModel` inheritance tree. It shows turbulence models available. Turbulence models are defined using the `RANS` and `LES` constructors and passing a specific turbulence model type. As it will be illustrated in the flowing sections.

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractTurbulenceModel) # hide
```
### RANS models constructors

Laminar model: no user input is required.
```julia
RANS{Laminar}() # only constructor needed
```

KOmega model: the standard Wilcox model coefficients are passed by default.
```julia
RANS{KOmega}() # will set the default value shown below
RANS{KOmega}(; β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5) # set defaults
RANS{KOmega}(β1=0.075) # user can choose to change a single coefficient
```
KOmegaLKE model: the user must provide a reference turbulence intensity (`Tu`) and a tuple of symbols specifying wall boundaries.
```julia
RANS{KOmegaLKE}(; Tu::Number, walls::Tuple) # no defaults defined
RANS{KOmegaLKE}(Tu = 0.01, walls=(:cylinder,)) # user should provide information for Tu and walls
```

### LES models

Smagorinsky model: the standard model constant is pass by default
```julia
LES{Smagorinsky}() # default constructor will use value below
LES{Smagorinsky}(; C=0.15) # default value provided by default
LES{Smagorinsky}(C=0.1) # user selected value
```

## Energy models
---

Currently, XCALibre.jl offers two options to model the energy equation. A tree of the `AbstractEnergyModel` is shown below. The top-level constructor for energy models is the `Energy` type. Specific constructor signatures are also illustrated below.
```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractEnergyModel) # hide
```

Isothermal model: assumes temperature effects are negligible. Used for incompressible solvers.
```julia
Energy{Isothermal}() # default constructor
```

SensibleEnthalpy model: uses the sensible enthalpy model for the energy equation. Required for the compressible solvers.
```julia
Energy{SensibleEnthalpy}(; Tref)  # constructor definition. No default values given to Tref keyword
Energy{SensibleEnthalpy}(Tref = 300)  # Users must provide a referent temperature value
```

## Boundary conditions
---

The final step to completely capture the physics for the simulation is to define boundary conditions in order to find a concrete solution of the model equations being solved. XCALibre.jl offers a range of boundary condition. As before, boundary conditions are specified by type and the are classified under the `AbstractBoundary` type and subdivided into 4 additional abstract types `AbstractDirichlet`, `AbstractNeumann`, `AbstractPhysicalConstraint` and `AbstractWallFunction`. The complete abstract tree is illustrated below.

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractBoundary) # hide
```

Philosophically, the four subtypes represent different physical types of boundary conditions:

* `AbstractDirichlet` boundary conditions are used to assign a concrete value to a boundary.
* `AbstractNeumann` boundaries are used to fix the gradient at the boundary.
* `AbstractPhysicalConstraint` boundaries represent physical constraints imposed on the domai.
* `AbstractWallFunction` represent models for treating flow or turbulence quantities in wall regions.

### `AbstractDirichlet` conditions
```julia
Dirichlet(name, value)
```

* `name` is a symbol providing the boundary name
* `value` is a vector or scalar
  
```julia
FixedTemperature(name, T, EnergyModel<:AbstractEnergyModel)
```
* `name` is a symbol providing the boundary name
* `T` is the temperate value to be assigned at the boundary
* `EnergyModel` is an instance of the energy model to be used e.g. `SensibleEnergy`

```julia
DirichletFunction(name, func)
```

* `name` is a symbol providing the boundary name
* `func` is a function identifier. `func` is a user-defined function (but can also be a neural network) that returns a scalar or vector as a function of time and space.
* `func` must adhere to an internal contract. See XXX for more details

### `AbstractNeumann` conditions

```julia
Neumann(name, value)
```

* `name` is a symbol providing the boundary name
* `value` is a scalar defining the gradient normal to the boundary

!!! warning

    At present the Neumann boundary should be treated at providing a zero gradient condition only. Internally, a zero gradient value is hard-coded. This behaviour be extended in the near future to allow arbitrary gradients to be defined.

### `AbstractPhysicalConstraint` conditions

`Wall`
```julia
Wall(name, value)
```
* `name` is a symbol providing the boundary name
* `value` is a scalar defining the gradient normal to the boundary

`Symmetry`
```julia
Symmetry(name, value)
```
* `name` is a symbol providing the boundary name
* `value` is a scalar defining the gradient normal to the boundary

`Periodic`
```julia
Periodic(name, value)
```
* `name` is a symbol providing the boundary name
* `value` is a scalar defining the gradient normal to the boundary

### `AbstractWallFunction` conditions

`KWallFunction` provides a turbulent kinetic energy boundary condition for high-Reynolds models.
```julia
KWallFunction(name)
```
* `name` is a symbol providing the boundary name

`OmegaWallFunction` provides a value for the specific dissipation rate for both low- and high-Reynolds model.
```julia
OmegaWallFunction(name)
```
* `name` is a symbol providing the boundary name

`NutWallFunction` provides a value for the eddy viscosity for high-Reynolds models
```julia
NutWallFunction(name)
```
* `name` is a symbol providing the boundary name

### Assigning conditions (macro)

The `@assign` macro ....