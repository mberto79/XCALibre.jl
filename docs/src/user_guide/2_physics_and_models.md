# Physics and models
*Information about setting up a `Physics` object and boundary conditions to represent the flow physics*

## Physics model definition
---

The `Physics` object is part of the highest level API in `XCALibre.jl`. `Physics` objects are a means for users to set up the physics and models that are relevant to their particular CFD simulation. Internally, the `Physics` model created is passed to solvers and is used for dispatch (solvers, algorithms and models). Thus, it is important to ensure that the information provided to this object is correct and representative of the user's intentions. `Physics` models consist of a struct with the  fields shown below. All fields must be provided to the solvers otherwise the construction of the object will fail.

```@docs; canonical=false
Physics
```

 For convenience, `XCALibre.jl` provides a more user-friendly constructor that will automatically take care of the construction of derived fields.

```@docs; canonical=false
Physics(; time, fluid, turbulence, energy, domain)
```

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

Earlier in this section, the dynamic nature of Julia was mentioned in the context of extracting fields for the `Physics` model used in XCALibre.jl. In the following sections this benefit of using Julia will be exploited further. XCALibre.jl takes advantage of Julia's rich type system and we define `Abstract` types to organise major functionality. For example, time models are subtypes of the abstract type `AbstractTimeModel`. Therefore, out-of-the-box we get for free a means to explore implemented features in XCALibre.jl. For example, to identify the time models implemented, we simply need to type the following in the REPL:

```@repl 
using XCALibre
# import Main.subtypes as subtypes # hide
using InteractiveUtils # Load from standard library
Main.subtypes(AbstractTimeModel)
```

From the output it can be seen that there are two time models in XCALibre.jl for `Steady` or `Transient` simulations. These are singleton types and contain (at present) no internal fields or data. They are largely used by XCALibre.jl to dispatch either steady or transient solvers. We are starting to get a picture of how the `Physics` object is constructed. For example, to specify a Steady simulation

The time model can be specified as `Steady` as follows:
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
    using InteractiveUtils # Load from standard library
    # using Pkg; Pkg.add("AbstractTrees") # run to install AbstractTrees
    using AbstractTrees 
    # import Main.subtypes as subtypes # hide
    AbstractTrees.children(d::DataType) = Main.subtypes(d)
    print_tree(AbstractFluid)
end
```

From the subtype tree above, we can see that XCALibre.jl offers 2 major abstract fluid types, `AbstractIncompressible` and `AbstractCompressible`. There are 3 concrete fluid types:

- `Incompressible` - for simulations were the fluid density does not change with pressure
- `WeaklyCompressible` - for simulation were the fluid density is allowed to change (no shockwaves)
- `Compressible` - for simulations were discontinuities may appear (not available for general use yet)

To specify a given fluid type, the `Fluid` wrapper type is used as a general constructor which is specialised depending depending on the fluid type from the list above provided by the user. The constructors require the following inputs:

For incompressible fluid flow
```julia
Fluid{Incompressible}(; nu, rho=1.0) 
```

For compressible fluids (weak formulation)
```julia
Fluid{WeaklyCompressible}(; nu, cp, gamma, Pr)
```
where the input variables represent the following:

- `nu` - kinematic viscosity
- `rho` - fluid density
- `gamma` - specific heat ratio
- `Pr` - Prandlt number
- `cp` - specific heat at constant pressure

For example, an incompressible fluid can be specified as follows
```julia
Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=1e-5),
    ...
)
```

## Turbulence models
---

Below is a representation of the `AbstractTurbulenceModel` inheritance tree. It shows turbulence models available. Turbulence models are defined using the `RANS` and `LES` constructors and passing a specific turbulence model type. As it will be illustrated in the flowing sections.

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractTurbulenceModel) # hide
```
### RANS models constructors

Laminar model: no user input is required. This is a dummy model that does not contribute to the momentum equation.
```julia
RANS{Laminar}() # only constructor needed
```

KOmega model: the standard 2 equation Wilcox model coefficients are passed by default. This model solves 2 transport equations for the turbulent kinetic energy and the specific dissipation rate,  `k` and `omega`, respectively. Subsequently, `k` and `omega` are used to update the turbulent eddy viscosity, `nut`. These 3 fields must be provided with boundary conditions. 
```julia
RANS{KOmega}() # will set the default coefficient values shown below
RANS{KOmega}(; β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5) # set defaults
RANS{KOmega}(β1=0.075) # user can choose to change a single coefficient
```
KOmegaLKE model: the user must provide a reference turbulence intensity (`Tu`) and a tuple of symbols specifying wall boundaries. This model uses 3 equations (`k`, `kl`, `omega`) to update the eddy viscosity (`nut`). These fields must be provided with boundary conditions.
```julia
RANS{KOmegaLKE}(; Tu::Number, walls::Tuple) # no defaults defined
RANS{KOmegaLKE}(Tu = 0.01, walls=(:cylinder,)) # user should provide information for Tu and walls
```

For example, a steady, incompressible simulation using the `KOmega` model can be specified as
```julia
Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=1e-5),
    turbulence = RANS{KOmega}(),
    ...
)
```

### LES models

Laminar model: no user input is required. This is a dummy model that does not contribute to the momentum equation. It can be used to carry out LES simulations of laminar flows. For sufficiently refined grids, this model can be used for DNS simulations. Since `XCALibre.jl` is a Finite Volume Method solver, for complex geometries the simulation should be regarded as a quasi-DNS simulation. For second order accurate DNS simulations very high quality hex grids should be used.
```julia
LES{Laminar}() # only constructor needed
```

Smagorinsky model: the standard model constant is passed by default. Boundary conditions for `nut` must be provided, generally zero gradient conditions work well. No special wall functions for `nut` in LES mode are available.
```julia
LES{Smagorinsky}() # default constructor will use value below
LES{Smagorinsky}(; C=0.15) # default value provided by default
LES{Smagorinsky}(C=0.1) # user selected value
```

For example, an incompressible LES simulation with the `Smagorinsky` model can be specified as
```julia
Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=1e-5),
    turbulence = LES{Smagorinsky}(),
)
```

!!! note

    In the specification above the time model was set as `Transient` since `LES` models are strictly time-resolving. A simulation might run when the `Steady` model is chosen, but the results would likely not be reliable. `XCALibre.jl` by design offers flexibility for users to customise their setup, the consequence is that it falls on users to define a combination of models that is appropriate. Likewise, in the example above an `Isothermal` energy model would have to be selected. See [Energy models](@ref)

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
Energy{SensibleEnthalpy}(Tref=300)  # Users must provide a referent temperature value
```

For example, a steady, compressible `RANS` simulation with the `KOmegaLKE` model can be specified as
```julia
Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(nu=1e-5, cp=1005, gamma=1.4, Pr=0.7),
    turbulence = RANS{KOmegaLKE}(),
    energy = Energy{SensibleEnthalpy}(Tref=300),
    ...
)
```

## Domain definition

The final step when defining a `Physics` model is to specify the domain on which the various models will be used. This is simply a reference to the mesh object.

Continuing from the previous example, a steady, compressible `RANS` simulation with the `KOmegaLKE` model can be specified as
```julia
Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(nu=1e-5, cp=1005, gamma=1.4, Pr=0.7),
    turbulence = RANS{KOmegaLKE}(),
    energy = Energy{SensibleEnthalpy}(Tref=300),
    domain = mesh_dev
)
```
!!! warning

    When passing the mesh objective to complete the definition of the `Physics` object, the mesh must be adapted to the target device where the computation will be performed. See [Backend selection](@ref) for more details.

Notice that the transfer to the target compute backend can also be done inline. For example,

```julia
Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(nu=1e-5, cp=1005, gamma=1.4, Pr=0.7),
    turbulence = RANS{KOmegaLKE}(),
    energy = Energy{SensibleEnthalpy}(Tref=300),
    domain = adapt(CUDABackend(), mesh) # for Nvidia GPUs
)
```

## Boundary conditions
---

The final step to completely capture the physics for the simulation is to define boundary conditions in order to find a concrete solution of the model equations being solved. XCALibre.jl offers a range of boundary conditions. As before, boundary conditions are specified by type and the are classified under the `AbstractBoundary` type and subdivided into 4 additional abstract types `AbstractDirichlet`, `AbstractNeumann`, `AbstractPhysicalConstraint` and `AbstractWallFunction`. The complete abstract tree is illustrated below.

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractBoundary) # hide
```

Philosophically, the four subtypes represent different physical types of boundary conditions:

- `AbstractDirichlet` boundary conditions are used to assign a concrete value to field at the boundary.
- `AbstractNeumann` boundaries are used to fix the field gradient at the boundary.
- `AbstractPhysicalConstraint` boundaries represent physical constraints imposed on the domain.
- `AbstractWallFunction` represent models for treating flow or turbulence quantities in wall regions.

### `AbstractDirichlet` conditions

```julia
Dirichlet(name, value)
```
- `name` is a symbol providing the boundary name
- `value` is a vector or scalar defining desired value at the boundary
  
```julia
FixedTemperature(name; T, model::EnergyModel<:AbstractEnergyModel)
```
- `name` is a symbol providing the boundary name
- `T` is a keyword argument to define the temperate value to be assigned at the boundary
- `model` is a keyword argument that expects an instance of the energy model to be used e.g. `SensibleEnergy`

```julia
DirichletFunction(name, func)
```

- `name` is a symbol providing the boundary name
- `func` is a function identifier. `func` is a user-defined function (but can also be a neural network) that returns a scalar or vector as a function of time and space.
- `func` must adhere to an internal contract. See [`XCALibre.Discretise.DirichletFunction`](@ref) for more details.

### `AbstractNeumann` conditions

```julia
Extrapolated(name, value)
```

- `name` is a symbol providing the boundary name
- `value` is a scalar defining the gradient normal to the boundary

### `AbstractPhysicalConstraint` conditions

`Wall` boundary conditions can be used to provide a boundary with a wall constraint. This boundary type, at present, can only be used to define vectors. For scalar quantities in wall regions a `Extrapolated` (zero gradient) should be imposed.
```julia
Wall(name, value)
```
- `name` is a symbol providing the boundary name
- `value` is a vector defining wall velocity e.g. [0, 0, 0]

```@docs; canonical=false
RotatingWall
```

`Symmetry` boundary condition can be used to assign a symmetry constraint to a given boundary patch in the domain. It can be used for both vector and scalar quantities.
```julia
Symmetry(name)
```
- `name` is a symbol providing the boundary name

`Periodic` boundaries consist of a pair of boundary patches that behave as if they are physically connected. The periodic boundary essentially provides a mapping between these patches and helps in calculating the face values at the interface. The construction of periodic boundaries is different to other boundary conditions because the addressing between each face for the patch pair needs to be calculated and stored. Periodic boundary can be constructed as follows:

```julia
periodic::Tuple = construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol)
```

- `mesh` is a reference to the mesh object
- `backend` defines the expected backend e.g. CPU(), CUDABackend, etc.
- `patch1` and `patch2` symbols of the two patch pair we would like to flag as periodic

The output is a tuple containing two `Periodic` boundary types with information relevant to each boundary patch pair and it can be used directly to assign a periodic boundary for both patches (by splatting into the assignment macro e.g. `periodic...`)

### `AbstractWallFunction` conditions

`KWallFunction` provides a turbulent kinetic energy boundary condition for high-Reynolds models.
```julia
KWallFunction(name)
```
- `name` is a symbol providing the boundary name

`OmegaWallFunction` provides a value for the specific dissipation rate for both low- and high-Reynolds model.
```julia
OmegaWallFunction(name)
```
- `name` is a symbol providing the boundary name

`NutWallFunction` provides a value for the eddy viscosity for high-Reynolds models
```julia
NutWallFunction(name)
```
- `name` is a symbol providing the boundary name

### Assigning conditions (macro)

XCALibre.jl requires that a boundary condition is assigned to every single patch in the domain (as defined within the mesh object) for every field that is part of the solution. To facilitate this process, XCALibre.jl provides the `@assign` macro for convenience. The `@assign` macro has the following signature:

```julia
@assign! model::Physics <physical model> <field> (
    <patch1 boudary>, 
    <patch2 boundary>, 
    ...
    )
```

For example, for a laminar incompressible simulation, only the momentum equation is being solved. Therefore, users need to provide conditions for every patch for the `U` and  `p` fields in the momentum model:

```jldoctest;  filter = r".*"s => s"", output = false
using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU() #  run on CPU
hardware = Hardware(backend=backend, workgroup=4)
mesh_dev = mesh # dummy assignment 

# Flow conditions
velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu # Reynolds number

# Physics models
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

# Define boundary conditions
BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ], 
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall), # scalar wall - set up as zero gradient
            Wall(:top)   # scalar wall - set up as zero gradient
        ]
    )
)

# output

```
!!! hint

    Julia is a dynamic language and objects can be interrogated on the fly (dynamically). Say you created a `Physics` model named `mymodel`, you can interrogate the contents of any of the fields in the `Physics` structure using the `fieldnames` function, e.g. `fieldnames(mymodel.momentum)`, to find which fields need to be provided with boundary conditions. Any fields not ending in `f` should be set. 