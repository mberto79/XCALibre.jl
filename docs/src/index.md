# XCALibre.jl

*XPU CFD Algorithms and libraries*

# What is XCALibre.jl?
---

XCALibre.jl (pronounced as the mythical sword *Excalibur*) is a general purpose Computational Fluid Dynamics library for 2D and 3D simulations on structured/unstructured grids using the finite volume method. XCALibre.jl has been designed to act as a platform for developing, testing and using *XPU CFD Algorithms and Libraries* to give researchers in both academia and industry alike a tool that can be used to test out ideas easily within a framework that offers acceptable performance. To this end, XCALibre.jl has been implemented to offer both CPU multi-threaded capabilities or GPU acceleration using the same codebase (thanks to the unified programming framework provided by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)). Additionally, XCALibre.jl also offers a friendly API for those users who are interested in running CFD simulations with the existing solvers and models built into XCALibre.jl. 

# Why XCALibre.jl?
---

# Main features
---

### Multiple backends
XCALibre.jl embraces parallelism out-of-the-box on all the compute backends supported by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/). That is,

* GPU acceleration on Nvidia, AMD, Intel and Apple hardware
* Multi-threaded runs on CPUs

!!! note
    
    GPU functionality has only been tested on Nvidia hardware due to availability. Although GPUs from other vendors should also work correctly. Please open an issue if this is not the case so we can investigate. Notice that Apple hardware will not work as expected due to support for sparse matrices not being implemented yet on `Metal.jl`

### Mesh formats
XCALibre.jl uses its own mesh format to allow the geometry and boundary information to be stored in a format that is suitable for both CPU and GPU calculations. `XCALibre.jl` does not yet provide mesh generation tools. Therefore, some mesh conversion tools for the following mesh formats are provided:

* OpenFOAM meshes for 3D simulations (ascii only)
* UNV meshes for 2D simulations (ascii only)
* UNV meshes for 3D simulations (ascii only)

!!! note

    Although the mesh conversion tools for the OpenFOAM format can import grids designed for 2D simulations, it is not recommended to use OpenFOAM grids for 2D cases (at present, support for 2D OpenFOAM grids in progress). Instead, use a 2D grid generated in the .unv format. Also for 2D grids some requirements must be met when defining the geometry (see [Mesh generation and requirements](@ref))

### Solvers
XCALibre.jl ships with fluid solvers for steady and transient simulations, based on the SIMPLE and PISO algorithms. Currently, the following flow solvers are provided:

* [Steady incompressible flows](@ref XCALibre.Solvers.simple!)
* [Steady weakly compressible flows](@ref XCALibre.Solvers.csimple!)
* Transient incompressible flows
* Transient weakly compressible flows

!!! note

    A solver for highly compressible flows (shock capturing) is currently in testing and will be available in the next major release. Currently the compressible solvers are based use a sensible energy approach for the energy equation.

### Turbulence and energy models
The list of available turbulence models available is expected to expand. The following turbulence models are already available in XCALibre.jl

* Reynolds-Averaged Navier-Stokes (RANS)
  * ``k-\omega`` - available in low-Reynolds (wall-resolving) and in high-Reynolds (wall functions) mode
  * ``k-\omega`` LKE - transitional model using the Laminar Kinetic Energy concept to model transition onset
  
* Large Eddy Simulation (LES with implicit filtering)
  * Smagorinsky - classic eddy-viscosity sub-grid scale Smagorinsky model

### Boundary conditions 
* User defined functions
* Neural network defined BCs

### Numerical schemes
* Linear - second order schemes for gradients, laplacian and divergence terms
* Upwind - first order upwind-biased scheme for divergence terms
* LUST (divergence) - mixed order upwind-biased scheme for divergence terms
* Midpoint - skew-corrected scheme for gradient calculations
* SteadyState - dummy scheme used to dispatch solvers for operation in steady mode
* Euler - first order semi-implicit time scheme

### Simple API for transport equations

Code example

```julia
U_eqn = (

        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        -Source(∇p.result)

    ) → VectorEquation(mesh)
```



# Planned development
---

#### Capabilities, solvers, algorithms, etc.

#### API

#### Internals
