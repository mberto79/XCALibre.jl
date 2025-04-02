# XCALibre.jl

*XPU CFD Algorithms and libraries*

## What is XCALibre.jl?
---

XCALibre.jl (pronounced as the mythical sword *Excalibur*) is a general purpose Computational Fluid Dynamics (CFD) library for 2D and 3D simulations on structured/unstructured grids using the finite volume method. XCALibre.jl has been designed to act as a platform for developing, testing and using *XPU CFD Algorithms and Libraries* to give researchers in both academia and industry alike a tool that can be used to test out ideas easily within a framework that offers acceptable performance. To this end, XCALibre.jl has been implemented to offer both CPU multi-threaded capabilities or GPU acceleration using the same codebase (thanks to the unified programming framework provided by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)). XCALibre.jl also offers a friendly API for those users who are interested in running CFD simulations with the existing solvers and models built into XCALibre.jl. 

## Why XCALibre.jl?
---

For CFD researchers whose research involves developing new numerical methods, turbulence models, or novel CFD methodologies, the development process can be taxing when using commercial packages and their imagination might be constrained by having to adhere to either limited access to internal code functionality or exhausted as they reformulate their ideas to fit within any interfaces provided by the code. There are excellent open source CFD packages where access to functionality or internals is available, however, they are either written in static languages such as C/C++ or dynamic languages such as Python. In static languages, the resulting code is likely highly performant but implementation can be slow and often has a high learning curve (especially if the developer/researcher has no prior knowledge of the language). On the other hand, dynamic languages such as Python can offer a nice development experience at the cost of low runtime or reduced performance. The development of XCALibre.jl was motivated when we discovered the Julia programming language, which promises an interactive and enjoyable implementation experience whilst being able to generate performant code. Thanks to the tools available in the Julia ecosystems (see [Main dependencies](@ref)) it is also possible to generate CPU and GPU code using Julia. As an added bonus, XCALibre.jl can also link readily with the entire Julia ecosystem, including machine learning frameworks such as Flux.jl, Lux.jl, Knet.jl, etc. Thanks to a user-friendly API, ultimately, we hope that XCALibre.jl can be useful to anyone who has an interest in CFD. Enjoy and give us [feedback](@ref #10).


## Main features
---

### Multiple backends
XCALibre.jl embraces parallelism out-of-the-box on all the compute backends supported by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/). That is,

* GPU acceleration on Nvidia, AMD and Intel hardware (Apple hardware is not supported yet)
* Multi-threaded runs on CPUs

!!! note
    
    GPU functionality has only been tested on Nvidia hardware due to availability. However, AMD and INTEL GPUs should also work correctly. Please open an issue if this is not the case so we can investigate. Apple hardware is currently not supported since sparse matrices have not yet been implemented in `Metal.jl`. Notice that some hardware may only support `Float32` operations, in such cases, the mesh should be loaded using the keyword argument `float_type=Float32` (please refer to the [Pre-processing](@ref) section in the user guide). 

### Mesh formats
XCALibre.jl uses its own mesh format to allow the geometry and boundary information to be stored in a format that is suitable for both CPU and GPU calculations. `XCALibre.jl` does not yet provide mesh generation tools. Therefore, some mesh conversion tools for the following mesh formats are provided:

* OpenFOAM meshes for 3D simulations (ascii only)
* UNV meshes for 2D simulations (ascii only)
* UNV meshes for 3D simulations (ascii only)

!!! note

    Although the mesh conversion tools for the OpenFOAM format can import grids designed for 2D simulations, it is not possible to use OpenFOAM grids for 2D cases (at present, support for 2D OpenFOAM grids is in progress). Instead, use a 2D grid generated in the .unv format. Also for 2D grids some requirements must be met when defining the geometry (see [Mesh generation and requirements](@ref))

### Solvers
XCALibre.jl ships with fluid solvers for steady and transient simulations, based on the SIMPLE and PISO algorithms. Currently, the following flow solvers are provided:

* [Steady incompressible flows](@ref XCALibre.Solvers.simple!)
* [Steady weakly compressible flows](@ref XCALibre.Solvers.csimple!)
* [Transient incompressible flows](@ref XCALibre.Solvers.piso!)
* [Transient weakly compressible flows](@ref XCALibre.Solvers.cpiso!)

!!! note

    A solver for highly compressible flows (shock capturing) is currently in testing and will be available in the next major release. Currently the compressible solvers use a sensible energy approach for the energy equation.

### Turbulence and energy models
The list of turbulence models available is expected to expand. The following turbulence models are already available in XCALibre.jl:

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

## Planned development
---

### Capabilities, solvers, algorithms, models, etc.
* Solver for highly compressible flows (including shockwaves)
* Implement multithreaded sparse-matrix multiply for better multithreaded performance (done in v0.3.3)
* Conjugate heat transfer
* ``k-\epsilon`` turbulence model
* Implement parallel versions of more efficient preconditioners

### API
* Pass boundary conditions as a separate object. The current approach results in some internal methods/objects not being fully compatible with GPU kernels, and results is some performance degradation (due to unnecessary data transfer between GPU and host device when boundary condition information is needed/applied). A separation of boundary conditions from field data (scalar and vector fields primarily) would address both of these issues.
* There are no immediate plans for changing the user API
* Fine-tuning of the public API expected based on of user feedback

### Internals
* Overhaul of field data. Currently, both vectors and tensors are built on the primitive scalar field object. Whilst this was convenient during the early development stage, the package has reached a level of maturity that makes this approach hard to maintain, adding unneeded complexity when working with tensors. We plan to define separate internals (how tensors are defined and stored in memory). It is anticipated that this will ease the implementation of models working with tensors, give some performance gains and allow all fields to participate in Julia's broadcasting framework.


## Main dependencies
---

XCALibre.jl is possible (and relies) on the functionality provided by other packages in the Julia ecosystem. For a full list of direct dependencies please refer to the Project.toml file included with this repository. We are thankful to the teams that have helped develop and maintain every single of our dependencies. Major functionally is provided by the following:

* KernelAbstractions.jl - provides a unified parallel programming framework for CPUs and GPUs
* Krylov.jl - provide solvers for linear systems at the heart of XCALibre.jl
* LinearOperators.jl - wrappers for matrices and linear operators
* Atomix.jl - enable atomix operations to ensure race conditions are avoided in parallel kernels
* CUDA.jl, AMD.jl, Metal.jl and OneAPI.jl - not direct dependencies but packages enable GPU usage in Julia

## Related projects
---

There are other wonderful fluid simulation packages available in the Julia ecosystem (please let us know if we missed any):

* [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) 
* [Waterlilly.jl](https://github.com/WaterLily-jl/WaterLily.jl) 
* [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
  