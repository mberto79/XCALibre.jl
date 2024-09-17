[![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://mberto79.github.io/XCALibre.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://mberto79.github.io/XCALibre.jl/dev/


# XCALibre.jl

*XPU CFD Algorithms and libraries*

## What is XCALibre.jl?


XCALibre.jl (pronounced as the mythical sword *Excalibur*) is a general purpose Computational Fluid Dynamics (CFD) library for 2D and 3D simulations on structured/unstructured grids using the finite volume method. XCALibre.jl has been designed to act as a platform for developing, testing and using *XPU CFD Algorithms and Libraries* to give researchers in both academia and industry alike a tool that can be used to test out ideas easily within a framework that offers acceptable performance. To this end, XCALibre.jl has been implemented to offer both CPU multi-threaded capabilities or GPU acceleration using the same codebase (thanks to the unified programming framework provided by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)). Additionally, XCALibre.jl also offers a friendly API for those users who are interested in running CFD simulations with the existing solvers and models built into XCALibre.jl. 


## Installation


First, you need to [download and install Julia in your system](https://julialang.org/downloads/). Once you have a working installation of Julia, XCALibre.jl can be installed using the built in package manager. 

To install XCALibre.jl directly from Github, open a Julia REPL, press `]` to enter the package manager. The REPL prompt icon will change from **julia>** (green) to **pkg>** (and change colour to blue) or **(myenvironment) pkg>** where `myenvironment` is the name of the currently active Juila environment. Once in package manager mode enter

```julia
pkg> add XCALibre https://github.com/mberto79/XCALibre.jl.git
```

A specific branch can be installed by providing the branch name precided by a `#`, for example, to install the `dev-0.3-main` branch

```julia
pkg> add XCALibre https://github.com/mberto79/XCALibre.jl.git#dev-0.3-main
```

We plan to register XCALibre.jl so that is added to the General Julia Registry. Once this has been completed, to install XCALibre.jl use the following command (in package mode as detailed above)

```julia
pkg> add XCALibre
```

## Main features


* Multiple backends as supported by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
* Ability to import *.unv* and OpenFOAM grids
* Incompressible and (weakly) compressible flow solvers
* RANS and LES turbulence modelling (`KOmega` and `KOmegaLKE` for RANS and `Smagorinsky` for LES, for now!)
* Energy modelling using Sensible Energy model
* Classic boundary conditions, including Direchlich, Neumann (zero gradient), Wall, etc.
* User-defined boundary conditions as neural networks or user-defined functions (source/sink terms soon)
* Easy to link with Julia ecosystem
* A good selection of discretisation schemes available e.g. Euler, Upwind, LUST, etc.
* Simple API for defining new transport equations or solvers

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

## Main dependencies


XCALibre.jl is possible (and relies) on the functionality provided by other packages in the Julia ecosystem. For a full list of direct dependencies please refer to the Project.toml file included with this repository. We are thankful to the teams that have helped develop and maintain every single of our dependencies. Major functionally is provided by the following:

* KernelAbstractions.jl - provides a unified parallel programming framework for CPUs and GPUs
* Krylov.jl - provide solvers for linear systems at the heart of XCALibre.jl
* LinearOperators.jl - wrappers for matrices and linear operators
* Atomix.jl - enable atomix operations to ensure race conditions are avoided in parallel kernels
* CUDA.jl, AMD.jl, Metal.jl and OneAPI.jl - not direct dependencies but packages enable GPU usage in Julia

## Related projects


There are other wonderful fluid simulation packages available in the Julia ecosystem (please let us know if we missed any):

* [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) 
* [Waterlilly.jl](https://github.com/WaterLily-jl/WaterLily.jl) 
* [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
  