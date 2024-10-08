---
title: 'XCALibre.jl: A general-purpose unstructured finite volume Computational Fluid Dynamics library'
tags:
  - Julia
  - Computational Fluid Dynamics
  - Finite Volume Method
  - Fluid simulation
  - CFD Solver
  - LES
  - RANS
authors:
  - name: Humberto Medina
    orcid: 0000-0001-5691-9292
    corresponding: true
    affiliation: 1
  - name: Christopher D. Ellis
    orcid: 0000-0001-5237-6673
    affiliation: 1
  - name: Tom Mazin
    affiliation: 1
  - name: Oscar Osborn
    affiliation: 1
  - name: Timothy Ward
    orcid: 0009-0007-4346-1754
    affiliation: 1
  - name: Stephen Ambrose
    orcid: 0000-0002-5833-4084
    affiliation: 1
  - name: Svetlana Aleksandrova
    orcid: 0000-0002-7398-6531
    affiliation: 2
  - name: Benjamin Rothwell
    orcid: 0000-0003-2503-7232
    affiliation: 1
  - name: Carol Eastwick
    orcid: 0000-0001-5773-6439
    affiliation: 1
affiliations:
 - name: The University of Nottingham, UK
   index: 1
 - name: The University of Leicester, UK
   index: 2
date: 18 September 2024
bibliography: paper.bib
---

# Summary

Understanding the behaviour of fluid flow, such as air over a wing, oil lubrication in gas turbines, or cooling air flow in a combustor or turbine is crucial in many engineering applications, from designing aircraft and automotive components to optimising energy systems. Computational Fluid Dynamics (CFD) enables engineers to model real-world processes, optimise designs, and predict performance for a wide range of scenarios, and it has become a vital part of the modern engineering design process for creating efficient, safe, and sustainable designs. As engineers seek to develop and optimise new designs, particularly in fields where there is a drive to push the current state-of-the-art or physical limits of existing design solutions, often, new CFD methodologies or physical models are required. Therefore, extendable and flexible CFD frameworks are needed, for example, to allow seamless integration with machine learning models. In this paper, the features of the  first release of the Julia package XCALibre.jl are presented.  Designed with extensibility in mind, XCALibre.jl is aiming to facilitate the rapid prototyping of new fluid models and to easily integrate with Julia's powerful ecosystem, enabling access to optimisation libraries and machine learning  frameworks to enhance its functionality and expand its application potential, whilst offering multi-threaded performance on CPUs and GPU acceleration. 


# Statement of need

Given the importance of fluid flow simulation in engineering applications, it is not surprising that there is a wealth of CFD solvers available, both open-source and commercially available. Well established open-source codes include: OpenFOAM [@OpenFOAM], SU2 [@SU2], CODE_SATURN [@CODE_SATURN], Gerris [@Gerris], etc. It is a testament to the open-source philosophy, and their developers, that some of these codes offer almost feature parity with commercial codes. However, established open-source and commercial codes have large codebases and, for performance reasons, have been implemented in statically compiled languages which makes it difficult to adapt and incorporate recent trends in scientific computing, for example, GPU computing and interfacing with machine learning frameworks to support the development of new models [@Ellis1; @Ellis2]. As a result, the research community has been actively developing new CFD codes, which is evident within the Julia ecosystem. 

The Julia programming language offers a fresh approach to scientific computing, with the benefits of dynamism whilst retaining the performance of statically typed languages thanks to its just-in-time compilation approach (using LLVM compiler technology). Thus, Julia makes it easy to prototype and test new ideas whilst producing machine code that is performant. This simplicity-performance dualism has resulted in a remarkable growth in its ecosystem offering for scientific computing, which includes state-of-the-art packages for solving differential equations e.g. `DifferentialEquations.jl` [@rackauckas2017differentialequations] , building machine learning models such as `Flux.jl` [@innes:2018], `Knet.jl` [@KNET] and `Lux.jl` [@pal2023lux], optimisation frameworks e.g. `JUMP.jl` [@Lubin2023] , automatic differentiation, such as `Enzyme.jl` [@NEURIPS2020_9332c513] , etc. Likewise, excellent CFD packages have also been developed, most notoriously: `Oceananigans.jl` [@OceananigansJOSS], which provides tools for ocean modelling, `Trixi.jl` [@schlottkelakemper2021purely] which provides high-order solvers using the Discontinuous Garlekin method, and `Waterlilly.jl` [@WeymouthFont2024], which implements the immersed boundary method on structured grids using a staggered finite volume method. In this context, `XCALibre.jl` aims to complement and extend the Julia ecosystem by providing a cell-centred and unstructured finite volume general-purpose CFD framework for the simulation of both incompressible and weakly compressible flows. The package is intended primarily for researchers and students, as well as engineers, who are interested in CFD applications using the built-in solvers or those who seek a user-friendly framework for developing new CFD solvers or methodologies. 

# Key features

The main features available in the first public release (version `0.3.x`) are highlighted here. Users are also encouraged to explore the latest version of [the user guide](https://mberto79.github.io/XCALibre.jl/stable/) where the public API and current features are documented. 

* **XPU computation** `XCALibre.jl` is implemented using `KernelAbstractions.jl` which allows it to support both multi-threaded CPU and GPU calculations. 
* **Unstructured grids and formats** `XCALibre.jl` is implemented to support unstructured meshes using the Finite Volume method for equation discretisation. Thus, arbitrary polyhedral cells are supported, enabling the representation and simulation of complex geometries. `XCALibre.jl` provides mesh conversion functions to import externally generated grids. Currently, the Ideas (`unv `) and `OpenFOAM` mesh formats can be used. The `.unv` mesh format supports both 2D and 3D grids (note that the `.unv` format only supports prisms, tetrahedral, and hexahedral cells). The `OpenFOAM` mesh format can be used for 3D simulations (this format has no cell restrictions and supports arbitrary polyhedral cells).
* **Flow solvers** Steady and transient solvers are available, which use the SIMPLE and PISO algorithms for steady and transient simulations, respectively. These solvers support simulation of both incompressible and weakly compressible fluids (using a sensible energy model).
* **Turbulence models** RANS and LES turbulence models are supported. RANS models available in the current release include: the standard Wilcox $k-\omega$ model [@Wilcox] and the transitional $k-\omega$ LKE model [@HMedina]. For LES simulations the classic `Smagorinsky` model [@Smagorinsky] is available.
* **VTK simulation output** Simulation results are written to `vtk` files for 2D cases and `vtu` for 3D simulations. This allows to perform simulation post-processing in `ParaView`, which is the leading open-source project for scientific visualisation.
* **Linear solvers and discretisation schemes** Users are able to select from a growing range of pre-defined discretisation schemes, e.g. `Upwind`, `Linear` and `LUST` for discretising divergence terms. By design, the choice of discretisation strategy is made on a term-by-term basis, offering great flexibility. Users must also select and configure the linear solvers used to solve the discretised equations.  Linear solvers are provided by `Krylov.jl` [@montoison-orban-2023] and reexported in `XCALibre.jl` for convenience (please refer to the user guide for details on exported solvers). 

# Examples

Users are referred to the documentation where examples for using `XCALibre.jl` are provided, including advanced examples showing how it is possible to integrate the Julia ecosystem to extend the functionality in XCALibre.jl, with examples that include flow optimisation, and integration with the `Flux.jl` machine learning framework. 

## Verification: laminar flow over a backward facing step

The use of `XCALibre.jl` is now illustrated using simple backward-facing step configuration with 4 boundaries as depicted in figure \ref{fig:domain}. The flow will be considered as incompressible and laminar. The `:wall` and `:top` boundaries will be considered as solid wall boundaries. The inflow velocity is 1.5 $m/s$, and the outlet boundary is set up as a pressure outlet (i.e. a `Dirichlet` condition with a reference value of 0 $Pa$). Notice that this case uses a structured grid for simplicity, however, in `XCALibre.jl` the grid connectivity information is unstructured and complex geometries can be used. Here the simulation is set up to run on the CPU. The number of CPU threads to be used can be specified when launching the Julia process, e.g. `julia --threads 4`. The steps needed to carry out the simulation on GPUs can be found in the [documentation](https://mberto79.github.io/XCALibre.jl/stable/).

![Computational domain\label{fig:domain}](domain_mesh.png){ width=80% }

The corresponding simulation setup is shown below:

```Julia
using XCALibre

# Get path to mesh file
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

# Convert and load mesh
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Define flow variables & do checks
velocity = [1.5, 0.0, 0.0]; nu = 1e-3; H = 0.1
Re = velocity[1]*H/nu # check Reynolds number

# Define models
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

# Assign boundary conditions
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0]),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

# Specify discretisation schemes
schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes() # no input provided (will use defaults)
)

# Configuration: linear solvers
solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, 
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, 
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

# Configuration: runtime and hardware information
runtime = set_runtime(iterations=2000, time_step=1, write_interval=2000)
hardware = set_hardware(backend=CPU(), workgroup=1024)

# Configuration: build Configuration object
config = Configuration(
  solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

# Initialise fields
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Run simulation
residuals = run!(model, config);
```

The velocity and pressure results are verified with those obtained using OpenFOAM [@OpenFOAM] and shown in figure \ref{fig:comparison}, showing that they are in excellent agreement. 

![Comparision with OpenFOAM \label{fig:comparison}](BFS_verification.svg){ width=80% }



# References
