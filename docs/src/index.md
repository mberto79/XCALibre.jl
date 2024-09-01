# XCALibre.jl

# What is XCALibre.jl?

XCALibre.jl (pronounced as the mythical sword *Excalibur*) is a general purpose Computational Fluid Dynamics library for 2D and 3D simulations on structured/unstructured grids using the finite volume method. XCALibre.jl has been designed to act as a platform for developing, testing and using *XPU CFD Algorithms and Libraries* to give researchers in both academia and industry alike a tool that can be used to test out ideas easily within a framework that offers acceptable performance. To this end, XCALibre.jl has been implemented to offer both CPU multi-threaded capabilities or GPU acceleration using the same codebase (thanks to the unified programming framework provided by `KernelAbstractions.jl`). Additionally, XCALibre.jl also offers a friendly API for those users who are interested in running CFD simulations with the existing solvers and models built into XCALibre.jl. 

# Why XCALibre.jl?

# Main features

### Multiple backends
XCALibre.jl embraces parallelism out-of-the-box on all the compute backends supported by KernelAbstractions (the developers have only been able to test GPU operation Nvidia GPUs due no access to other hardware)
* GPU acceleration on 
* Multi-threaded runs on CPUs

### Mesh formats
XCALibre.jl uses its own mesh format to allow the geometry and boundary information for the case to be stored in a format that is suitable for both CPU and GPU calculations. However, mesh conversion tools for the following mesh formats are provided:

* OpenFOAM meshes for 3D simulations (ascii only)
* UNV meshes for 2D simulations (ascii only)
* UNV meshes for 3D simulations (ascii only)

### Solvers
* Incompressible steady and transient (based on the SIMPLE and PISO algorithms)
* Weakly-compressible steady and transient (based on the SIMPLE and PISO algorithms)

### Turbulence and energy models
* Reynolds-Averaged Navier-Stokes (RANS)
  * ``k-\omega``
  * ``k-\omega`` LKE
* Large Eddy Simulation (LES with implicit filtering)
  * Smagorinsky 
* Energy models
  * Sensible energy (enthalphy)

### Numerical schemes
* Linear - second order schemes for gradients, laplacian and divergence terms
* Upwind - first order upwind-biased scheme for divergence terms
* LUST (divergence) - mixed order upwind-biased scheme for divergence terms
* Midpoint - skew-corrected scheme for gradient calculations
* SteadyState - dummy scheme used to dispatch solvers for operation in steady mode
* Euler - first order semi-implicit time scheme

# Installation

# Example

# Planned development

### Capabilities, solvers, algorithms, etc.

### API

### Internals
