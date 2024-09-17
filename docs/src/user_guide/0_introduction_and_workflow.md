# Introduction
*This page explains the overarching workflow in XCALibre.jl and provides a list of contents*

## Workflow overview
---

XCALibre.jl has been designed to incorporate a logical workflow, that is, the sequence of setup steps that would take a user naturally from the start of a simulation to its final post-processing. The key steps are listed below

* [Pre-processing](@ref) - this step involves defining the computational domain and selecting the corresponding backend to performs the calculations
* [Physics and models](@ref) - this step involves defining the fluid type, flow properties, selecting appropriate models and setting suitable boundary conditions
* [Numerical setup](@ref) - in this phase of the simulation set up all aspect related to the numerics are chosen, from distretisation schemes all the way to solvers and preconditioners. 
* [Runtime and solvers](@ref) - once everything has been set up, the simulation is ready to run, once runtime information such as time steps and solution saving intervals has been selected. The final step is to actually start the simulation.
* [Post-processing](@ref) - the final step is to enjoy all the pretty pictures!

The user guide is structured such that the information provided is grouped following the workflow illustrated above. For convenience the contents of the user guide are included below, with links to the relevant sections.

## Contents
---

```@contents
Pages = Main.USER_GUIDE_PAGES
Depth = 2
```