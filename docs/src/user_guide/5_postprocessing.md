# Post-processing
*Information about postprocessing XCALibre.jl results*

## ParaView
---

All solvers in XCALibre.jl will write simulation results in formats that can be loaded directly in [ParaView](https://www.paraview.org/), which is the leading open-source project for scientific visualisation and postprocessing. More information about how to use [ParaView](https://www.paraview.org/) can be found in the [resources](https://www.paraview.org/resources/) page on their website.

XCALibre.jl can output simulation results to either `VTK` compliant formats or `OpenFOAM` format. XCALibre.jl uses two different `VTK` formats depending on the type of flow solver used. For 2D simulations, the results are written to file using the `.vtk` file format. 3D simulations are written using the unstructured `VTK` file format, `.vtu`. 

!!! note

    A limitation of the current `VTK` writers in XCALibre.jl is that boundary information is stored along with internal mesh cell information, and results are stored at cell centres only. Thus, care must be taken when visualising results at boundary faces. Boundary information for `fixedValue` boundaries is displayed corrently when the results are saved in the `OpenFOAM` format. 

## Available functions
---

Although [ParaView](https://www.paraview.org/) offers considerable flexibility for postprocessing results, users may also wish to carry out more advanced or different analyses on their CFD results. At present XCALibre.jl offers a limited set of pre-defined postprocessing functions, however, defining new custom postprocessing functions is reasonably straight-forward since these can be written in pure Julia. In this section, examples of postprocessing functions will be provided as an illustration. 

!!! note

    At present all postprocessing functions available in XCALibre.jl will only execute on CPUs and should be considered experimental. Once we settle on a "sensible" (maintainable and extensible) API, we plan to offer a larger selection of postprocessing tools  which are likely to include options for runtime postprocessing.

### Example: Calculate boundary average

In this example, a function is shown that can be used to calculate the average on a user-provided boundary. 

```@docs; canonical=false
boundary_average
```

To calculate pressure and viscous forces, the following functions are available:

```@docs; canonical=false
pressure_force
viscous_force
```
