# Post-processing
*Information about postprocessing XCALibre.jl results*

## ParaView
---

All solvers in XCALibre.jl will write simulation results in formats that can be loaded directed in [ParaView](https://www.paraview.org/), which is the leading open-source project for scientific visualisation and postprocessing. More information about how to use [ParaView](https://www.paraview.org/) can be found in the [resources](https://www.paraview.org/resources/) page on their website.

XCALibre.jl uses two different `VTK` format depending on the type of flow solver used. For 2D simulations, the results are written to file using the `.vtk` file format. 3D simulations are written using the unstructured `VTK` file format, `.vtu`. 

!!! note

    A limitation of the current `VTK` writers in XCALibre.jl is that boundary information is stored along with internal mesh cell information, and result are stored at cell centres only. Thus, care must be taken when visualising results at boundary faces. In future releases a separation of boundary and internal mesh results is planned. 

## Available functions
---

Although [ParaView](https://www.paraview.org/) offers considerable flexibility for postprocessing results. Users may also wish to carry out more advanced or different analyses on their CFD results. At present XCALibre.jl offer a limited set of pre-defined postprocessing functions, however, defining new custom postprocessing functions is reasonably straight forward since these can be written in pure Julia. In this section, examples of postprocessing functions will be provided as an illustration. 

!!! note

    At present all postprocessing functions available in XCALibre.jl will only execute on CPUs and should be considered experimental. In time, we plan to offer a larger selection of postprocessing tools once we settle on a "sensible" (maintainable and extensible) API which is likely to include options for runtime postprocessing. Suggestion are welcome, we have opened issue [#11](@ref) for this purpose. 