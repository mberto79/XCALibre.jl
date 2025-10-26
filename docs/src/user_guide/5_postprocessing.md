# Post-processing
*Information about postprocessing XCALibre.jl results*

## ParaView
---

All solvers in XCALibre.jl will write simulation results in formats that can be loaded directly in [ParaView](https://www.paraview.org/), which is the leading open-source project for scientific visualisation and postprocessing. More information about how to use [ParaView](https://www.paraview.org/) can be found in the [resources](https://www.paraview.org/resources/) page on their website.

XCALibre.jl can output simulation results to either `VTK` compliant formats or `OpenFOAM` format. XCALibre.jl uses two different `VTK` formats depending on the type of flow solver used. For 2D simulations, the results are written to file using the `.vtk` file format. 3D simulations are written using the unstructured `VTK` file format, `.vtu`. 

!!! note

    A limitation of the current `VTK` writers in XCALibre.jl is that boundary information is stored along with internal mesh cell information, and results are stored at cell centres only. Thus, care must be taken when visualising results at boundary faces. Boundary information for `fixedValue` boundaries is displayed corrently when the results are saved in the `OpenFOAM` format. 

## Available functions

Although [ParaView](https://www.paraview.org/) offers considerable flexibility for postprocessing results, users may also wish to carry out more advanced or different analyses on their CFD results. At present XCALibre.jl offers a limited set of built-in runtime postprocessing functions, currently these include time averaging a scalar or vector field over a specified range of iterations, and also the root mean square (RMS) of the fluctuations of a field. 
### Example: Calculate time averaged field 
As an example, to average any Scalar or Vector field an instance of `FieldAverage` must be created.
```@docs; canonical=false
FieldAverage
```
Once created this is simply passed to the `Configuration` object as an extra argument with the keyword `postprocess`. For example to average the velocity field over the whole simulation, 
```julia
postprocess = FieldAverage(model.momentum.U; name="U_mean")
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)
```
The rest of the case would remain exactly the same. 
### Example: Calculate field RMS 
The RMS of a Scalar of Vector field can be obtained in a similar way to the time averaged field, instead an instance of `FieldRMS` is created which has the following definition. 
```@docs; canonical=false
FieldRMS
```
The RMS of the velocity field can be easily calculated by creating an instance of `FieldRMS` and passing it to the `Configuration` object with the keyword `postprocess`, with the rest of the case remaining unchanged.
```julia
postprocess = FieldRMS(model.momentum.U; name="U_rms")
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)
```
### Example: Calculate time average and RMS for multiple fields at a time
To post-process multiple fields as a time, a vector of `FieldAverage` and `FieldRMS` objects can be passed instead e.g. 
```julia
postprocess = [FieldRMS(model.momentum.U; name="U_rms"), FieldAverage(model.momentum.U; name ="U_mean"), FieldAverage(model.momentum.p; name ="p_mean")]
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)
```


If more functionality is required, defining new custom postprocessing functions is reasonably straight-forward since these can be written in pure Julia. In this section, examples of postprocessing functions will be provided as an illustration. 

!!! note

    At present the postprocessing functions available in XCALibre.jl shown below will only execute on CPUs and should be considered experimental. Once we settle on a "sensible" (maintainable and extensible) API, we plan to offer a larger selection of postprocessing tools which are likely to include options for runtime postprocessing.

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
