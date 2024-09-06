# Pre-processing
*super brief summary*

## Mesh generation and requirements
---

As with all CFD solvers, defining a suitable mesh is one of the most important steps. XCALibre.jl does not provides mesh generation utilities (yet). Thus, the user will have to generate the grids using external mesh generation tools. However, XCALibre.jl does provide a number of mesh conversion tools to allow user to import their grids. 

XCALibre.jl is an unstructured Finite Volume Method (FVM) library, therefore, we are able to support grids of arbitrary polyhedral cells. In XCALibre.jl a cell-centred FVM approach has been implemented, which is popular since it allows the representation of complex geometries and it is used by most commercial and many open source CFD solvers.

### Mesh conversion

XCALibre.jl at present supports `.unv` mesh formats (which can be generated using [SALOME](https://www.salome-platform.org/)) for simulations in 2D and 3D domains. Additionally, XCALibre.jl also supports the [OpenFOAM](https://openfoam.org/) mesh format for simulations in 3D only (for now). 

!!! note

    Currently, XCALibre.jl only supports loading mesh files stored in ASCII format. Please ensure that when saving grid files they are not saved in binary. Most mesh generation programmes offer the option to export in ASCII (test-based) formats.

The following functions are provided for importing mesh files:

```@docs
UNV2D_mesh
UNV3D_mesh
FOAM3D_mesh
```

### Mesh limitations and requirements

In this section we summarise the key limitations of the mesh loaders presented above, and we also highlight specific requirements. 

#### UNV mesh files
* Only ASCII files are supported
* For 2D simulations the mesh must be contained in the X-Y plane only
* In 3D only hex, tet and prism elements are supported

#### OpenFOAM mesh files
* Only ASCII files are supported
* Boundary groups are not supported (must be deleted manually or the conversion may fail)
* Boundary information is not preserved (walls, symmetry, etc)
* 2D setups are not currently supported (but will be)

## Backend selection
---

In XCALibre.jl the mesh object is very important, as it will not only provide geometry information about the simulation/s, but it is also used to automatically dispatch methods to run on the appropriate backend. Therefore, users must first select the backend they wish to use for the simulations, and then "adapt" the mesh to use the correct backend. 

`XCALIbre.jl` aims to work with all the backends supported by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/). However, since internally `XCALibre.jl` uses sparse arrays to reduce its memory footprint some GPU backends are not currently supported since this functionality is not yet available. Thus, currently only a subset of backends are supported:

* CPU (multithreaded and tested)
* NVidia GPUs (tested)
* AMD GPUs (not tested - feedback welcome)

Selecting a given backend is straight forward. Below examples are provided by assigning a backend to the symbol `backend` and assigning the mesh object to the appropriate backend device using the symbol `mesh_dev`

### CPU backend

Selecting the CPU backend is straight forward. See the example below. Notice that `CPU()` is a backend type provided by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/) which we re-export for convenience.

CPU Example 
```julia
mesh = # call function to load mesh e.g. UNV2_mesh, UNV3_mesh or FOAM3D_mesh
backend = CPU()
mesh_dev = mesh # dummy reference to emphasise the mesh in on our chosen dev (or backend)
```

### GPU backends 

To run on GPUS, the process is also quite simple, but does require a few additional steps.
* Install the corresponding Julia library that supports your hardware. For NVidia GPUs, the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package is required. For AMD GPUs, the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) package is needed.
* Move the mesh object to the backend device using the `adapt` method which for convenience we re-export from [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl)

Example for Nvidia GPU
```julia
mesh = # call function to load mesh e.g. UNV2_mesh, UNV3_mesh or FOAM3D_mesh
backend = CUDABackend()
mesh_dev = adapt(backend, mesh) # make mesh object backend compatible and move to GPU
```

Example for AMD GPU
```julia
mesh = # call function to load mesh e.g. UNV2_mesh, UNV3_mesh or FOAM3D_mesh
backend = ROCBackend()
mesh_dev = adapt(backend, mesh) # make mesh object backend compatible and move to GPU
```