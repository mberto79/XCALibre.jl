# Contributor Guide
*Guidelines and practical information for contributing to XCALibre.jl*

## Introduction
---  

In time, our ambition is to document the internal API completely, however, this will take some time and it is an ongoing process. Since XCALibre.jl uses a modular approach and the current functionality covers a good portion of the CFD stack, those interested in working with the internal API should be able to work from existing implementations. In this page we provide key information for those users who want to customise, refine, improve, or extend XCALibre.jl.

From its humble beginning as a 1D diffusion example code to explore the features of the Julia programming language (and speed claims which turned out to be true 😄), XCALibre.jl was meant to be shared and used to help students understand the Finite Volume Method and as an entry point to explore the implementation details underpinning CFD. As the code base has grown, this original purpose remains. However, XCALibre.jl is now a more complete CFD software stack which can be used by both researchers and students to test out new ideas, even on complex geometry with acceptable performance. XCALibre.jl will hopefully continue to grow and offer more functionality as it has been the case since work on XCALibre.jl started. In the sharing spirit of open-source software we also welcome code contributions, and we hope to make this process as simple as possible. However, we ask contributors to follow a few guidelines to ensure that we grow XCALibre.jl in a sustainable and maintainable manner.

## Some guidelines
---

To help use keep the codebase consistent and allow us to merge future Pull Requests (PRs) more easily, we kindly request that contributors adhere to some basic guidelines. We are trying to strike a balance between consistency and ease of contribution by aiming not to be overly demanding on contributors. A minimum set of guidelines is provided below (subject to review as the codebase evolves), in no particular order:

### Code style

* Follow the [style guide](https://docs.julialang.org/en/v1/manual/style-guide/) recommended in the official Julia documentation
* Use camel case format for custom types e.g. `MyType`
* We prefer easy-to-read function names e.g. use `calculate_flux` over `calf` or similar. All in lower case and words separated with an underscore
* For internal variables feel free to use Unicode symbols, or camel case identifiers. However, please refrain from doing this for any top-level or user-facing API variables.
* Although Julia allows some impressive one-liners, please avoid. These can be hard to reason sometimes, aim to strike a balance between succinctness and clarity. 

### Code contribution
* Please open a PR for any code contributions against our `main` branch if your are contributing top level functionality that builds on existing code e.g. a new turbulence model or a new boundary condition (these contributions will typically be included inside one of the existing sub modules)
* For contributions that may require code reorganisation please do get in touch to ensure this aligns with any planned changes (open an issue). PRs will likely be requested against the current `dev` branch
* Ideally, all contributions will also include basic documentation and tests.

### Help wanted
* If you have specific expertise in MPI or Multi-GPU implementation and wish to get involved please get in touch.

## Module organisation
---

| Module | Description |
|:-------|:------------|
XCALibre.Mesh | This module defines all types required to construct the mesh object used by the flow solvers. Two main mesh types are used `Mesh2` and `Mesh3` used for 2D and 3D simulations, respectively. Some access functions are also included in this module. |
XCALibre.Fields | This module defines the fields used to hold and represent the flow variable. Scalar, vector and tensor fields are defined e.g. `ScalarField`, etc. Information is stored at cell centres. These fields also have face variant where information is stored at face centres e.g. `FaceVectorField`. These fields are generally used to store fluxes. A limited set of field operations are also defined e.g. `getfield` to allow indexing field object directly. |
XCALibre.ModelFramework | This module provides the framework used to define scalar and vector model equations whilst storing information about the operators used. The data structure also defines sparse matrices used to store discretisation information.  |
XCALibre.Discretise | This module defines the various operators needed to represent each terms in a model equation and the main discretisation loop that linearises each term according to the schemes available. Boundary conditions are also implemented in this module. |
XCALibre.Solve | This module includes all functions and logic needed to solve the linear system of equations that follows the equation discretisation. The internal API to solve these systems of equations is included in this module. |
XCALibre.Calculate | Implementation of functions use to carry out calculations essential to the implementation of flow solvers is included in this module. This includes interpolation of variables from cell centroid to cell faces, gradient calculation, surface normals, etc. |
XCALibre.ModelPhysics | This module includes the implementations of all the physical models i.e. fluid, turbulence and energy models. |
XCALibre.Simulate | This model contains information needed to set up a simulation, including the [`Configuration`](@ref) type used by all flow solvers. |
XCALibre.Solvers | Implementations of the SIMPLE and PISO flow solvers from steady and unsteady solutions, including their compressible variant. |
XCALibre.Postprocess | A limited set of functions for postprocessing are implemented in this module.  |
XCALibre.IOFormats | Functionality to write simulation results to VTK or OpenFOAM output files are implemented in this module. |
XCALibre.FoamMesh | Stand alone module to parse, process (geometry calculation) and import OpenFOAM mesh files into XCALibre.jl |
XCALibre.UNV3 | Stand alone module to parse, process (geometry calculation) and import UNV (3D) mesh files into XCALibre.jl |
XCALibre.UNV2 | Stand alone module to parse, process (geometry calculation) and import UNV (2D) mesh files into XCALibre.jl |

## Key types and structures
---

### Mesh type and  format

The definitions of the data structures used to define `Mesh3` objects is given below. Note that for succinctness only 3D structures are shown since all the 2D structures are identical. The type is only used for dispatching solvers to operate in 2D or 3D. 

```@docs; canonical=false
Mesh3
Node
Face3D
Cell
Boundary
```

To fully characterise how mesh information is represented in XCALibre.jl, it is important to highlight the following "contracts" that are exploited throughout:

* Node, face and cell IDs correspond to the index where they are stored in their corresponding vector in the `Mesh3` structure e.g. `Mesh3.Faces[10]` would return information for the face whose ID is 10. These vectors are 1-indexed as standard in Julia.
* Face normals at boundary faces is always pointing outside the domain e.g. they point in the direction expected in the FVM
* Face normals for internal faces is always pointing in the direction from the ownerCell with the smallest ID to the largest. Since the discretisation loop is cell based, for the cell with the highest ID the direction must be reversed. This information is tracked in `Mesh3.nsign` which stores 1 if the face normal is correctly aligned or -1 if the normal needs to be reversed.
* Boundary faces (e.g. patches) are stored consecutively in `Mesh3.Faces` starting at the beginning of the array followed by all the internal faces.
* Boundary faces are those connected only to 1 `Cell`, thus, for these faces the entry `Face3D.ownerCells` is a 2-element vector with a repeated index e.g. [3, 3]
* Boundary cells only store information for internal faces. This improves performance for the main discretisation loop (cell based) since it can always been assumed that none of the faces will be a boundary face, which are dealt with in a separate loop.


### Field types

After the `Mesh2` and `Mesh3` objects, the most fundamental data structure in XCALibre.jl are fields used to represent the flow variables. In the current implementation, the prime field is the `ScalarField` for storing information at cell centres, and the corresponding `FaceScalarField` to store information at cell faces (normally fluxes). Internally, both are identical, therefore, the internal structure of the "Face" variants will not be discussed. `ScalarFields` have the following definition:

```@docs; canonical=false
ScalarField
```

Vector and tensors are represented internally by their individual components, as an illustration the `VectorField` type is shown below. Notice that each component of both vector and tensor fields are themselves represented by the same `ScalarField` type shown above. This has some implementation benefits, i.e. reducing duplication and allowing for rapid development. However, the performance of other means of storing these fields is being investigated. Thus, these internals may change if we identify performance gains.

```@docs; canonical=false
VectorField
```

All fields behave (mostly) like regular arrays and can be indexed using the standard Julia syntax e.g. say `U` is a `VectorField`, then `U[3]` would return the vector stored in cell with ID = 3 (as a `SVector{3}` for improved performance).

!!! note

    The implementation of broadcasting operations has been put on hold until a more thorough investigation of alternative data structures for vector and tensor fields has been completed. This is ongoing work. 



## Boundary conditions
---

To implement a new boundary condition the following elements are required (see source code for [`Dirichlet`](@ref), for example):

* type definition: a structure containing the fields `:ID` and `:value`
* `fixedValue` function: used to check that user provided information is suitable for this boundary being implemented
* Implementation of the boundary face value: functor defining the boundary condition implementation, facilitated by a macro (see details below)
* Scalar and vector face value interpolation: kernels to specify how to transfer cell information to the boundary. 

Developers and contributors are encouraged to explore the source code for examples on existing implementations of boundary conditions. To ease their implementation the [`XCALibre.Discretise.@define_boundary`](@ref) is provided.

```@docs; canonical=false
XCALibre.Discretise.@define_boundary
```

## Implementing new models
---

The internal API for models is still somewhat experimental, thus, it is more instructive to explore the source code. Although not fully finalised, the implementation is reasonably straight forward to follow thanks to the abstractions that have been adopted, some of which are described above, as well as the use of descriptive names for internal functions. If you need help on any aspect of the internals, for now, it is recommended to contact us by opening an issue.