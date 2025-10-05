# Release notes

The format used for this `changelog` is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Notice that until the package reaches version `v1.0.0` minor releases are likely to be `breaking`. Starting from version `v0.3.1` breaking changes will be recorded here. 

## Version [v0.5.2] - 2025-XX-XX

### Added
*  Initial support for mixed precision (UNV meshes only) [#67](@ref)
*  New solver for simulating conduction in solids [#65](@ref)
*  New LES turbulent kinetic energy one equation model (`KEquation`) [#71](@ref)
*  Surface tension model for fluids [#72](@ref)
*  High fidelity viscosity models for H2 and N2 [#72](@ref)
*  High fidelity thermal conductivity models for H2 and N2 [#72](@ref)
*  `SetFields` utility that allows to set a field to desired value within a box / circle / sphere [#73](@ref)
*  Helmholtz Energy equation of state and supporting framework for it for H2 and N2 [#75](@ref)
*  Added `RotatingWall` velocity boundary condition [#81](@ref)


### Fixed
* The `UNV3D_mesh` reader has been updated to ensure that the ordering of face nodes is determined in a more robust manner. This resolves some issues when loading a `UNV` mesh that is later used to store simulation results in the `OpenFOAM` format [#64](@ref)
* In the construction of a `Physics` object, the `boundary_map` function returned a `boundary_info` struct which was incorrectly using an abstract type `Integer`. This resulted in a failure to convert a `Physics` object to the cpu and back to the gpu [#67](@ref)
* Fixed calculation of interpolation weights for 2D UNV grids [#68](@ref)
* Added method for boundary interpolation when using `LUST` and `DirichletFunction`[#68](@ref)
* Fixed `boundary_interpolation` for `DirichletFunction` when a function is passed [#79](@ref)
  
### Changed
* The constructors for `ScalarField` and `FaceScalarField` now include a `store_mesh` keyword argument to request a reference of the mesh to be stored (default) or not (setting `store_mesh=false`). This can be used to not include references to the mesh for each field in `VectorFields` and `TensorFields`. This has improved compile times and decreased simulation times (particularly on the GPU - perhaps due to freeing registers used to carry unnecessary type information) [#69](@ref)
* Internally, the calculation of interpolation weights and other geometric properties are calculated using the same function (defined in the `Mesh` module) [#69](@ref)
* The default discretisation for laplacian terms uses the over-relaxed formulation by default. This will have no effect on orthogonal grids, but tends to be more robust in complex geometries at the expense of accuracy, which can be recovered by adding additional orthogonal correction loops (using the key word argument `ncorrectors` in the `run!` function) [#73](@ref)
* Cleaned code for all solvers and improved stability of incompressible solver by removing the update of the mass flow based on the velocity field from the previous iteration. The mass flow is now corrected directly from the latest pressure solution [#76](@ref)

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.5.1] - 2025-07-03

### Added
*  New `Empty` boundary condition allowing 2D simulations with OpenFOAM 2D-compatible grids[#63](@ref)
*  Tests for the `Smagorinsky` LES model have been included [#63](@ref)

### Fixed
* No fixes included
  
### Changed
* Transient simulation results for `VTK` and `VTU` files use the format `time_<iteration>` [#63](@ref)

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.5.0] - 2025-06-28

### Added
* New boundary conditions `Extrapolated` and `Zerogradient` have been added. Both assign a zero gradient boundary condition, however, their implementation differs. `Extrapolated` assigns the zero gradient condition semi-implicitly (using the cell centre unknown and the cell centre value from the previous iteration). `Zerogradient` assigns the gradient in the boundary faces explicitly [#61](@ref)
* The workgroup size for the CPU backend can now be automatically chosen when `workgroup=AutoTune()`. This uses a simple ceiling division internally where the number of elements in the kernel are divided by the number of available threads. This results in a small 10% performance gain [#61](@ref)

### Fixed
* Fixed the implementation for the calculation of the wall distance to work on GPUs [#49](@ref)
* The new approach for handling user-provided boundary conditions now allows extracting the wall velocity specified by the user for the `Wall` boundary condition [#61](@ref)

### Changed
* In the calculation of wall function properties the user-provided wall velocity is now used, instead of hard-coded to no-slip [#49](@ref)
* The functions `set_production!`, `set_cell_value!` and `correct_nut_wall!`, in `RANS_functions.jl` have been updated, removing conditional branch used in the generated calling them. Now these functions use multiple dispatch to allow specialising the wall function framework to ease the development of new wall functions [#57](@ref)
* New `NeumannFunction` has been created, mirroring the DirichletFunction, providing a Neumann boundary condition defined with a user-provided struct (a generic framework accepting a function to set the gradient at the boundary is not yet available) [#57](@ref)
- `update_user_boundary` function, extension has been reverted, overwriting the changes made to expose the `ModelEquation` type to it in [#55](@ref)
* User-provided boundary conditions are no longer stored within fields, instead a `NamedTuple` is constructed using the function `assign`, which is passed to solvers using the `Configuration` struct. 
* Configuration setting provided by the user at the top-level API are now stored in predefined structs, instead of using `NamedTuples`, this change should put less pressure on the compiler [#61](@ref)
* Internally, kernel launches have been updated to use a static `NDrange`. This resulted in a 20% speed improvement on GPU backends (only NVidia GPUs tested) [#61](@ref)

### Breaking
* The definition of Krylov solvers in the previous API used types exported directly from `Krylov.jl`. Now solvers are defined using instances of types defined in `XCALibre.jl`. As an example, previously the CG solver was defined using the type `CgSolver` now this solver is defined using the instance `Cg()` where the suffix "Solver" has been dropped. This applies to all previously available solver choices [#60](@ref)
* The Green-Gauss method for calculating the gradient is now `Gauss` which is more descriptive than the previous name `Orthogonal`
* The internals for handling user-provided boundary conditions have been updated in preparation for extending the code for handling multiple regions. Thus, the syntax for assigning boundary conditions has changed. The most noticeable change is the removal of the the `@assign!` macro, replaced by the function `assign`. See the documentation for details [#61](@ref)
* The `set_solver`, `set_hardware` and `set_runtime` top-level functions have been replaced with `SolverSetup`, `Hardware` and `Runtime` structures. This allowed storing user-provided setup information in structs instead of `NamedTuples` to reduce the burden on the compiler [#61](@ref)
* The implementation of the `FixedTemperature` boundary condition has been simplified, resulting in a breaking change. The definition of the boundary condition now requires users to provide an `Enthalphy` model. This interface should make it easier to extend the current implementation to new forms of the energy equation [#61](@ref)

### Deprecated
* No functions deprecated

### Removed
* The functions `set_solver`, `set_runtime`, `set_hardware` and the macro `@assign!` have been removed/replaced [#61](@ref)

## Version [v0.4.2] - 2025-04-02

### Added
* A very simple 2D block mesh generator has been added (not ready for general use as it needs to be documented)[#41](@ref)
* Implementation of `Wall` boundary conditions specialised for `ScalarField` [#45](@ref)
* Simulation results can now be written to `VTK` and `OpenFOAM` formats. The format can be selected using the `output` keyword argument in the `run!` function. The formats available are `VTK()` and `OpenFOAM()` [#47](@ref)

### Fixed
* Fixed the implementation for the calculation of the wall distance [#45](@ref)

### Changed
* In preparation for hybrid RANS/LES models, the object `Turbulence` is now passed to the `TurbulenceModel` object to allow for a more general call of `turbulence!`. This changes the implementation of `turbulence!` for all models slightly [#46](@ref)
* Improved stability on meshes with warped faces by changing how face normals are calculated. XCALibre now uses an area-weighted face normal calculation based on the decomposition of faces into triangles [#47](@ref)
* Removed `VTK` module and moved functionality to a new `IOFormats` module [#47](@ref)
* Internally the function `model2vtk` has been replaced with `save_output` within all solvers. The specialisation for writing `ModelPhysics` models to file has also changed to `save_output`. The arguements pass to this function have also changed, the `name` of the file is not passed to the function, instead, the current runtime variable `time` is required [#47](@ref)
* The `Momentum` object, part of the `ModelPhysics` object, now includes the `FaceScalarField` and `FaveVectorField` in addition to existing fields in preparation for non-uniform boundary definition when using the `OpenFOAM` output format [#47](@ref)

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.4.1] - 2025-03-06

### Added
* No new functionality has been added

### Fixed
* Remove fragile precompile statements of the form `var"#*#*"` causing errors in Julia v1.12 [#40](@ref)

### Changed
* In preparation for implementation of hybrid models e.g. DES, the signature of the `turbulence!` function has been updated to use the `AbstractTurbulenceModel` [#39](@ref)
* New semi-implicit implementation for periodic boundary conditions (implicit treatment of laplacian terms)[#40](@ref)

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.4.0] - 2025-02-17

### Added
* Implementation of `Symmetry` boundary condition for `ScalarField` types [#30](@ref)
* New macro to help define boundary conditions that will dispatch to `Scalar` or `VectorField` types [#30](@ref)
* Added `eltype` method for both `Scalar` and `VectorField` types to simplify the development of new kernels where type information is needed [#30](@ref)
* New gradient limiters `FaceBased` and `MFaceBased` for limiting gradients based on cell faces, where `MFaceBased` is a multidimensional version, and it is generally recommended over `FaceBased` [#30](@ref)
* Support for INTEL hardware [#32](@ref)

### Fixed
* Calling `JacobiSmoother` now works on the GPU [#30](@ref)
* Implemented `SparseXCSR` as wrapper for `SparseMatrixCSR` on the CPU to resolve display/print errors [#30](@ref)
* The convergence criteria for solvers is used consistently to stop/control simulation runtime [#36](@ref)
* Consistent display of residuals on screen during simulations [#36](@ref)
* Fixed calculation of wall distance and `apply_boundary_conditions!` arguments for `LKE` model [#36](@ref)

### Changed
* The calculation of gradients has been improved by merging computations into a single kernel, improving performance of gradient kernels by around 10-30%, most noticable for vector gradients [#30](@ref)
* Improved calculation of non-orthogonal calculation (more tests are still needed), although tests have proven to be stable [#30](@ref)
* Improved documentation/readme on supported GPU backends/hardware and make users aware of potential `F32` limitation on some hardware

### Breaking
* The top level API for all solvers no longer takes the keyword argument `limit_gradient` for activating gradient limiter. New gradient limiters have been added and can be selected/configured when assigning numerical schemes with the `Schemes` function [#30](@ref)

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.3] - 2024-12-24

### Added
* Added experimental support for NVIDIA ILU0 and IC0 preconditioners [#23](@ref)
* Added `JacobiSmoother` that can be used with all linear solvers (improving initial guess) [#23](@ref)
* New function `activate_multithread` is available to set up matrix-vector multiplication in parallel on the CPU
* Initial benchmark added in the documentation

### Fixed
* No fixes

### Changed
* Internally the sparse matrix format has been changed to CSR. This has improved performance by 2x-4x (case dependent)
* Multithreaded sparse matrix vector multiplication is now functional [#23](@ref)
* Precompilation errors on Julia v1.11 addressed by bringing code from `ThreadedSparseCSR.jl` [#24](@ref)
* Update compat entry for `Atomix.jl` to v1.0,0 [#24](@ref)
- `DILU` preconditioner is now implemented to work with sparse matrices in CSR format and uses a hybrid approach (running on CPU) to allow working when using GPU backends (further work needed) [#26](@ref)
* The implementation of RANS models and wall functions has been improved for consistency (resulting in some computational gains). The calculation of `yPlusLam` is only done once when constructing the wall function objects. The calculation of the velocity gradient is now only done within the turbulence model main function (`turbulence!`) [#28](@ref)

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* The `ILU0` and `LDL` preconditioners has been temporarily removed [#23](@ref)

## Version [v0.3.2] - 2024-11-08

### Added
* Added support for CI testing, Dependabot, and CompatHelper

### Fixed
* Fixed tests for mesh conversion and standardised tolerances for test checks of incompressible solvers [#16](@ref)

### Changed
* No changes

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.1] - 2024-10-18

### Added
* Vastly improved documentation with new examples provided [#12](@ref)
* Changelog added to record changes more clearly. Record kept in [Release notes](@ref)

### Fixed
* The calculation of gradients can be limited for stability. This functionality can be activated by passing the key work argument `limit_gradient` to the `run!` function. The implementation has been improved for robustness [#12](@ref)
* Removed face information being printed when `Mesh` objects are created to stop printing a `ERROR: Scalar indexing is disallowed` message [#13](@ref)

### Changed
* Master branch protected and requires PRs to push changes

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.0] - 2024-09-21

* New name - XCALibre.jl - which is now registered in the General Julia registry
* Can do 3D and GPU accelerated simulations
* Can read .unv and OpenFOAM mesh files (3D)
* Can do incompressible and compressible simulations
* RANS and LES models available
* User-provided functions or neural networks for boundary conditions
* Reasonably complete "user" documentation now provided
* Made repository public (in v0.2 the work was kept in a private repository and could only do 2D simulations)
* Tidy up mesh type definitions by @mberto79 in #5
* Adapt code base to work with new mesh format by @mberto79 in [#6](@ref)
* Mesh boundary struct changes PR by @TomMazin in [#7](@ref)
* Mesh boundary struct changes PR fix by @TomMazin in [#8](@ref)


## Version [v0.2.0] - 2023-01-23

* New mesh format and type implemented that are GPU friendly.
* No functionality changes

## Version [v0.1.0] - 2023-01-23

### Initial release

2D implementation of classic incompressible solvers for laminar and turbulent flows:

* Framework for equation definition
* SIMPLE nad PISO algorithms
* Read UNV meshes in 2D
* Capability for RANS models
* Various discretisation schemes available
* Planned extension to 3D and GPU acceleration!