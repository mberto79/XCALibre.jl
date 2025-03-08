# Release notes

The format used for this `changelog` is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Notice that until the package reaches version `v1.0.0` minor releases are likely to be `breaking`. Starting from version `v0.3.1` breaking changes will be recorded here. 

## Version [v0.4.2] - 2025-xx-xx

### Added
* A very simple 2D block mesh generator has been added (not ready for general use as it needs to be documented)[#41](@ref)

### Fixed
* No fixes have been added

### Changed
* No changes have been made

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
* Remove fragile precompile statments of the form `var"#*#*"` causing errors in Julia v1.12 [#40](@ref)

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
* The top level API for all solvers no longer takes the keyword argument `limit_gradient` for activating gradient limiter. New gradient limiters have been added and can be selected/configured when assigning numerical schemes with the `set_schemes` function [#30](@ref)

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
* `DILU` preconditioner is now implemented to work with sparse matrices in CSR format and uses a hybrid approach (running on CPU) to allow working when using GPU backends (further work needed) [#26](@ref)
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