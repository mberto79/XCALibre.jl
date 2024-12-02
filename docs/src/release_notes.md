```@meta
EditURL = "https://github.com/github.com/mberto79/XCALibre.jl/blob/master/CHANGELOG.md"
```

# Release notes

The format used for this `changelog` is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Notice that until the package reaches version `v1.0.0` minor releases are likely to be `breaking`. Starting from version `v0.3.1` breaking changes will be recorded here. 

## Version [v0.3.3](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.3.3) - 2024-XX-XX

### Added
* Added experimental support for NVIDIA ILU0 and IC0 preconditioners
* Added `JacobiSmoother` that can be used with all linear solvers (improving initial guess).
* New function `activate_multithread` is available to set up matrix-vector multiplication in parallel on the CPU

### Fixed
* No fixes

### Changed
* Internally the sparse matrix format has been changed to CSR. This has improved performance by 2x-4x (case dependent)
* Multithreaded sparse matrix vector multiplication is now functional

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* The `ILU0` and `LDL` preconditioners has been temporarily removed

## Version [v0.3.2](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.3.2) - 2024-11-08

### Added
* Added support for CI testing, Dependabot, and CompatHelper

### Fixed
* Fixed tests for mesh conversion and standardised tolerances for test checks of incompressible solvers [#16](https://github.com/github.com/mberto79/XCALibre.jl/issues/16)

### Changed
* No changes

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.1](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.3.1) - 2024-10-18

### Added
* Vastly improved documentation with new examples provided [#12](https://github.com/github.com/mberto79/XCALibre.jl/issues/12)
* Changelog added to record changes more clearly. Record kept in [Release notes](@ref)

### Fixed
* The calculation of gradients can be limited for stability. This functionality can be activated by passing the key work argument `limit_gradient` to the `run!` function. The implementation has been improved for robustness [#12](https://github.com/github.com/mberto79/XCALibre.jl/issues/12)
* Removed face information being printed when `Mesh` objects are created to stop printing a `ERROR: Scalar indexing is disallowed` message [#13](https://github.com/github.com/mberto79/XCALibre.jl/issues/13)

### Changed
* Master branch protected and requires PRs to push changes

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.0](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.3.0) - 2024-09-21

* New name - XCALibre.jl - which is now registered in the General Julia registry
* Can do 3D and GPU accelerated simulations
* Can read .unv and OpenFOAM mesh files (3D)
* Can do incompressible and compressible simulations
* RANS and LES models available
* User-provided functions or neural networks for boundary conditions
* Reasonably complete "user" documentation now provided
* Made repository public (in v0.2 the work was kept in a private repository and could only do 2D simulations)
* Tidy up mesh type definitions by @mberto79 in #5
* Adapt code base to work with new mesh format by @mberto79 in [#6](https://github.com/github.com/mberto79/XCALibre.jl/issues/6)
* Mesh boundary struct changes PR by @TomMazin in [#7](https://github.com/github.com/mberto79/XCALibre.jl/issues/7)
* Mesh boundary struct changes PR fix by @TomMazin in [#8](https://github.com/github.com/mberto79/XCALibre.jl/issues/8)


## Version [v0.2.0](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.2.0) - 2023-01-23

* New mesh format and type implemented that are GPU friendly.
* No functionality changes

## Version [v0.1.0](https://github.com/github.com/mberto79/XCALibre.jl/releases/tag/v0.1.0) - 2023-01-23

### Initial release

2D implementation of classic incompressible solvers for laminar and turbulent flows:

* Framework for equation definition
* SIMPLE nad PISO algorithms
* Read UNV meshes in 2D
* Capability for RANS models
* Various discretisation schemes available
* Planned extension to 3D and GPU acceleration!