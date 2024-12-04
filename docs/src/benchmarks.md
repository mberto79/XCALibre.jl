# Benchmarks

*Exploring the performance of `XCALibre.jl`*

## Introduction
---

This space will be used to explore the performance of various flow solvers and `XCALibre.jl` more broadly, and tests/benchmarks will be added as they become available. Eventually, our hope is to have benchmarks for all the solvers available in `XCALibre.jl`, but this will take time. Users are welcome to share their own benchmarks.

In general, it is accepted that benchmarking code is no easy task. Benchmarks that explore relative performance between different functionality within the same codebase tend to be easier, or more reliable. However, when comparing performance of "similar" functionality between different codebases is particularly challenging, and comparing the performance of different CFD codes is no exception. This is due to different implementation choices, difference in numerics, algorithm differences, etc., etc. Thus, it should be stressed that any results discussed herein should be accepted in this context. To enable some degree of transparency, our goal is to make all benchmark results and setup available to the community for scrutiny. Benchmark cases will be published in a separate Github repository, [`XCALibre_benchmarks`](https://github.com/mberto79/XCALibre_benchmarks). Any errors or methodology inconsistencies identified by the community can be identified, and we encourage user to report them.

Due to licensing considerations, it is not possible to report any direct benchmarks comparing performance between `XCALibre.jl` and commercial CFD codes. Thus, unless otherwise stated, the benchmarks included here are all likely to compare performance against [OpenFOAM (foundation)](https://openfoam.org/) or [OpenFOAM (ESI)](https://www.openfoam.com/). Specific versions used will be reported (and updated as far as possible). It should be stressed that the main `XCALibre.jl` developers are all OpenFOAM users and we hold this CFD package is very high regard, and as an established open-source CFD code it makes a perfect candidate to use in our benchmarking efforts, especially for ouf CPU implementation performance. 

This page will be updated as new results become available, but do bare with us, this exercise will take time. Thanks!

## Laminar incompressible solver (3D)
---

### Case and set up information

In this first benchmark, the performance of the *laminar solver* implementation in `XCALibre.jl` is explored. The configuration corresponded to a backward-facing step geometry with a mesh with approximately 500,000 cells. The test case mesh file, along with setup files for both `XCALibre.jl` and OpenFOAM can be found [here](https://github.com/mberto79/XCALibre_benchmarks/tree/main/3D_BFS_laminar).

The following cases were tested:

| Case    | Code | Version | Backend |Variable | Solver | Preconditioner |
| -------- | ----|---------|---------|---------|--------|---------------|
| OF11 Similar | OpenFOAM  | OpenFOAM 11 (foundation) | CPU | Velocity \\ Pressure | PBiCGStab \\ PCG | diagonal \\ diagonal |
| OF11 Best | OpenFOAM  | OpenFOAM 11 (foundation) |  CPU |Velocity \\ Pressure | PBiCGStab \\ GAMG | DILU \\ none |
| v0.3.2 | `XCALibre.jl` | v0.3.2 (Julia v0.10.6)  |  CPU |Velocity \\ Pressure | BiCGStab \\ CG | diagonal \\ diagonal |
| v0.3.3 | `XCALibre.jl` | v0.3.3 (Julia v0.10.6)  |  CPU |Velocity \\ Pressure | BiCGStab \\ CG | diagonal \\ diagonal |
| v0.3.2 GPU | `XCALibre.jl` | v0.3.2 (Julia v0.10.6)  |  GPU |Velocity \\ Pressure | BiCGStab \\ CG | diagonal \\ diagonal |
| v0.3.3 GPU | `XCALibre.jl` | v0.3.3 (Julia v0.10.6)  |  GPU |Velocity \\ Pressure | BiCGStab \\ CG | diagonal \\ diagonal |

Notice that the `OF11 Similar` case aims to provide a configuration in OpenFOAM that is similar to the numerical setup used (and available) in `XCALibre.jl`. The `OF11 Best` uses the most performant setup available in OpenFOAM, in terms of linear solver and preconditioner combination. Discretisation schemes and solver tolerance are kept as similar as possible for all cases. The `XCALibre.jl` test cases aim to provide relative performance between difference versions and backends.

### Hardware

All tests were performed on an ASUS TUF Gaming A15 FA506 with 16Gb of RAM, an AMD Ryzen™ 7 4800H CPU and. GPU test cases ran on the laptop's NVIDIA RTX 2060 with 6Gb of VRAM. Ubuntu 21.04 LTS was the operating system and it was set up to "performance" mode to ensure that power was reduced during tests (particularly important of GPU tests).

### Results

![Execution time](figures/benchmarks/laminar_3D_BFS/execution_time_comparision.svg)
![Parallel scaling](figures/benchmarks/laminar_3D_BFS/parallel_scaling.svg)
![Speed up comparison](figures/benchmarks/laminar_3D_BFS/speedup_vs_v0.3.2.svg)

