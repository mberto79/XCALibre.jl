# P1-M24 - binary output and restart (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R12. Governing decisions: D72.

## Problem, quantified

The decomposed writer is ASCII with one `println` per value (`Distribute_7_io.jl:249`); nothing reads fields back, so a run of days cannot be resumed. `gather` reconstructs on rank 0 (`:4-22`) and inherits the rank-0 ceiling.

## Approach

OpenFOAM binary format for mesh and fields (the reader already parses both), the face flux written as `phi`, and a `read_fields!` that restores exactly the state SIMPLE and PISO carry between iterations. Restart correctness is a bitwise question under Jacobi: the state is U, p, `mdotf`, the turbulence fields and, for PISO, `dt` and the previous time level.

## Configuration space

algorithm {SIMPLE laminar, PISO laminar, SIMPLE SST} x ranks {1, 2, 4} x format {binary}; verdict: resume at iteration 50 of 100 equals the straight run.

## Steps

- [x] **P1-M24-S1** DELIVERED (D130; ParaView substituted by foamToVTK, not installed): `format binary` in the decomposed writer for points, faces, owner, neighbour and every field, with `phi` (`mdotf`) written as a `surfaceScalarField` over internal then boundary faces - mechanism: OpenFOAM's own binary layout, which ParaView and `reconstructPar` read - cost: none - verdict: ParaView opens the case; `reconstructPar` completes; write time at 4 mm n=2 recorded against ASCII.
- [ ] **P1-M24-S2** `read_fields!(model, dir, time; comm)` reads each rank's `processor<rank>/<time>/` internal fields into the owned prefix, restores `mdotf` from `phi`, then syncs ghosts; `run!` gains `restart=<time>` - mechanism: the written state is the loop state - cost: none per iteration - verdict: SIMPLE laminar BFS resumed at 50 equals straight 100 to 1e-10 on the p residual at n=1,2,4; PISO the same with `dt` restored; SST the same with k, omega, nut, y restored.
- [ ] **P1-M24-S3** the guide documents checkpointing and restart, and `write_interval` semantics for it - mechanism: documentation - verdict: docs build green.

## Exit criterion

Restart test green for the three algorithms at n=1,2,4; write time recorded in `dev/telemetry/io.md`.

## Open questions

- Whether `y` (wall distance) is recomputed or read on restart; read is exact, recompute costs one Laplace solve. Read.
