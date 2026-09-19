# P1-M22 - fused, overlapped, stream-aware communication (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, Q3. Governing decisions: D59, D72, D73.

## Problem, quantified

Per laminar SIMPLE iteration (`dev/archive/reviews/p1/audit-2026-09-18.md` § change 3): 8 blocking exchange rounds (three width-1 for U inside each component's `solve_system!`, `rD`, `Hv`, p after solve, p after `explicit_relaxation!`, `∇p`), 8 residual all-reduces (`Distribute_5_solvers.jl:88-89`), and on CUDA 12 `device_synchronize` (three per component solve, `XCALibrePETScExt.jl:195-233`) plus two stream syncs per halo (`Distribute_2_halo.jl:126`,`:138`). `wrap_eqn` builds a `HaloExchange` per equation (`:27`) beside the mesh's cached w1/w3; every message uses `tag=0`. Under `-log_sync` the solve is balanced (D59), so what remains at n=8 is arrival spread at reductions. GPU iteration is 0.0746 s at 500k cells n=1.

## Approach

Fewer rounds first (pure dependency analysis, no new machinery), then one shared schedule with persistent requests, then a two-phase `sync_begin!`/`sync_end!` measured at two body points, then stream events on the GPU. Each step keeps residuals bitwise identical under Jacobi; the M19-S4 counters and `check_ghosts` are the guards. Overlap that shows no gain at n=8 here is withdrawn, not kept on faith (D73 leaves its HPC re-test to P2).

## Configuration space

ranks {2, 8} CPU pinned (`dev/scripts/equal_thermal.sh` NS="2 8") x GPU {n=1, n=2 conda ompi} x case {BFS 5 mm laminar, cavity SST for the turbulence rounds}; metrics: rounds and all-reduces per iteration (M19-S4), per-iteration time, residuals.

## Steps

- [x] **P1-M22-S1** LANDED (D110) one schedule per mesh: `HaloCache` becomes a concrete parametric struct built on first use per backend; `wrap_eqn` stores no halo and `DistributedEqn` reads `dm.halos`; each schedule gets its own tag base (width 1: 10, width 3: 30, width 4: 40) - mechanism: one neighbour list, one buffer set, unambiguous tags - cost: none - verdict: `test_perf.jl` halo allocations unchanged or lower; residuals bitwise identical.
- [x] **P1-M22-S2** LANDED (D111) the vector `solve_equation!` in `Distribute_5` solves x, y, z with `passemble!`/`psolve!` only, then one width-3 `sync!(psi)`, then the three residuals. The local residual sums `(num, den)` of each component are taken immediately after ITS solve against ITS assembled matrix (the next component's `update_equation!` and relaxation overwrite `nzval` and `b`); only the ghost sync and the all-reduce are deferred, and the residual kernel of component c must read ghost values of c that the deferred exchange has filled, so the local sums are computed after the width-3 exchange on a per-component snapshot of `(nzval, b)` or, cheaper, `residual` is split into a local kernel run after each solve with that component's ghosts synced by a width-1 exchange that S2 keeps for x and y and drops for z only if the counters show a gain; choose by measurement and record it - mechanism: the y and z solves read owned rows and boundary faces only, never x ghosts - cost: none - verdict: rounds 8 to 6 (or 7 if x and y keep their exchange); residuals bitwise identical at n=2,8; `check_ghosts` zero after the U solve.
- [x] **P1-M22-S3** LANDED (D112) `residual` returns local `(num, den)`; a `residuals!(deqns...)` helper all-reduces one vector for U's three components (and p when called after its solve), so the iteration issues one 6-element and one 2-element reduction instead of eight - mechanism: sums commute; the reduction count is the latency - cost: none - verdict: all-reduces 8 to 2; residuals identical to the last bit (same summands, same order within each rank).
- [x] **P1-M22-S4** LANDED (D113) a width-4 exchange for `(rD, Hv)`: `sync!` accepts a tuple of fields packed into one buffer; the self-sync tails leave `inverse_diagonal!` and `H!`, and the SIMPLE and PISO bodies call `sync!((rD, Hv), mesh, config)` after `H!` - mechanism: both read only the momentum matrix and the already-synced U - cost: none - verdict: rounds 6 to 5; `check_ghosts` zero on `rD` and `Hv`; serial bodies unchanged (identity `sync!`).
- [x] **P1-M22-S5** LANDED (D114) persistent requests: `MPI.Send_init`/`Recv_init` per neighbour at schedule build, `MPI.Start`/`Startall` per exchange (MPI.jl exposes them, checked 2026-09-18); the Irecv for the next exchange of the same schedule is started right after unpack - mechanism: request set-up leaves the per-iteration path - cost: none - verdict: `test_perf.jl` halo allocation budget lowered to the pack/unpack launches; per-iteration time at n=8 within noise or better.
- [-] **P1-M22-S6** WITHDRAWN (D115); P2 re-measures on the HPC — `sync_begin!`/`sync_end!` on the schedule (pack, start sends and receives; wait, unpack), with `partition` carrying `halo_cells` (owned cells with a ghost neighbour) and `interior_cells`; applied at two body points only, `grad!` of p (green-gauss interior between begin and end, halo cells after) and `H!` - mechanism: interior work hides the round - cost: two kernel launches per point instead of one - verdict: accepted if n=8 pinned per-iteration time improves by at least 3 percent with residuals bitwise identical; otherwise WITHDRAWN with the number, the API kept for P2.
- [-] **P1-M22-S7** WITHDRAWN (D116) — GPU handoff by events: XCALibre records a `CuEvent` on its stream and PETSc's stream (`PetscDeviceContextGetCurrentContext` + `PetscDeviceContextGetStreamHandle`, both wrapped) waits on it through `cuStreamWaitEvent`; after `KSPSolve` the reverse; the three `device_synchronize` per solve go, and the second `KernelAbstractions.synchronize` in `halo_exchange!` goes (the unpack is stream-ordered before the next kernel) - mechanism: stream ordering replaces device drains - cost: two events per solve - verdict: GPU 5 mm n=1 per-iteration time recorded against 0.0746 s; `test_gpu.jl` n=1,2 green ten times under `compute-sanitizer` (racecheck on the halo kernels).
- [ ] **P1-M22-S8** (from P1-M21-S1, D92) the cell kernels that run over ghost rows (`discretise!`, `green_gauss!`, `div!`, `inverse_diagonal!`, `H!`) take their range from `n_owned` on a `DistributedMesh`; the CSR stays square - mechanism: ghost values arrive by halo exchange, so their rows are never used - cost: none - verdict: residual hashes bitwise equal at n=2 and n=8, `check_ghosts` zero, n=8 per-iteration time recorded (ghosts are 2.9 percent of rows at n=8, 10.8 at n=64 on 5 mm); withdrawn if the time does not move.

Every verdict run in this plan fits the five-minute cap (D101): use 10 mm for bitwise hashes and n=8 only where the language server is closed; one A/B pair per timing run.

## Exit criterion

Counters: 5 rounds and 2 all-reduces per laminar SIMPLE iteration (budget lowered by decision); n=8 pinned and GPU n=1 per-iteration times recorded in `dev/telemetry/communication.md`; residuals bitwise identical under Jacobi at every configuration; S6 accepted or withdrawn with its number.

## Open questions

- S7: ANSWERED (D116) - one global context on the legacy NULL stream.
- S6: moot, withdrawn (D115); PISO keeps separate rD and Hv exchanges (D113).
