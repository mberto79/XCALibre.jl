# distributed_plan_detailed.md — Detailed Phase Plan for the `Distribute` Module

Companion to `distributed_plan.md`. Incorporates Humberto's decisions and the
module/dispatch constraints. All function/field names verified against the current source.

## 1. Locked decisions

- Online partitioning only (rank 0 reads + scatters). Offline = Phase 8 stretch.
- Static meshes; no dynamic load balancing, no ParMETIS.
- Processor boundaries treated implicitly (see §2.3 — implemented as interior faces in the local mesh; a lightweight `ProcessorBoundary` patch record is kept for bookkeeping/IO only).
- PETSc options: curated keyword mapping AND raw option-string passthrough.
- PETSc.jl is the default distributed solver; HYPRE.jl later as a second extension.
- Julia ≥ 1.10 minimum (bump `[compat]`). No silent CPU fallback: if the user requests a GPU distributed run and CUDA-enabled PETSc / device path is unavailable → hard error with instructive message. (Host-staged MPI buffers are allowed — compute stays on GPU.)
- CI + initial development on CPU (`mpiexecjl -n 2/4`); GPU testing on local machines only.
- Float64 first; every `Distribute` type parameterised on `TF = _get_float(mesh)` so Float32 is a Phase 8 switch, not a redesign (PETSc.jl ships Float32 petsclibs).
- ALL distributed code lives in `src/Distribute/` (+ solver backends in `ext/`). Serial source files are not modified. Extension points are created only by overloading existing generic functions with methods dispatched on `Distribute` types.

## 2. Core architecture

### 2.1 The local-serial principle
Each rank holds a plain `Mesh3` containing its owned cells `1:n_owned` followed by one
layer of ghost cells `n_owned+1:n_owned+n_ghost`. All existing kernels — `discretise!`,
`apply_boundary_conditions!`, `grad!`, `interpolate!`, `flux!`, `div!`, relaxation,
`H!`, `correct_*` — run unmodified on this local mesh. Distributed awareness exists only in:
1. `DistributedMesh` (partition + halo metadata around the local `Mesh3`),
2. `halo_exchange!` calls at defined sync points,
3. the linear solve + global reductions (PETSc, `MPI.Allreduce`),
4. the orchestrating solver loops (`prun!` → `psimple!`/`ppiso!`/`plaplace!`).

### 2.2 User-facing workflow
```julia
using XCALibre, MPI, PETSc   # PETSc load activates ext/XCALibrePETScExt
MPI.Init()
dmesh = distribute(FOAM3D_mesh(file); comm=MPI.COMM_WORLD)  # rank0 reads+partitions+scatters
mesh_dev = adapt(backend, dmesh)                            # Phase 6
model = Physics(..., domain=dmesh)                          # per-rank, local partition
# BCs, schemes, solvers, config exactly as serial
residuals = prun!(model, config)
```
Primary design: `DistributedMesh{M<:Mesh3,P,H} <: AbstractMesh` with `getproperty`
forwarding to the wrapped local mesh, so `Physics`, field constructors, and kernels see
a normal mesh. Fallback (decided by a Phase 1 spike): if too many signatures are typed
`::Mesh3`, keep `DistributedMesh` as a plain container and pass it explicitly:
`prun!(dmesh, model, config)`.

### 2.3 Processor faces are interior faces (implicit coupling for free)
In the local mesh, a processor face keeps `ownerCells = [owned, ghost]` and lives in the
interior-face range (physical boundary faces stay at `1:n_bfaces`, preserving the mesh
invariant). Consequences:
- `discretise!` writes the off-diagonal coefficient `A[owned, ghost]` naturally — the
  implicit treatment Humberto approved, with zero new BC code and no `@define_boundary` entry.
- Face loops (`grad!`, `flux!`, `div!`, `nonorthogonal_face_correction`) are correct
  provided ghost cell values/gradients are current (halo sync points, §Phase 5).
- Ghost rows of the local CSR are garbage (incomplete stencils); only owned rows are
  shipped to PETSc, and residuals/reductions restrict to `1:n_owned`.
- A `ProcessorBoundary` record (neighbour rank, face list) is stored in
  `DistributedMesh` metadata only — not in `mesh.boundaries`, so BC assignment and
  `@generated` BC dispatch are untouched.

### 2.4 Fields
Model fields are plain `ScalarField`/`VectorField` on the local mesh — they ARE the
partition (`values` length `n_owned+n_ghost`). `Distribute` adds thin wrappers used
inside the distributed loops and for user-level reductions:
```julia
struct DistributedScalarField{F<:ScalarField,H}  field::F; halo::H end
struct DistributedVectorField{F<:VectorField,H}  field::F; halo::H end
sync!(df) = halo_exchange!(...)   # scalar: 1 buffer; vector: x,y,z (3-wide buffers)
```

### 2.5 Package layout
- `src/Distribute/Distribute.jl` — module, included from `XCALibre.jl`, reexported.
  Numbered files: `Distribute_0_types.jl`, `_1_partition.jl`, `_2_halo.jl`,
  `_3_fields.jl`, `_4_linalg.jl` (interface only), `_5_solvers.jl` (prun!, psimple!, ...).
- Hard deps added: `MPI`, `MPIPreferences`, `Metis` (all fine on every OS).
- `[weakdeps]`: `PETSc` → `ext/XCALibrePETScExt.jl` (default backend; keeps PETSc_jll
  off Windows serial users). Later `HYPRE` → `ext/XCALibreHYPREExt.jl`.
- `prun!` errors with "load PETSc.jl (`using PETSc`) to enable distributed solves" if no
  backend extension is active.

### 2.6 Complete overload budget (the "minimum dispatch" list)
Methods added by `Distribute`/extensions on existing generics — nothing else is overloaded:
- `adapt` for `DistributedMesh`, `HaloExchange`, `DistributedField`s (GPU move).
- `initialise!(::DistributedScalarField, v)` / vector variant (delegate + sync).
- `solve_system!(::DistributedEqn, setup, result, component, config)` → PETSc KSP.
- `solve_equation!(::DistributedEqn, ...)` (scalar + vector methods) — same orchestration
  as serial but distributed solve + owned-row residual.
- `residual(::DistributedEqn, component, config)` — owned rows + `MPI.Allreduce`.
- `setReference!(::DistributedEqn, pRef, cellID, config)` — owning rank only.
- `run!` is NOT overloaded; `prun!` is the new exported entry point (per Humberto).
Internal new names (not overloads): `distribute`, `sync!`, `halo_exchange!`,
`halo_exchange_adjoint!`, `pnorm`, `pdot`, `pmean`, `psimple!`, `ppiso!`, `plaplace!`.

---

## Phase 0 — Scaffolding & environment (Low, ~1–2 days)

Deliverables
- `src/Distribute/Distribute.jl` skeleton + `Distribute_0_types.jl` (Partition,
  NeighbourComm, HaloExchange, DistributedMesh, DistributedScalarField/VectorField,
  `abstract type AbstractDistributedSolver`, interface stubs `passemble!`, `psolve!`,
  `psolve_transpose!`).
- Project.toml: add MPI, MPIPreferences, Metis to `[deps]`; PETSc under
  `[weakdeps]`/`[extensions]`; `julia = "1.10"` compat.
- Empty `ext/XCALibrePETScExt.jl` that defines `PETScSolver <: AbstractDistributedSolver`.
- `test/distributed/runtests.jl` harness: launches `mpiexecjl -n N julia --project t.jl`
  from the serial test suite (precompile once serially first — MPI precompile race).
- CI: GitHub Actions job, CPU only, `-n 2` and `-n 4`.

Exit criteria
- `using XCALibre` unchanged for serial users (no MPI.Init needed, Windows still works).
- `using XCALibre, PETSc` loads the extension; `mpiexecjl -n 2` smoke test passes
  (each rank reports rank/nranks via a trivial `Distribute.hello()`).

## Phase 1 — Partitioning + local mesh construction (High, ~1–2 weeks; the critical phase)

Deliverables — `Distribute_1_partition.jl`
1. `build_dual_graph(mesh)::SparseMatrixCSC` — iterate interior faces
   (`fID > n_bfaces`), edge between `face.ownerCells[1]`/`[2]`.
2. `partition_cells(mesh, nparts)` — `Metis.partition(G, nparts; alg=:KWAY)`;
   log balance (max/min cells) and edge-cut.
3. `extract_subdomain(mesh, part, rank)` (runs on rank 0, per rank):
   - owned cells `O_r`; ghosts `H_r` = neighbours of `O_r` across cut edges (1 layer);
   - faces: every face incident to an owned cell; nodes of those cells/faces;
   - renumber cells owned-then-ghost; renumber faces physical-boundary-first
     (grouped per patch → rebuild `Boundary` `IDs_range` and `boundary_cellsID`),
     then interior faces (owned–owned and owned–ghost mixed);
   - rebuild all flat arrays + ranges: `cell_nodes/cell_faces/cell_neighbours/cell_nsign`
     (ghost cells get faces_range covering only their processor faces), `face_nodes`,
     `node_cells`; copy face geometry (centre, normal, e, delta, weight) verbatim from
     the global mesh — no recomputation, so serial/distributed geometry is bitwise equal;
   - record processor-face groups per neighbour rank (`ProcessorBoundary` metadata) with
     matching orderings on both sides (sort by global face ID so send/recv lists align).
4. Global block renumbering: new global cell numbering where rank r owns rows
   `row_start:row_end` contiguously; `local_to_global` (length n_owned+n_ghost),
   `owner` per local cell, permutation to original numbering kept for I/O.
5. `scatter_mesh` / `distribute(mesh; comm)` — rank 0 serializes each local mesh +
   Partition + processor metadata (`Serialization` into an `IOBuffer` → `MPI.send`);
   ranks deserialize → `DistributedMesh`. Setup-time only; simplicity over speed.
6. Spike: `DistributedMesh <: AbstractMesh` + `getproperty` forwarding — build a
   `Physics` + `ScalarField` on it; if `::Mesh3`-typed signatures block it, switch to
   the explicit `prun!(dmesh, model, config)` fallback (§2.2) and record the decision.

Tests (`test/distributed/test_partition.jl`, n = 1, 2, 4)
- `sum(n_owned) == ncells`; `local_to_global` restricted to owned cells is a bijection.
- Global volume sum over owned cells == serial total (to round-off).
- Every interior/physical face appears exactly once (owned-cell test); every processor
  face appears on exactly two ranks with identical centre/area/normal (sign-flipped).
- Per-patch physical boundary face counts sum to serial counts.
- n=1 degenerates to the serial mesh (same cells/faces up to renumbering).

## Phase 2 — Halo exchange + distributed fields (Medium-High, ~1 week)

Deliverables — `Distribute_2_halo.jl`, `Distribute_3_fields.jl`
1. Build `HaloExchange` from processor metadata: per neighbour `send_cells`
   (owned cells adjacent to that rank), `recv_ghosts` (ghost slots), preallocated
   send/recv buffers (backend arrays via `KernelAbstractions.allocate`), request vectors.
2. `pack!`/`unpack!`/`unpack_add!` KA kernels (as in `distributed_plan.md` §5).
3. `halo_exchange!(vals, H, backend, workgroup)` — Irecv-first, pack, sync, Isend,
   Waitall, unpack. Scalar and 3-component variants (vector fields, gradients:
   pack x,y,z into one 3-wide buffer per neighbour — one message, not three).
4. `halo_exchange_adjoint!` (reverse scatter with `unpack_add!`) — written now, used Phase 7.
5. Host-staging path: pinned host mirror buffers when `!MPI.has_cuda()`; selected once at
   construction (`cuda_aware` flag). CPU runs use device==host buffers directly.
6. `DistributedScalarField`/`DistributedVectorField` + `sync!`; `initialise!` overloads;
   `pnorm`/`pdot`/`pmean` over owned entries with `MPI.Allreduce`.

Tests (n = 2, 4, 8 CPU)
- Exchange cell centres: each ghost's received centre equals the neighbour's owned
  centre bitwise.
- Linear field f(x)=a·x+b: ghosts match analytic values exactly.
- Vector exchange: same via `VectorField`.
- `pnorm`/`pdot` match serial `norm`/`dot` on the gathered field to 1e-14.
- Repeat-call safety: two consecutive exchanges are idempotent (buffer reuse correct).

## Phase 3 — PETSc assembly + SpMV verification (High, ~1–1.5 weeks)

Deliverables — `Distribute_4_linalg.jl` (interface) + `ext/XCALibrePETScExt.jl`
1. `PETScSolver` construction from a `ModelEquation` + `Partition`:
   `MatCreateAIJ` (type `mpiaij`) with exact `d_nnz`/`o_nnz` preallocation computed once
   from the local CSR sparsity restricted to owned rows; `VecMPI` b/x; KSP created once.
2. `passemble!(s, eqn, partition)` — for owned row i: global row `row_start+i-1`, columns
   `local_to_global[colval[j]]`, values from `_nzval(_A(eqn))`; `MatSetValues` +
   `MatAssemblyBegin/End`. Sparsity static (static mesh) → values-only updates after
   first assembly. Copy `_b(eqn, component)` into the PETSc Vec (device↔host as needed).
3. KSP mapping from `SolverSetup`: `Cg()→KSPCG`, `Bicgstab()→KSPBCGS`, `Gmres()→KSPGMRES`;
   `Jacobi()→PCJACOBI`; pressure default `PCGAMG`; `KSPSetTolerances(rtol, atol, dtol, itmax)`.
   Raw passthrough: `SolverSetup` gains nothing — instead `prun!(...; petsc_options="...")`
   inserts into the global PETSc options DB before KSP setup (curated + raw, per Q4).
4. Copy-back: KSP solution Vec → owned entries of `phi.values`, then `sync!`.

Tests (n = 1, 2, 4)
- Assemble the diffusion operator from `test/unit_test_laplace.jl`'s case; compare
  `MatMult(A, x)` against serial XCALibre SpMV (`Fx .= A*x`) gathered to rank 0, ≤1e-12.
- Row sums / symmetry of the assembled global matrix match serial.
- KSP CG solve of that SPD system matches a serial Krylov.jl solve to solver tolerance.

## Phase 4 — Distributed Laplace solver: `plaplace!` (Medium, ~1 week)

Deliverables — first end-to-end distributed solve, in `Distribute_5_solvers.jl`
1. `DistributedEqn` wrapper (eqn + PETScSolver + partition + halo) created at setup.
2. Overloads: `solve_system!`, `residual` (owned rows, `Allreduce` of `sum(R)` and
   `norm(b)²`), `setReference!` (apply nzval/b edit only on the rank owning the global
   reference cell).
3. `plaplace!(model, config)` mirroring serial `laplace!`: discretise → BCs → distributed
   solve → sync → output.
4. `prun!(model, config; ...)` dispatch skeleton (keys on `model` like serial `run!`).
5. Per-rank VTK output: rank-suffixed files + rank-0 `.pvtu` master (minimal version).

Tests (n = 1, 2, 4, 8)
- 3D box diffusion: per-cell solution vs serial `laplace!` (gathered via
  `local_to_global`), relative L2 < 1e-8; identical across rank counts.
- Residual histories match serial to tolerance; reference-cell pinning consistent.

## Phase 5 — Incompressible SIMPLE/PISO: `psimple!`, `ppiso!` (High, ~2–3 weeks)

Approach: `psimple!` is a copy of `SIMPLE` (`Solvers_1_SIMPLE.jl`) living in `Distribute`,
with sync calls inserted and equations wrapped as `DistributedEqn`. Accepted duplication —
keeps serial files untouched per the module-isolation constraint. Maintenance note: any
upstream change to `SIMPLE`/`PISO` must be mirrored; revisit shared-hook refactor post-v1.

Sync points (from the actual SIMPLE body; each is `sync!` = one halo exchange)
1. After `initialise!`/start: sync U, p before initial `interpolate!`/`grad!`.
2. After momentum `solve_equation!` → sync U (H! reads neighbour U through off-diagonals).
3. After `inverse_diagonal!` → sync rD (ghost diagonals are garbage locally) before
   `interpolate!(rDf, rD, ...)`.
4. After `H!(Hv, ...)` → sync Hv before `interpolate!(Uf, Hv, ...)`.
5. After pressure solve + `explicit_relaxation!` → sync p.
6. After `grad!(∇p, ...)` → sync ∇p (3-component) — needed by `limit_gradient!`,
   `nonorthogonal_face_correction`, and face interpolation of gradients.
7. Inside each non-orthogonal corrector: repeat 5–6.
8. `correct_mass_flux!` needs no extra sync (p synced; `aN = A[owned, ghost]` is in the
   local CSR). `correct_velocity!` is per-owned-cell — no sync.
9. Convergence check: residuals already global (Phase 4 overloads); `MPI.Allreduce(&&)`
   not needed since every rank computes identical global residuals.

Scope
- v1 turbulence: `RANS{Laminar}` only (its `turbulence!` is a no-op). KOmega/LES need the
  same treatment inside `ModelPhysics/Turbulence` solves — explicitly deferred to v1.1
  with a clear error if a non-laminar model reaches `prun!`.
- Periodic BCs across partition boundaries: unsupported in v1 — detect and error at
  `distribute` time (Metis is free to cut a periodic pair).
- `ppiso!` after `psimple!` converges: same sync map + transient loop.

Tests (n = 1, 2, 4, 8 CPU)
- Lid-driven cavity + 2D backward-facing step (existing verification cases): fields match
  serial within 1e-6 relative; centreline profiles and integral quantities match;
  same converged solution independent of rank count.
- Residual histories vs serial: near-identical iteration counts (differences only from
  Krylov-vs-KSP arithmetic; assert same order and monotonicity).
- Transient: `ppiso!` cavity spin-up matches serial time history.

## Phase 6 — Multi-GPU + CUDA-aware MPI (High, ~1–2 weeks, local machines only)

Deliverables
1. `adapt(backend, dmesh)` — adapts local mesh, halo index arrays, buffers; rank→device
   binding `CUDA.device!(rank % ndevices)` in `distribute`/init helper.
2. CUDA-aware path: device buffers straight into `Isend/Irecv!` when `MPI.has_cuda()`;
   else pinned-host staging (already built, Phase 2).
3. PETSc GPU: `-mat_type mpiaijcusparse -vec_type cuda` when backend is `CUDABackend()`.
   Per Q6, NO silent fallback: if CUDA PETSc unavailable → error naming the fix
   (env vars, system PETSc, or explicitly request host solve via
   `prun!(...; solve_on=CPU())` — device↔host copies of A/b each solve, opt-in only).
4. Optional (time-permitting): interior-first overlap — assemble/compute interior faces
   while halo messages fly, process processor faces after `Waitall`.

Tests (local 2-GPU machine; not in CI)
- Cavity: multi-GPU result matches CPU-distributed and serial to 1e-5 (Float64 fields).
- Host-staging vs CUDA-aware produce identical results.
- Sanity scaling: 2 GPUs faster than 1 on a large-enough mesh; identical iterations.

## Phase 7 — AD / adjoint boundary (High, ~1–2 weeks)

Deliverables
1. `ChainRulesCore.rrule` for `halo_exchange!` (pullback = `halo_exchange_adjoint!`,
   ghost cotangents accumulated into owners).
2. `psolve_transpose!` via `KSPSolveTranspose`; `rrule` for the distributed solve:
   `b̄ = solve(Aᵀ, x̄)`, `Ā = -b̄ ⊗ x` (lazy, restricted to the local sparsity).
3. `rrule`s for `pnorm`/`pdot`/`pmean` (Allreduce of the seed).

Tests (n = 2, 4)
- Adjoint identity `⟨v̄, Hx⟩ == ⟨Hᵀv̄, x⟩` to 1e-12.
- Gradient of a functional on a small distributed diffusion case vs serial AD reference
  (1e-6); identical across rank counts; FD spot-check on 1–2 design variables.

## Phase 8 — Float32, HYPRE extension, I/O + docs (Medium, ~1–2 weeks)

1. Float32: instantiate PETSc Float32 petsclib keyed on `_get_float(mesh)`; verify halo
   buffers/reductions are TF-generic (they are by construction); cavity test in Float32
   vs Float64 to loose tolerance. Mixed-precision (F32 fields + F64 solve) noted as
   future work.
2. `ext/XCALibreHYPREExt.jl`: `HYPRESolver <: AbstractDistributedSolver` via IJ
   assembler; BoomerAMG for the pressure Poisson. Selection:
   `SolverSetup(...; solver=...)` untouched — backend chosen by which extension is loaded
   plus a `prun!(...; linear_backend=:petsc|:hypre)` keyword.
3. Offline partitioning: `partition_mesh(meshfile, nparts; dir)` writing per-rank JLD2
   local meshes + Partition; `distribute(dir; comm)` loads them in parallel. Same
   `DistributedMesh` afterwards → identical solutions test.
4. I/O polish: proper `.pvtu` (and OpenFOAM decomposed-case writer if cheap); gather
   utility `gather(field, dmesh)` for postprocessing on rank 0.
5. Docs: user guide page (workflow §2.2), cluster setup (MPIPreferences system binary,
   CUDA-aware env vars), CI notes, limitations list (laminar-only v1, no cross-partition
   periodics, Windows serial-only).

---

## Carried risks (unchanged from distributed_plan.md §11, with owners per phase)
- CUDA-enabled PETSc_jll availability → Phase 6 gate; opt-in host solve, never silent (Q6).
- `Mesh3`-typed signatures blocking `DistributedMesh <: AbstractMesh` → Phase 1 spike.
- psimple!/SIMPLE drift → review checklist item; refactor to shared hooks post-v1.
- Ghost-layer width: 1 layer assumed (Gauss gradient, compact stencils); halo width kept
  as a constructor parameter; least-squares gradients would need 2 → error if requested.
- MPI precompile race / UCX quirks → serial precompile step in test harness + docs.
