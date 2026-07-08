# Architecture Review: unify serial + distributed solvers

Goal: one solver body runs serial and distributed. Delete the p-twins
(plaplace!/psimple!/ppiso!/PSIMPLE/PPISO). Parallelism sits below the
solver-author API; MPI never appears in a solver body.

## 1. Diff: p-twins vs serial (categorised divergences)

Loop structure of PSIMPLE==SIMPLE, PPISO==PISO, plaplace==LAPLACE line-for-line.
Only these seams differ:

(a) Halo exchange points (cell-field ghost fills; serial has none):
- before loop: sync U (w3), p (w1) prior to interpolate/flux/grad
- after grad!(∇p): sync ∇p.result (w3)
- after inverse_diagonal!(rD): sync rD (w1)
- after H!(Hv): sync Hv (w3)
- after p-solve + explicit_relaxation!: sync p (w1)
- inside distributed solve_system! (line 87): sync solved field after psolve!
- PPISO repeats the Hv/p/∇p syncs per inner loop
All are on CELL fields; processor faces are interior faces so face fields
(Uf, mdotf) never need exchange.

(b) Global reductions:
- residual: serial sum(R)/sum(Fx) over full arrays → distributed sums view(·,1:n_owned)
  then MPI.Allreduce(+) for num and den (Distribute_5 residual).
- courant: serial maximum(cellsCourant) → distributed MPI.Allreduce(max)
  (pmax_courant_number!).
- pnorm/pdot/pmean exist (Distribute_3) but solver bodies don't use them;
  they're for user diagnostics, not on the seam. Not load-bearing here.

(c) Linear solve: serial solve_system! = apply_smoother! + krylov_solve!
  (Krylov.jl) + residual. Distributed solve_system!(deqn,…) = passemble! +
  psolve! (PETSc, ext) + halo_exchange! + residual. Dispatched on the
  DistributedEqn wrapper type. passemble!/psolve! live in XCALibrePETScExt.

(d) Owned-row rules (ghost CSR rows are garbage by design):
- pmake_symmetric!: reads min(owner) row instead of ownerCells[1]
  (owner1 may be a ghost). Value identical since coeff symmetric.
- pcorrect_mass_flux!: same min/max canonical-row trick; p2-p1 unchanged.
- residual: owned rows only (see b).

(e) Reference pressure: setReference!(deqn,…) treats cellID as an ORIGINAL
  global id, maps to local via orig_cells, only owning rank edits its row.
  Serial setReference! uses the local id directly.

(f) Residual: see (b)+(d) — owned rows + Allreduce.

(g) I/O: pinit_writer/pmaybe_write_results + decomposed OF writer
  (processor<rank>/) vs initialise_writer/save_output. write_results already
  has a DistributedMesh method (Distribute_7).

(h) Other: preconditioner setup skipped (PETSc PC owns it); periodic BCs
  rejected; no ProgressMeter/postprocess; Laminar-only guard; converged @info
  gated on rank==0; HaloExchange objects (H1,H3) built in loop setup.

## 2. Key lever already in place

`run!` (Solvers_3) dispatches on Physics with domain D as a FREE type var, so
`D<:DistributedMesh` already routes to the SAME simple!/piso!/laplace!. Once
the bodies self-handle distribution, `prun!` is entirely redundant — delete it.

DistributedMesh forwards getproperty to the wrapped Mesh3, so fields/kernels
already treat it as a normal mesh. The wrapper is the natural dispatch key.

## 3. Minimal abstraction (function seams + file locations)

Principle: push each divergence into a low-level primitive that dispatches on
mesh/eqn type; serial method is an inlined no-op (zero-cost hard constraint).

Seam S1 — sync (halo). Preferred: SELF-SYNCING PRIMITIVES. Append
`sync!(result_cell_field, mesh, config)` to the tail of: grad! (∇p.result),
inverse_diagonal! (rD), H! (Hv), correct_velocity! (U), explicit_relaxation!
(p). Distributed solve_system! already syncs the solved field. Result: solver
bodies become IDENTICAL for serial/distributed with zero explicit sync calls.
- `sync!(x, ::Union{Mesh2,Mesh3}, config) = nothing` (inlined) — serial no-op.
- `sync!(x, dm::DistributedMesh, config)` in Distribute → halo_exchange! with a
  width-keyed HaloExchange cached on dm (build w1 and w3 in distribute()).
  Width chosen by field type (scalar=1, vector=3).
Fallback if redundant exchanges blow the perf gate: keep sync! but place it
explicitly at the ~6 body points (still no-op serial). Recommend self-syncing;
correctness-by-construction beats hand-placed syncs a new solver can forget.

Seam S2 — linear solve. Already dispatched via solve_system!(eqn,…). Introduce
a wrap factory in setup: `wrap_eqn(eqn, mesh, setup)` = identity (serial) /
DistributedEqn (DistributedMesh, in Distribute). Body calls generic
solve_equation!/solve_system!; distributed overrides already exist.

Seam S3 — residual + setReference!. Already dispatch on DistributedEqn.
No body change; keep the Distribute overrides.

Seam S4 — owned-row kernels. Fold min/max canonical row INTO the serial
kernels `_make_symmetric!` (Solve_1_api) and `_correct_mass_flux!`
(Solvers_1_SIMPLE). min/max is correct+cheap in serial (both owners real,
coeff symmetric). Deletes pmake_symmetric!/pcorrect_mass_flux! outright.

Seam S5 — courant reduction. In max_courant_number! (Solvers_0) wrap the final
maximum in `global_max(v, mesh)` = identity / Allreduce(max) in Distribute.
Kernel already mesh-dispatched. Deletes pmax_courant_number!.

Seam S6 — I/O. initialise_writer/save_output/write_results already dispatch on
mesh; add/keep DistributedMesh methods (mostly exist). Guard VTK-distributed.
Deletes pinit_writer/pmaybe_write_results.

## What serial files change (small)
- Solve_1_api.jl: `_make_symmetric!` → min/max (2 lines); add `sync!` no-op.
- Solvers_1_SIMPLE.jl: `_correct_mass_flux!` → min/max (2 lines); SIMPLE setup
  wraps eqns + guards preconditioner.
- Calculate_0_gradient.jl / Solvers_0_functions.jl: sync! tail on grad!,
  inverse_diagonal!, H!, correct_velocity!; global_max seam in courant.
- Solve_1_api.jl: sync! tail on explicit_relaxation! (or in body).
- No MPI import in any serial file; the DistributedMesh methods of every seam
  live in Distribute. Bodies stay MPI-free.

## Files deleted
- Distribute_6_simple.jl entirely (PSIMPLE/PPISO/psetup/pcorrect_mass_flux/
  pmake_symmetric/pmax_courant).
- Distribute_5: plaplace!/prun! deleted. KEEP DistributedEqn + its
  solve_equation!/solve_system!/residual/setReference! and PETScSolver — that
  IS the below-API parallel layer.
- Distribute_7 writer kept.

## 4. Migration phases (each gate = tests green + serial perf budgets flat)
- A: min/max fold in the two serial kernels + sync! no-op seam. Gate: full
  serial suite; serial Laplace/SIMPLE perf unchanged.
- B: self-sync tails in primitives + DistributedMesh sync!/halo cache in
  Distribute. Gate: test_perf.jl / test_scaling.jl alloc budgets; serial flat.
- C: wrap_eqn in setup_incompressible_solvers/setup_laplace_solver; guard
  preconditioner+writer; delete plaplace/psimple/ppiso; prun!→run!. Gate:
  test_psimple.jl, test_ppiso.jl, test_laplace.jl field-match <1e-6 vs serial.
- D: global_max courant seam; unify save_output for DistributedMesh. Gate:
  test_io.jl, ppiso courant path, test_gpu.jl.
- E: extend wrap_eqn to turbulence/energy eqns (hook in
  initialise(model.turbulence,…)); enable a non-Laminar distributed case.
  Gate: new turbulence-distributed test vs serial.

## 5. Risks / open questions
- @generated BC dispatch: BCs iterate boundary faces only; processor faces are
  interior faces → BC tuples never touch ghosts. Should stay untouched; verify
  no BC path reads ghost cells.
- KA kernels over ghost cells: bodies loop 1:length(cells) incl ghosts; ghost
  rows computed then overwritten by sync — harmless waste, avoids a branch.
  Restricting ndrange to n_owned needs mesh-dependent range; skip unless hot.
- Self-syncing width: grad vector result needs w3 halo; ensure both widths
  cached on dm and sync! selects by field type without allocation/instability.
- Type stability: sync!(x, mesh, config) must be concrete no-op methods on
  Mesh2/Mesh3; model.domain is a concrete type in the body → inferred. Keep
  @inferred residual/solve_system! green (test_perf).
- PETSc ext boundary: wrap_eqn(::DistributedMesh) needs PETScSolver, which
  errors without `using PETSc`; that's the intended load-time guard.
- Reference cell: cellID=1 must mean a deterministic global cell owned by
  exactly one rank; confirm orig_cells mapping is unique.
- Turbulence/energy generality: models "just work" distributed ONLY once their
  transported-scalar eqns pass through wrap_eqn (sync + owned-row residual).
  That wrap hook inside initialise is the single extension point for every
  future model — the whole point of the exercise.
- Redundant exchanges: self-syncing may add 1–2 exchanges/iter vs the hand-
  minimal p-twin. Cheap vs a PETSc solve; fall back to explicit body syncs only
  if a perf gate proves it matters.
