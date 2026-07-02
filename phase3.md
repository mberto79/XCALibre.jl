# Phase 3 — PETSc assembly + SpMV verification (High)

Umbrella: `distributed_plan_detailed.md` §2.5–2.6; `distributed_plan.md` §6.
All PETSc code lives in `ext/XCALibrePETScExt.jl`; `Distribute` holds only the interface.

## Interface (in `Distribute_4_linalg.jl`)
```julia
abstract type AbstractDistributedSolver end   # already stubbed Phase 0/1
passemble!(s, eqn, partition)                 # local CSR owned rows → global matrix + b
psolve!(s, x)                                 # KSP solve, copy back into owned values
psolve_transpose!(s, x)                       # adjoint solve (Phase 7)
```
`prun!`/`solve_equation!` never talk to PETSc directly — only through these three.

## PETScSolver construction (once per equation; static mesh ⇒ static sparsity)
1. Exact preallocation: `d_nnz[i]` = owned-row entries with owned columns, `o_nnz[i]` =
   entries with ghost columns — count from local CSR (`_rowptr`,`_colval`) restricted to
   rows `1:n_owned`. `MatCreateAIJ(comm; type=mpiaij)` with the owned row block
   `row_start:row_end`.
2. `VecMPI` for b and x, same row layout. KSP created once, `KSPSetOperators`.

## passemble!
- For owned row i: global row `row_start + i - 1`; columns `local_to_global[colval[j]]`
  for `j in rowptr[i]:rowptr[i+1]-1`; values from `_nzval(_A(eqn))`. `MatSetValues`
  row-by-row (INSERT), then `MatAssemblyBegin/End`. Ghost rows of the local CSR are
  NEVER shipped (they are garbage by design, §2.3).
- After first assembly: values-only updates (same sparsity) — skip preallocation path.
- b: copy `_b(eqn, component)[1:n_owned]` into the Vec (host staging if GPU arrays).

## KSP mapping (curated + raw, per Q4)
- `Cg()→KSPCG`, `Bicgstab()→KSPBCGS`, `Gmres()→KSPGMRES`; `Jacobi()→PCJACOBI`;
  pressure default `PCGAMG`; `KSPSetTolerances(rtol, atol, dtol=PETSC_DEFAULT, itmax)`.
- Raw passthrough: `prun!(...; petsc_options="-ksp_view -pc_type hypre ...")` inserted
  into the global PETSc options DB before KSP setup.
- Float type: instantiate the petsclib matching `_get_float(mesh)` (Float32 lib exists —
  Phase 8 activates it; everything here written TF-generic).

## Tests (`test/distributed/test_assembly.jl`, n = 1, 2, 4)
- Assemble the diffusion operator from the Laplace unit-test case; compare
  `MatMult(A, x)` against serial `Fx .= A*x` gathered to rank 0 via `orig_cells`, ≤1e-12.
- Global row sums and symmetry match serial.
- KSP CG solve of that SPD system matches serial Krylov.jl solve to solver tolerance.
- Re-assembly (values-only) after perturbing coefficients gives the updated MatMult.

## Exit criteria
All green at n=1,2,4 under the MPI harness. n=1 PETSc result == serial XCALibre result
(same matrix, same answer) — proves the mapping before any multi-rank debugging.
