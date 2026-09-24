# P1-M35 - behaviour changes documented or fixed (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R14. Governing decisions: D175, D198, D208. Source: `dev/archive/reviews/p1/pre-merge-review-2026-09-24.md` § Behaviour changes (B1-B7). Diff base is `origin/main` (local `main` is one merge behind: fetch first).

## Problem, quantified

- B1: hand-written BC functors typed `cell::Cell{TF}` and user `scheme!` methods reading `face.weight` now hit MethodError; CHANGELOG files it under Changed and says definitions "compile unchanged" (true only for `@define_boundary` bodies).
- B2: `KernelAbstractions.get_backend(mesh.cells)` throws on `ElementArrays`; `propertynames(mesh)` omits `cells`/`faces`/`nodes`; `cell_nsign` is `Int8` and dispatch on `Mesh2{...}` parameters changed, undocumented.
- B3: `config.postprocess` is silently skipped on distributed meshes (`src/Solvers/Solvers_1_SIMPLE.jl:268-288`, `Solvers_2_PISO.jl:216-220`).
- B4: `activate_multithread` default: origin/main #161 made it `Threads.nthreads()`; this branch returns it to 1 (D198) and removed main's CHANGELOG line; D198's "unreleased" premise was wrong (D208).
- B5: serial OpenFOAM writer writes `dimensions [0 0 0 0 0 0 0]` for every field (`src/IOFormats/OpenFOAM/OpenFOAM_writer.jl:333`).
- B6: internal names exported: `_check_index_capacity`, `_with_index_capacity`, `_sized`, `_index_type` (and `passemble!`, `halo_exchange_adjoint!`, `decompose`, `gather` from Distribute).
- B7: CHANGELOG/release notes `[#160]`/`[#161]` placeholders point at merged PRs (D137); PISO end-of-run `@time` print removed while FilmModel/Multiphase keep it; `docs/src/contributor_guide.md:69-73` describes `Mesh3.Faces[10]`/`Mesh3.nsign`; docs env pulls PETSc; MPI/Metis now hard deps (say so); `_foam_binary` reads whole ASCII files to test the header (`src/FoamMesh/FoamMesh_1_read.jl:199`); dead `src/precompile.jl` uses 1.11-only `get_bool_env`; no Julia 1.10 CI job; CHANGELOG and `release_notes.md` differ on the `progress` entry.

## Approach

Code fixes where the old behaviour was reasonable and cheap to keep (B2 `get_backend` + `propertynames`, B3 one-time `@warn`, B5 per-field dimensions, B6 un-export, B7 `_foam_binary` header read, dead file); documentation for the intended changes (B1, B2 Int8/dispatch, B4, MPI/Metis). `docs/src/release_notes.md` mirrors CHANGELOG entry for entry.

## Configuration space

Not a numerical change except B5 (writer output) and B7 reader: verify on the test grids and one OpenFOAM case readable by `foamDictionary`/`checkMesh` if available, else by the reader round trip.

## Steps

Expected 3 steps.

- [ ] **P1-M35-S1** code: B2 (`KernelAbstractions.get_backend(x::ElementArrays) = get_backend(getfield(x, 1))`, `Base.propertynames` for Mesh2/Mesh3 including the views), B3 (`@warn` once when `postprocess` is non-empty on a distributed mesh, from rank 0), B5 (dimensions per field: U `[0 1 -1 0 0 0 0]`, kinematic p `[0 2 -2 0 0 0 0]`, k `[0 2 -2 ...]`, omega `[0 0 -1 ...]`, nut `[0 2 -1 ...]`, T `[0 0 0 1 0 0 0]`, unknown fields dimensionless), B6 (drop underscore exports; qualify internal uses), B7 (`_foam_binary` reads only the header bytes; delete `src/precompile.jl` if nothing includes it) - mechanism: restore main's behaviour where nothing required the change - cost: none per iteration - verdict: `test_mesh_conversion.jl`, `test_io.jl` (n=2), a new unit check for `get_backend(mesh.cells)`/`propertynames`, OpenFOAM writer round trip; full serial suite at close.
- [ ] **P1-M35-S2** docs: CHANGELOG Breaking gains B1 (custom BC/scheme signature, with a before/after snippet) and B2's Int8/type-parameter notes; Changed gains B4 (BLAS default back to 1, why: Julia-thread Krylov) and MPI/Metis hard deps; distributed guide notes B3; `contributor_guide.md` mesh text rewritten; `release_notes.md` synced to CHANGELOG; PISO `@time` either restored or the removal noted - verdict: docs build green, `grep` finds no `Mesh3.Faces`/`nsign` stale text.
- [ ] **P1-M35-S3** CI: add a Julia 1.10 job to the test matrix (`.github/workflows`), docs env without PETSc if no page executes PETSc code - verdict: workflow file parses (`act`-free check: YAML lint), job list shows 1.10.

## Exit criterion

B1-B7 each fixed or documented; docs build green; serial suite green.

## Open questions

- B7 PISO `@time`: restore for parity or remove from FilmModel/Multiphase too? User preference; default: remove everywhere (progress bar already reports time).
- B6: are `decompose`/`gather`/`passemble!` used by users of the distributed docs? Check `docs/src/user_guide/6_distributed_mpi.md` before un-exporting.
