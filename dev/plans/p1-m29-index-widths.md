# P1-M29 - every stored index at its narrowest safe width

Milestone row: `dev/phaseRoadmap.md`. Source: `dev/archive/reviews/p1/memory-scaling-2026-09-23.md` § index width and array inventory. Expected 4 steps (D159).

## mechanism

Local indices are signed and follow the mesh index type `TI`; Int32 caps one process near 300M cells (matrix `rowptr` ≈ 7 × cells), far above what one process holds in memory, while unsigned types save no bytes and wrap on subtraction and mismatch PETSc, METIS and cuSPARSE. Global IDs are the only quantity that outgrows Int32 first, so they carry their own Int64 type. Arrays that hard-code `Int` in stored solver state follow the finest matrix's index type. Floats are not narrowed (an accuracy question, not a layout one).

## bars (ranked)

- STRICT: CPU residuals bitwise equal to the preceding step at a fixed thread count; distributed hashes bitwise at n=2,4 (R8); a partition whose global IDs exceed `typemax(Int32)` builds with an Int32 mesh.
- Banded: AMG case time within noise or faster; no change to non-AMG timings.

## gate

Named test files the step reaches, each a separate command; `gate.jl` for distributed steps; the `run_psolver_study.sh` AMG point for S2, one point per command.

## steps

- [x] P1-M29-S1 LANDED (D191): global IDs Int64: `local_to_global`, `orig_cells`, `orig_faces` get their own type parameter, decoupled from the mesh's `VI`; test with synthetic offsets above `typemax(Int32)`. Blast radius: distributed setup, writer, restart. Bar: strict class, `gate.jl`, `test_restart.jl`.
- [ ] P1-M29-S2 AMG hierarchy index arrays (`I`, `J`, `diag_index`, `marker`, aggregation maps, `AMGMatrixCSR` `rowptr`/`colval`) built in the finest matrix's index type; matrix-free path made consistent. Blast radius: `AMG()` users. Bar: AMG unit tests, AMG case residuals bitwise.
- [ ] P1-M29-S3 periodic BC maps (`face_map`, `faceAddress1/2`, `i`/`j`) follow `TI`; one sweep of `src/` for `zeros(Int`, `Int64[`, `Int[`, `Vector{Int}` stored on a mesh, equation or solver struct. Blast radius: periodic cases. Bar: periodic tests bitwise.
- [ ] P1-M29-S4 `integer_type=Int32` default for mesh readers, with a clear error when a per-process face count or matrix nnz exceeds `typemax(Int32)`; `Int64` stays selectable. Blast radius: every reader and the whole suite. Bar: full serial suite by file, docs build.
