# GREENFIELD GPU AMG pipeline (Decision 0). Matrix-free fused multi-level V-cycle, one
# workgroup per macro-aggregate. Owns its own init/assembly/cycle; reuses only shared config
# types (AMG, AMGMatrixCSR) and the KA GPUBackend supertype from the reference module.
# Phase 1 = inert skeleton: nothing here is wired into solve_system!, so the live default
# path remains the reference implementation until the greenfield pipeline is functional
# (plan Phase 5d).

# Master gate: greenfield fused cycle is functional and validated (Cg parity, VRAM win). True now;
# the path is still OPT-IN via fuse_levels>0 (default 0), so the reference path stays the default.
_greenfield_implemented() = true

# Dispatch switch the solve_system! seam consults. Selected only on a GPU backend with fuse_levels>0
# AND a functional greenfield pipeline. The matrix-free cycle is weighted-Jacobi over Geometric
# (unsmoothed 1nz/row P), so it is gated to Geometric coarsening + an AMGJacobi smoother; any other
# configuration falls back to the reference materialized path. GPUBackend = KA's GPU supertype
# (CUDABackend/oneAPIBackend/ROCBackend), per Decision 0 — no new backend type.
_use_greenfield_amg(solver::AMG, backend) =
    _greenfield_implemented() && backend isa KernelAbstractions.GPU && solver.fuse_levels > 0 &&
    solver.coarsening isa Geometric && solver.smoother isa AMGJacobi
