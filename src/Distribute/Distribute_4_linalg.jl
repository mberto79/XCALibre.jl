export PETScSolver, passemble!, psolve!, psolve_transpose!

"""
    PETScSolver(eqn, dmesh::DistributedMesh, setup; comm=MPI.COMM_WORLD, petsc_options="")

Create a distributed PETSc solver (matrix, vectors and KSP) for `eqn` on `dmesh`.
Implemented in the `XCALibrePETScExt` extension — requires `using PETSc`.
"""
PETScSolver(args...; kwargs...) =
    error("PETScSolver requires the PETSc extension: add PETSc to your environment and `using PETSc`")

"""
    passemble!(s, eqn, partition; component=nothing)

Copy the owned rows of the local CSR matrix and RHS of `eqn` into the global
distributed system held by solver `s` (values-only update; sparsity is static).
"""
function passemble! end

"""
    psolve!(s, x)

Solve the assembled distributed system with initial guess `x[1:n_owned]`, writing the
solution back into the owned entries of `x`.
"""
function psolve! end

# adjoint solve (Phase 7)
function psolve_transpose! end
