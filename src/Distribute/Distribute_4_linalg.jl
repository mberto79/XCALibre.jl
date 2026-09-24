export PETScSolver

# implemented in XCALibrePETScExt; petsc_options is a string or a named tuple keyed by label
PETScSolver(args...; kwargs...) =
    error("PETScSolver requires the PETSc extension: add PETSc to your environment and `using PETSc`")

# copies owned rows of the local CSR and RHS into the solver's global system (values only)
function passemble! end

# solves with guess x[1:n_owned] and writes the solution back into the owned entries
function psolve! end

# adjoint solve
function psolve_transpose! end
