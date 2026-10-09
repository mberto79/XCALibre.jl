# NEW SECTION: mixed precision on distributed meshes

# PETSc loads one scalar type per process: a Float32 PETSc library solves A·d = b - Ax for the correction
# while fields stay in the mesh's float type. r and d are PETSc's b and x for that solve.
struct MixedPETScSolver{S,V,R,C} <: AbstractDistributedSolver
    petsc::S
    nzval::V          # Float32 copy of A; owned rows are written
    dinv::R           # Jacobi scratch written by the shared residual kernel (PETSc owns the PC)
    r::R
    d::R
    n_owned::Int
    config::C         # (; hardware) for sync! and kernel launches
end

_distributed_solver(::FullPrecision, eqn, dmesh, setup, config; kwargs...) =
    PETScSolver(eqn, dmesh, setup; kwargs...)

_distributed_solver(p::Solve.AbstractMixedPrecision, eqn, dmesh, setup, config; kwargs...) =
    throw(ArgumentError("$p on a distributed mesh: PETSc has no $(Solve._storage_type(p)) build; use MixedF32()"))

function _distributed_solver(::MixedF32, eqn, dmesh, setup, config; kwargs...)
    T = Float32
    (; backend) = config.hardware
    n = getfield(dmesh, :partition).n_owned
    nzval = similar(_nzval(_A(eqn)), T)
    copyto!(nzval, _nzval(_A(eqn)))
    petsc = PETScSolver(eqn, dmesh, setup; float_type=T, nzval, kwargs...)
    zeros_T() = KernelAbstractions.zeros(backend, T, n)
    MixedPETScSolver(petsc, nzval, zeros_T(), zeros_T(), zeros_T(), n, (; hardware=config.hardware))
end

# ghosts are refreshed first since r = b - Ax reads them
function _solve_owned!(deqn::DistributedEqn{E,<:MixedPETScSolver}, result, component) where E
    s, eqn = deqn.solver, deqn.eqn
    (; backend, workgroup) = s.config.hardware
    values = result.values
    sync!(result, get_phi(eqn).mesh, s.config)
    A = _A(eqn)
    kernel! = _sized(Solve._mixed_residual!, backend, workgroup, s.n_owned)
    kernel!(s.r, s.nzval, s.dinv, _rowptr(A), _colval(A), _nzval(A), values, _b(eqn, component))
    fill!(s.d, zero(eltype(s.d)))
    passemble!(s.petsc, s.nzval, s.r)
    psolve!(s.petsc, s.d)
    # Crank-Nicolson's explicit step 2x_new - x_old is x_old + 2d
    α = typeof(eqn.model.terms[1].type) <: Time{CrankNicolson} ? 2 : 1
    kernel! = _sized(Solve._add_correction!, backend, workgroup, s.n_owned)
    kernel!(values, s.d, eltype(values)(α))
    values
end
