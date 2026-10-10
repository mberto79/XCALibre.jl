# NEW SECTION: mixed precision on distributed meshes

# PETSc loads one scalar type per process: a Float32 PETSc library solves A·d = b - Ax for the correction
# while fields stay in the mesh's float type. r and d are PETSc's b and x for that solve.
struct MixedPETScSolver{S,V,R,X,C} <: AbstractDistributedSolver
    petsc::S
    nzval::V          # Float32 copy of A; owned rows are written
    r::R
    d::R
    x0::X             # Crank-Nicolson's old owned values
    n_owned::Int
    atol::Float64
    rtol::Float64     # requested; PETSc solves each correction to at most Solve._attainable_rtol
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
    inner = @set setup.rtol = max(setup.rtol, Solve._attainable_rtol(MixedF32()))
    petsc = PETScSolver(eqn, dmesh, inner; float_type=T, nzval, kwargs...)
    zeros_T() = KernelAbstractions.zeros(backend, T, n)
    x0 = KernelAbstractions.zeros(backend, eltype(_nzval(_A(eqn))), n)
    MixedPETScSolver(petsc, nzval, zeros_T(), zeros_T(), x0, n, Float64(setup.atol), Float64(setup.rtol),
        (; hardware=config.hardware))
end

# as Solve.mixed_solve!: corrections until the global residual meets rtol, a single one unless rtol is
# below what Float32 attains
function _solve_owned!(deqn::DistributedEqn{E,<:MixedPETScSolver}, result, component) where E
    s, eqn = deqn.solver, deqn.eqn
    values = result.values
    # Crank-Nicolson's explicit step 2x_new - x_old is x_old + 2d
    α = typeof(eqn.model.terms[1].type) <: Time{CrankNicolson} ? 2 : 1
    s.rtol ≥ Solve._attainable_rtol(MixedF32()) && return _correct!(deqn, result, component, α, nothing)
    vo = view(values, 1:s.n_owned)
    α == 1 || copyto!(s.x0, vo)
    rnorm = _correct!(deqn, result, component, 1, 0.0)
    target = max(s.atol, s.rtol*rnorm)
    for _ ∈ 2:Solve.MAX_REFINEMENTS
        _correct!(deqn, result, component, 1, target) ≤ target && break
    end
    α == 1 || (vo .= 2 .* vo .- s.x0)
    values
end

# one correction unless the global ‖b - Ax‖ ≤ target (`nothing` skips the norm); returns that norm.
# Ghosts are refreshed first since r = b - Ax reads them
function _correct!(deqn, result, component, α, target)
    s, eqn = deqn.solver, deqn.eqn
    (; backend, workgroup) = s.config.hardware
    values = result.values
    sync!(result, get_phi(eqn).mesh, s.config)
    A = _A(eqn)
    kernel! = _sized(Solve._mixed_residual!, backend, workgroup, s.n_owned)
    kernel!(s.r, s.nzval, _rowptr(A), _colval(A), _nzval(A), values, _b(eqn, component))
    rnorm = isnothing(target) ? 0.0 : sqrt(MPI.Allreduce(Float64(sum(abs2, s.r)), +, _comm(deqn)))
    !isnothing(target) && rnorm ≤ target && return rnorm
    fill!(s.d, zero(eltype(s.d)))
    # PETSc keeps the matrix between corrections of one solve; s.r stays its right-hand side
    (isnothing(target) || iszero(target)) && passemble!(s.petsc, s.nzval, s.r)
    psolve!(s.petsc, s.d)
    kernel! = _sized(Solve._add_correction!, backend, workgroup, s.n_owned)
    kernel!(values, s.d, eltype(values)(α))
    rnorm
end
