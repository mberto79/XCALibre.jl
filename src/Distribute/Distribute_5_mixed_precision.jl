# NEW SECTION: mixed precision on distributed meshes

# PETSc loads one scalar type per process: a Float32 PETSc library solves A·d = b - Ax for the correction
# while fields stay in the mesh's float type. r and d are PETSc's b and x for that solve.
struct MixedPETScSolver{S,V,R,C} <: AbstractDistributedSolver
    petsc::S
    nzval::V          # Float32 copy of A; owned rows are written
    r::R
    d::R
    w::R              # D⁻¹r, whose global norm the refinement checks
    n_owned::Int
    atol::Float64
    rtol::Float64     # requested; PETSc solves each correction to at most Solve._attainable_rtol
    config::C         # (; hardware) for sync! and kernel launches
    name::String      # equation label and method, for reporting an unconverged refinement
    root::Bool
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
    # with iterative refinement the outer loop reports, not each correction
    report = inner.rtol == setup.rtol
    petsc = PETScSolver(eqn, dmesh, inner; float_type=T, nzval, report, kwargs...)
    zeros_T() = KernelAbstractions.zeros(backend, T, n)
    comm = getfield(dmesh, :comm)
    MixedPETScSolver(petsc, nzval, zeros_T(), zeros_T(), zeros_T(), n, Float64(setup.atol), Float64(setup.rtol),
        (; hardware=config.hardware), Solve._solve_name(setup, get(kwargs, :label, "")),
        MPI.Comm_rank(comm) == 0)
end

# as Solve.mixed_solve!: corrections until the global residual meets rtol, a single one unless rtol is
# below what Float32 attains
function _solve_owned!(deqn::DistributedEqn{E,<:MixedPETScSolver}, result, component) where E
    s = deqn.solver
    values = result.values
    _inner_rtol(s) == s.rtol && (_correct!(deqn, result, component, nothing); return values)
    rnorm = _correct!(deqn, result, component, 0.0)
    iterations = piterations(s.petsc)
    target = Solve._stop_target(s.atol, s.rtol, rnorm)
    for _ ∈ 2:Solve.MAX_REFINEMENTS
        _correct!(deqn, result, component, target) ≤ target && return values
        iterations += piterations(s.petsc)
    end
    # the last correction is checked without solving again
    rnorm = _correct!(deqn, result, component, target; solve=false)
    rnorm ≤ target || (s.root && Solve._warn_unconverged(s.name, Solve._refinement_reason(), iterations,
        rnorm, "‖D⁻¹(b - Ax)‖₂", s, (:unconverged_solve, s.name)))
    values
end

_inner_rtol(s::MixedPETScSolver) = max(s.rtol, Solve._attainable_rtol(MixedF32()))

# one correction unless the global ‖D⁻¹(b - Ax)‖ meets target (`nothing`: the setup's own), stopping
# where the serial correction does; returns that norm. Ghosts are refreshed first since r = b - Ax reads them
function _correct!(deqn, result, component, target; solve=true)
    s, eqn = deqn.solver, deqn.eqn
    (; backend, workgroup) = s.config.hardware
    values = result.values
    sync!(result, get_phi(eqn).mesh, s.config)
    A = _A(eqn)
    kernel! = _sized(Solve._mixed_residual!, backend, workgroup, s.n_owned)
    kernel!(s.r, s.w, s.nzval, _rowptr(A), _colval(A), _nzval(A), eqn.equation.diag_nz, values, _b(eqn, component))
    rnorm = sqrt(MPI.Allreduce(Float64(sum(abs2, s.w)), +, _comm(deqn)))
    rtol = _inner_rtol(s)
    (!solve || rnorm ≤ something(target, Solve._stop_target(s.atol, rtol, rnorm))) && return rnorm
    fill!(s.d, zero(eltype(s.d)))
    # PETSc keeps the matrix between corrections of one solve; s.r stays its right-hand side
    (isnothing(target) || iszero(target)) && passemble!(s.petsc, s.nzval, s.r)
    atol_c, rtol_c = Solve._correction_tol(s.atol, rtol, target, rnorm)
    pset_tolerances!(s.petsc, atol_c, rtol_c, Solve._stop_target(atol_c, rtol_c, rnorm))
    psolve!(s.petsc, s.d)
    kernel! = _sized(Solve._add_correction!, backend, workgroup, s.n_owned)
    kernel!(values, s.d, one(eltype(values)))
    rnorm
end
