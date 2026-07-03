export DistributedEqn, plaplace!, prun!

"""
    DistributedEqn(eqn, solver, partition, halo)

Wraps a serial `ModelEquation` on the rank-local mesh with a distributed solver
(`PETScSolver`), the rank `Partition` and the field `HaloExchange`. Existing generics
(`solve_equation!`, `solve_system!`, `residual`, `setReference!`) dispatch on it.
"""
struct DistributedEqn{E<:ModelEquation,S<:AbstractDistributedSolver,P<:Partition,H<:HaloExchange}
    eqn::E
    solver::S
    partition::P
    halo::H
end

_comm(deqn::DistributedEqn) = deqn.halo.comm

function Solve.solve_equation!(
    deqn::DistributedEqn, phi, phiBCs, solversetup, config; time=nothing, ref=nothing, irelax=nothing)
    eqn = deqn.eqn
    discretise!(eqn, phi, config, rho_prev=eqn.model.terms[1].flux)
    apply_boundary_conditions!(eqn, phiBCs, nothing, time, config)
    _is_pure_laplacian(eqn) && pmake_symmetric!(eqn, config)
    setReference!(deqn, ref, 1, config)
    isnothing(irelax) || implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
    # preconditioner update skipped: the PETSc PC owns preconditioning
    solve_system!(deqn, solversetup, phi, nothing, config)
end

# mirrors serial VectorModel solve; each component sync happens inside solve_system!
function Solve.solve_equation!(
    deqn::DistributedEqn, psi, psiBCs, solversetup, xdir::XDir, ydir::YDir, zdir::ZDir, config;
    time=nothing)
    eqn = deqn.eqn
    discretise!(eqn, psi, config, rho_prev=eqn.model.terms[1].flux)
    update_equation!(eqn, config)
    apply_boundary_conditions!(eqn, psiBCs, xdir, time, config)
    implicit_relaxation_diagdom!(eqn, psi.x.values, solversetup.relax, xdir, config)
    resx = solve_system!(deqn, solversetup, psi.x, xdir, config)

    update_equation!(eqn, config)
    apply_boundary_conditions!(eqn, psiBCs, ydir, time, config)
    implicit_relaxation_diagdom!(eqn, psi.y.values, solversetup.relax, ydir, config)
    resy = solve_system!(deqn, solversetup, psi.y, ydir, config)

    resz = zero(_get_float(psi.mesh))
    if psi.mesh.mesh isa Mesh3
        update_equation!(eqn, config)
        apply_boundary_conditions!(eqn, psiBCs, zdir, time, config)
        implicit_relaxation_diagdom!(eqn, psi.z.values, solversetup.relax, zdir, config)
        resz = solve_system!(deqn, solversetup, psi.z, zdir, config)
    end
    return resx, resy, resz
end

_is_pure_laplacian(eqn) = length(eqn.model.terms) == 1 && eqn.model.terms[1] isa Laplacian

# serial make_symmetric! reads A[owner1, owner2]; on a partition the smaller local id is
# always the trustworthy row (owned block precedes ghosts, ghost rows are garbage)
function pmake_symmetric!(eqn, config)
    (; backend, workgroup) = config.hardware
    A = _A(eqn)
    mesh = get_phi(eqn).mesh
    (; faces) = mesh
    nbfaces = length(mesh.boundary_cellsID)
    ndrange = length(faces) - nbfaces
    kernel! = _pmake_symmetric!(_setup(backend, workgroup, ndrange)...)
    kernel!(_colval(A), _rowptr(A), _nzval(A), faces, nbfaces)
end

@kernel function _pmake_symmetric!(colval, rowptr, nzval, faces, nbfaces)
    i = @index(Global)
    fID = i + nbfaces
    (; ownerCells) = faces[fID]
    c1 = min(ownerCells[1], ownerCells[2])
    c2 = max(ownerCells[1], ownerCells[2])
    i1 = spindex(rowptr, colval, c1, c2)
    i2 = spindex(rowptr, colval, c2, c1)
    @inbounds nzval[i2] = nzval[i1]
end

function Solve.solve_system!(deqn::DistributedEqn, setup, result, component, config)
    (; backend, workgroup) = config.hardware
    passemble!(deqn.solver, deqn.eqn, deqn.partition; component)
    psolve!(deqn.solver, result.values)
    halo_exchange!(result, deqn.halo, backend, workgroup)
    residual(deqn, component, config)
end

# owned rows only (ghost CSR rows are garbage by design); identical value on every rank
function Solve.residual(deqn::DistributedEqn, component, config)
    eqn = deqn.eqn
    (; A, R, Fx) = eqn.equation
    b = _b(eqn, component)
    values = get_values(get_phi(eqn), component)
    (; backend, workgroup) = config.hardware
    n = deqn.partition.n_owned
    kernel! = Solve._scaled_residual!(_setup(backend, workgroup, n)...)
    kernel!(R, Fx, _rowptr(A), _colval(A), _nzval(A), values, b)
    num = MPI.Allreduce(sum(view(R, 1:n)), +, _comm(deqn))
    den = MPI.Allreduce(sum(view(Fx, 1:n)), +, _comm(deqn))
    den = ifelse(den > eps(den), den, one(den))
    num / den
end

# `cellID` is an ORIGINAL global cell id; only the owning rank edits its row
function Solve.setReference!(deqn::DistributedEqn, pRef, cellID, config)
    pRef === nothing && return nothing
    n = deqn.partition.n_owned
    orig = get_phi(deqn.eqn).mesh.orig_cells
    lid = findfirst(==(cellID), view(orig, 1:n))
    lid === nothing || setReference!(deqn.eqn, pRef, lid, config)
    nothing
end

# NEW SECTION: distributed LAPLACE

"""
    plaplace!(model, config; petsc_options="")

Distributed steady/transient Laplace (conduction) solver: `laplace!` on a
`DistributedMesh` with a PETSc distributed solve. Returns `(T=R_T,)` with the global
residual history (identical on every rank). Result output is deferred to Phase 8.
"""
function plaplace!(model, config; petsc_options="", solve_on=nothing, kwargs...)
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, dt) = runtime
    (; backend, workgroup) = hardware
    (; T) = model.energy
    (; k, kf, cp, rho, rhocp, rDf) = model.solid
    dmesh = model.domain
    dmesh isa DistributedMesh || error("plaplace! requires model.domain::DistributedMesh — build it with distribute(mesh)")

    T_eqn = (
        Time{schemes.time}(rhocp, T)
        - Laplacian{schemes.laplacian}(rDf, T)
        ==
        - Source(ScalarField(dmesh))
    ) → ScalarEquation(T, boundaries.T)

    initialise(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, config)

    deqn = DistributedEqn(
        T_eqn,
        PETScSolver(T_eqn, dmesh, solvers; petsc_options, solve_on),
        dmesh.partition,
        HaloExchange(dmesh, 1, backend))

    TF = _get_float(dmesh)
    R_T = ones(TF, iterations)
    dt_cpu = zeros(TF, 1)
    copyto!(dt_cpu, dt)

    halo_exchange!(T, deqn.halo, backend, workgroup)
    for iteration ∈ 1:iterations
        time = iteration * dt_cpu[1]
        rt = solve_equation!(deqn, T, boundaries.T, solvers, config; time=time)
        if model.solid isa NonUniform
            energy!(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, config)
        end
        R_T[iteration] = rt
        if rt <= solvers.convergence && model.time isa Steady
            deqn.partition.rank == 0 && @info "Simulation converged in $iteration iterations!"
            break
        end
    end
    return (T=R_T,)
end

"""
    prun!(model, config; petsc_options="", kwargs...)

Distributed counterpart of `run!`: dispatches on the `Physics` model to the matching
distributed solver. Requires `model.domain::DistributedMesh` and a distributed solver
extension (e.g. `using PETSc`).
"""
prun!(model::Physics{T,F,SO,M,Tu,E,D,BI}, config;
    petsc_options="", kwargs...
    ) where {T,F,SO,M,Tu,E<:Conduction,D<:DistributedMesh,BI} =
    plaplace!(model, config; petsc_options, kwargs...)

prun!(model, config; kwargs...) =
    error("prun!: no distributed solver for this Physics yet (Phase 4 supports Conduction/Laplace); " *
        "model.domain must be a DistributedMesh built with distribute(mesh)")
