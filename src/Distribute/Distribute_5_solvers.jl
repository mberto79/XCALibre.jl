export DistributedEqn

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

# seam methods (S2): below-API distributed layer. wrap_eqn builds the DistributedEqn; the
# solver body assembles/discretises the raw eqn (unwrap_eqn) but solves through the wrapper.
Solve.unwrap_eqn(deqn::DistributedEqn) = deqn.eqn

function Solve.wrap_eqn(eqn, dmesh::DistributedMesh, setup, config;
        petsc_options="", solve_on=nothing)
    (; backend) = config.hardware
    DistributedEqn(eqn, PETScSolver(eqn, dmesh, setup; petsc_options, solve_on),
        getfield(dmesh, :partition), HaloExchange(dmesh, 1, backend))
end

function Solve.solve_equation!(
    deqn::DistributedEqn, phi, phiBCs, solversetup, config; time=nothing, ref=nothing, irelax=nothing)
    eqn = deqn.eqn
    discretise!(eqn, phi, config, rho_prev=eqn.model.terms[1].flux)
    apply_boundary_conditions!(eqn, phiBCs, nothing, time, config)
    _is_pure_laplacian(eqn) && make_symmetric!(eqn, config)
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

# NEW SECTION: reduction + mesh seams (extend the serial identities from Solvers)

Solve.is_distributed_mesh(::DistributedMesh) = true
Solve.is_report_rank(dm::DistributedMesh) = getfield(dm, :partition).rank == 0

# global_max seam (S5): Courant dt must be identical on every rank
Solvers.global_max(v, ::DistributedMesh) = MPI.Allreduce(v, max, MPI.COMM_WORLD)
# courant kernel dispatches on Mesh2/Mesh3 geometry — unwrap the DistributedMesh
Solvers._base_mesh(dm::DistributedMesh) = getfield(dm, :mesh)

# cross-partition periodic BCs are not supported: halo maps carry no periodic adjacency.
# assert_distributable seam is a serial no-op; here it rejects periodic BCs on any field.
_has_periodic(BCs) = any(BC isa PeriodicParent || BC isa Periodic for BC ∈ BCs)
function Solve.assert_distributable(::DistributedMesh, boundaries)
    any(_has_periodic, boundaries) &&
        error("distributed runs do not support periodic boundaries (no cross-partition periodic halo)")
    nothing
end
