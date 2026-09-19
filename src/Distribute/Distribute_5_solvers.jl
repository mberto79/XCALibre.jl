export DistributedEqn

"""
    DistributedEqn(eqn, solver, partition)

Wraps a serial `ModelEquation` on the rank-local mesh with a distributed solver
(`PETScSolver`) and the rank `Partition`; ghost exchanges use the mesh's shared halo schedule.
Existing generics (`solve_equation!`, `solve_system!`, `residual`, `setReference!`) dispatch on
it. The local id of the reference cell (original global cell 1) is cached at construction, 0
when another rank owns it.
"""
struct DistributedEqn{E<:ModelEquation,S<:AbstractDistributedSolver,P<:Partition,V<:AbstractVector}
    eqn::E
    solver::S
    partition::P
    ref_cell::Int
    ref_lid::Int
    diag::V                   # owned diagonals of the x and y systems, kept for their deferred residuals
end
DistributedEqn(eqn, solver, partition) = DistributedEqn(eqn, solver, partition, 1,
    _ref_local(get_phi(eqn).mesh, 1), _diag_store(eqn, partition.n_owned))

_diag_store(eqn, n) = begin
    nz = _nzval(_A(eqn))
    KernelAbstractions.zeros(get_backend(nz), eltype(nz), _n_saved(eqn) * n)
end
_n_saved(::ModelEquation{<:VectorModel}) = 2
_n_saved(_) = 0

# local id of an ORIGINAL global cell id on this rank's owned block, 0 when not owned
function _ref_local(dm::DistributedMesh, cellID)
    n = getfield(dm, :partition).n_owned
    lid = findfirst(==(cellID), view(getfield(dm, :orig_cells), 1:n))
    lid === nothing ? 0 : Int(lid)
end

_comm(deqn::DistributedEqn) = getfield(get_phi(deqn.eqn).mesh, :comm)

# seam methods: below-API distributed layer. wrap_eqn builds the DistributedEqn; the
# solver body assembles/discretises the raw eqn (unwrap_eqn) but solves through the wrapper.
Solve.unwrap_eqn(deqn::DistributedEqn) = deqn.eqn

function Solve.wrap_eqn(eqn, dmesh::DistributedMesh, setup, config;
        petsc_options="", label="")
    DistributedEqn(eqn, PETScSolver(eqn, dmesh, setup; petsc_options, label), getfield(dmesh, :partition))
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

# mirrors serial VectorModel solve with one width-3 exchange after all components: a component's
# solve reads only owned rows, and the systems differ only on the diagonal (update_equation!
# restores A0), so the x and y residuals are taken after the exchange against their saved diagonals
function Solve.solve_equation!(
    deqn::DistributedEqn, psi, psiBCs, solversetup, xdir::XDir, ydir::YDir, zdir::ZDir, config;
    time=nothing)
    eqn = deqn.eqn
    n = deqn.partition.n_owned
    discretise!(eqn, psi, config, rho_prev=eqn.model.terms[1].flux)
    update_equation!(eqn, config)
    apply_boundary_conditions!(eqn, psiBCs, xdir, time, config)
    implicit_relaxation_diagdom!(eqn, psi.x.values, solversetup.relax, xdir, config)
    _solve_owned!(deqn, psi.x, xdir)
    _save_diag!(deqn, 0, config)

    update_equation!(eqn, config)
    apply_boundary_conditions!(eqn, psiBCs, ydir, time, config)
    implicit_relaxation_diagdom!(eqn, psi.y.values, solversetup.relax, ydir, config)
    _solve_owned!(deqn, psi.y, ydir)

    is3d = psi.mesh.mesh isa Mesh3
    if is3d
        _save_diag!(deqn, n, config)
        update_equation!(eqn, config)
        apply_boundary_conditions!(eqn, psiBCs, zdir, time, config)
        implicit_relaxation_diagdom!(eqn, psi.z.values, solversetup.relax, zdir, config)
        _solve_owned!(deqn, psi.z, zdir)
    end
    sync!(psi, psi.mesh, config)

    resx = _residual_saved(deqn, xdir, 0, config)
    if is3d
        resy = _residual_saved(deqn, ydir, n, config)
        resz = residual(deqn, zdir, config)
    else
        resy = residual(deqn, ydir, config)
        resz = zero(_get_float(psi.mesh))
    end
    return resx, resy, resz
end

_solve_owned!(deqn, result, component) = begin
    passemble!(deqn.solver, deqn.eqn, deqn.partition; component)
    psolve!(deqn.solver, result.values)
end

function _save_diag!(deqn, off, config)
    A = _A(deqn.eqn)
    (; backend, workgroup) = config.hardware
    kernel! = _save_diag_kernel!(_setup(backend, workgroup, deqn.partition.n_owned)...)
    kernel!(deqn.diag, off, _rowptr(A), _colval(A), _nzval(A))
end

@kernel function _save_diag_kernel!(diag, off, @Const(rowptr), @Const(colval), @Const(nzval))
    i = @index(Global)
    @inbounds diag[off + i] = nzval[spindex(rowptr, colval, i, i)]
end

# same arithmetic, in the same order, as Solve._scaled_residual! with the diagonal read from diag
@kernel function _scaled_residual_saved!(R, Fx, @Const(rowptr), @Const(colval), @Const(nzval),
        @Const(values), @Const(b), @Const(diag), off)
    i = @index(Global)
    Ax = zero(eltype(R))
    Dx = zero(eltype(R))
    xi = values[i]

    @inbounds for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        j = colval[nzi]
        Aij = j == i ? diag[off + i] : nzval[nzi]
        Ax += Aij * values[j]
        if j == i
            Dx = Aij * xi
        end
    end

    @inbounds begin
        R[i] = abs(b[i] - Ax)
        Fx[i] = abs(Dx)
    end
end

_is_pure_laplacian(eqn) = length(eqn.model.terms) == 1 && eqn.model.terms[1] isa Laplacian

function Solve.solve_system!(deqn::DistributedEqn, setup, result, component, config)
    _solve_owned!(deqn, result, component)
    sync!(result, get_phi(deqn.eqn).mesh, config)
    residual(deqn, component, config)
end

# all-reduces issued by the solver seams since load; budgeted per iteration in `test_perf.jl`
const ALLREDUCE_COUNT = Ref(0)

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
    _reduce_residual(deqn, R, Fx)
end

function _residual_saved(deqn::DistributedEqn, component, off, config)
    eqn = deqn.eqn
    (; A, R, Fx) = eqn.equation
    (; backend, workgroup) = config.hardware
    kernel! = _scaled_residual_saved!(_setup(backend, workgroup, deqn.partition.n_owned)...)
    kernel!(R, Fx, _rowptr(A), _colval(A), _nzval(A), get_values(get_phi(eqn), component),
        _b(eqn, component), deqn.diag, off)
    _reduce_residual(deqn, R, Fx)
end

function _reduce_residual(deqn, R, Fx)
    n = deqn.partition.n_owned
    ALLREDUCE_COUNT[] += 2
    num = MPI.Allreduce(sum(view(R, 1:n)), +, _comm(deqn))
    den = MPI.Allreduce(sum(view(Fx, 1:n)), +, _comm(deqn))
    den = ifelse(den > eps(den), den, one(den))
    num / den
end

# `cellID` is an ORIGINAL global cell id; only the owning rank edits its row
function Solve.setReference!(deqn::DistributedEqn, pRef, cellID, config)
    pRef === nothing && return nothing
    lid = cellID == deqn.ref_cell ? deqn.ref_lid : _ref_local(get_phi(deqn.eqn).mesh, cellID)
    lid == 0 || setReference!(deqn.eqn, pRef, lid, config)
    nothing
end

# NEW SECTION: reduction + mesh seams (extend the serial identities from Solvers)

Solve.is_distributed_mesh(::DistributedMesh) = true
Solve.is_report_rank(dm::DistributedMesh) = MPI.Comm_rank(getfield(dm, :comm)) == 0

# global_max seam: Courant dt must be identical on every rank
Solvers.global_max(v, dm::DistributedMesh) =
    (ALLREDUCE_COUNT[] += 1; MPI.Allreduce(v, max, getfield(dm, :comm)))
# courant kernel dispatches on Mesh2/Mesh3 geometry — unwrap the DistributedMesh
Solvers._base_mesh(dm::DistributedMesh) = getfield(dm, :mesh)
