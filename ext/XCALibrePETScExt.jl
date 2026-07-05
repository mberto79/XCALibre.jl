module XCALibrePETScExt

using XCALibre, MPI, PETSc
using PETSc: LibPETSc
import KernelAbstractions
using XCALibre.Distribute
import XCALibre.Distribute: PETScSolver, passemble!, psolve!, psolve_transpose!
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval
import XCALibre.Mesh: _get_float

# NEW SECTION: KSP/PC mapping (curated; anything else via petsc_options passthrough)

_ksp_type(::Cg) = "cg"
_ksp_type(::Bicgstab) = "bcgs"
_ksp_type(::Gmres) = "gmres"
_ksp_type(s) = error("no PETSc mapping for solver $(typeof(s)); use petsc_options=\"-ksp_type ...\"")
_pc_type(::Jacobi) = "jacobi"
_pc_type(::BoomerAMG) = "hypre" # PCHYPRE defaults to boomeramg; no transpose apply (SPD only)
_pc_type(p) = error("no PETSc mapping for preconditioner $(typeof(p)); use petsc_options=\"-pc_type ...\"")

# NEW SECTION: solver type

struct XPETScSolver{PL,TM,TV,TK,TF} <: Distribute.AbstractDistributedSolver
    petsclib::PL
    A::TM
    b::TV
    x::TV
    ksp::TK
    n_owned::Int
    nnz_owned::Int
    vals::Vector{TF}   # host staging: owned-row nzval slice
    bhost::Vector{TF}
    xhost::Vector{TF}
end

_petsc_has_pkg(petsclib, pkg) =
    LibPETSc.PetscHasExternalPackage(petsclib, Vector{Int8}(codeunits(pkg * "\0")))

# system PETSc builds may be Int32-indexed; select by scalar, keep the lib's PetscInt.
# NB runtime set_petsclib can NOT work here: LibPETSc wrappers are @for_petsc-generated at
# precompile time for the preference-configured lib(s) only — other precisions need their
# own project env with library_path/PetscScalar prefs (e.g. dev/petscenv_f32).
function _petsclib(TF)
    i = findfirst(l -> l.PetscScalar == TF, PETSc.petsclibs)
    i === nothing && error("no PETSc library with PetscScalar=$TF (available: " *
        join(("$(l.PetscScalar)/$(l.PetscInt)" for l ∈ PETSc.petsclibs), ", ") *
        "); run in an env whose PETSc preference points at a $TF build")
    PETSc.petsclibs[i]
end

function PETScSolver(eqn, dmesh::DistributedMesh, setup;
        comm=MPI.COMM_WORLD, petsc_options="", solve_on=nothing)
    part = dmesh.partition
    TF = _get_float(dmesh)
    petsclib = _petsclib(TF)
    PETSc.initialize(petsclib)
    PI = petsclib.PetscInt
    A = _A(eqn)
    # device fields + host PETSc = hard error unless solves are explicitly opted onto host
    device_solve = !(_nzval(A) isa Array) && !(solve_on isa KernelAbstractions.CPU)
    # backend ext declares its PETSc pairing (cuda/mpiaijcusparse, hip/mpiaijhipsparse)
    dev = device_solve ? Distribute.petsc_device_info(_nzval(A)) : nothing
    device_solve && !_petsc_has_pkg(petsclib, dev.pkg) && error(
        "PETScSolver: fields live on the GPU but this PETSc build has no $(dev.pkg) support. " *
        "Fixes: MPIPreferences.use_system_binary() + a $(dev.pkg)-enabled system PETSc " *
        "(JULIA_PETSC_LIBRARY), or opt into host-side solves with solve_on=CPU() " *
        "(A/b copied to host each solve).")
    rowptr, colval = Vector(_rowptr(A)), Vector(_colval(A))
    n = part.n_owned
    N = MPI.Allreduce(n, +, comm)
    # owned rows are the contiguous CSR prefix; ghost rows are garbage and never shipped
    nnz_owned = Int(rowptr[n+1]) - 1
    i0 = PI[rowptr[i] - 1 for i ∈ 1:n+1]
    l2g = part.local_to_global
    j0 = PI[l2g[colval[k]] - 1 for k ∈ 1:nnz_owned]
    vals = Vector{TF}(undef, nnz_owned)
    copyto!(vals, view(_nzval(A), 1:nnz_owned))
    Amat = LibPETSc.MatCreateMPIAIJWithArrays(petsclib, comm,
        PI(n), PI(n), PI(N), PI(N), i0, j0, vals)
    if device_solve
        # device-sparse mat/vecs; values still updated via MatUpdateMPIAIJWithArray
        mt = dev.mat
        # ponytail: LibPETSc.MatConvert nulls M.ptr and drops the converted handle;
        # MAT_INPLACE_MATRIX keeps the same C Mat (MatHeaderReplace), so restore it.
        orig = Amat.ptr
        GC.@preserve mt LibPETSc.MatConvert(petsclib, Amat, Cstring(pointer(mt)),
            LibPETSc.MAT_INPLACE_MATRIX, Amat)
        Amat.ptr = orig
    end
    x, b = LibPETSc.MatCreateVecs(petsclib, Amat)
    curated = (; ksp_type=_ksp_type(setup.solver), pc_type=_pc_type(setup.preconditioner))
    raw = isempty(petsc_options) ? (;) : PETSc.parse_options(String.(split(petsc_options)))
    opts = merge(curated, raw)
    # catches BoomerAMG and any "-pc_type hypre"/"-pc_hypre_type ..." passthrough
    if any(v -> occursin("hypre", string(v)), values(opts)) && !_petsc_has_pkg(petsclib, "hypre")
        error("PETScSolver: hypre requested but this PETSc build has no hypre support. " *
            "Rebuild PETSc with --download-hypre (see build_cuda_ucx_openmpi_petsc.sh) " *
            "or pick another preconditioner.")
    end
    ksp = PETSc.KSP(Amat; opts...)
    LibPETSc.KSPSetTolerances(petsclib, ksp, TF(setup.rtol), TF(setup.atol),
        TF(-2), PI(setup.itmax)) # -2 = PETSC_DEFAULT (dtol)
    LibPETSc.KSPSetInitialGuessNonzero(petsclib, ksp, LibPETSc.PETSC_TRUE)
    XPETScSolver(petsclib, Amat, b, x, ksp, n, nnz_owned, vals,
        Vector{TF}(undef, n), Vector{TF}(undef, n))
end

# NEW SECTION: assembly and solve

function passemble!(s::XPETScSolver, eqn, partition; component=nothing)
    copyto!(s.vals, view(_nzval(_A(eqn)), 1:s.nnz_owned))
    LibPETSc.MatUpdateMPIAIJWithArray(s.petsclib, s.A, s.vals)
    copyto!(s.bhost, view(_b(eqn, component), 1:s.n_owned))
    PETSc.withlocalarray!(s.b; read=false, write=true) do arr
        copyto!(arr, s.bhost)
    end
    s
end

_copy_owned_in!(s, x) = begin
    copyto!(s.xhost, view(x, 1:s.n_owned))
    PETSc.withlocalarray!(s.x; read=false, write=true) do arr
        copyto!(arr, s.xhost)
    end
end

_copy_owned_out!(s, x) = begin
    PETSc.withlocalarray!(s.x; read=true, write=false) do arr
        copyto!(s.xhost, arr)
    end
    copyto!(view(x, 1:s.n_owned), s.xhost)
end

function psolve!(s::XPETScSolver, x::AbstractVector)
    _copy_owned_in!(s, x)
    PETSc.solve!(s.x, s.ksp, s.b)
    _copy_owned_out!(s, x)
    x
end

function psolve_transpose!(s::XPETScSolver, x::AbstractVector)
    _copy_owned_in!(s, x)
    LibPETSc.KSPSolveTranspose(s.petsclib, s.ksp, s.b, s.x)
    _copy_owned_out!(s, x)
    x
end

end # module
