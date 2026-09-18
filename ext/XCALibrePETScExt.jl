module XCALibrePETScExt

using XCALibre, MPI, PETSc
using PETSc: LibPETSc
using XCALibre.Distribute
import XCALibre.Distribute: PETScSolver, passemble!, psolve!, psolve_transpose!
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval
import XCALibre.Mesh: _get_float

# NEW SECTION: KSP/PC mapping (curated; anything else via petsc_options passthrough)

# unmapped types return nothing and must be named through petsc_options
_ksp_type(::Cg) = "cg"
_ksp_type(::Cgs) = "cgs"
_ksp_type(::Bicgstab) = "bcgs"
_ksp_type(::Gmres) = "gmres"
_ksp_type(s) = nothing
_pc_type(::Jacobi) = "jacobi"
_pc_type(::DILU) = "bjacobi" # per-rank ILU(0) blocks: closest PETSc relative of DILU
_pc_type(::BoomerAMG) = "hypre" # PCHYPRE defaults to boomeramg; no transpose apply (SPD only)
_pc_type(::GAMG) = "gamg" # PETSc native aggregation AMG (SPD only)
_pc_type(p) = nothing

# curated PC kwargs -> PETSc options; each kwarg k=v becomes -pc_<prefix>_<k> v
_pc_options(p) = (;)
_pc_options(p::BoomerAMG) =
    NamedTuple(Symbol("pc_hypre_boomeramg_$k") => v for (k, v) ∈ pairs(p.opts))
_pc_options(p::GAMG) =
    NamedTuple(Symbol("pc_gamg_$k") => v for (k, v) ∈ pairs(p.opts))

# PCs that manage their own hierarchy across solves carry a `freeze` count (rebuild every N solves)
_pc_freeze(p) = 1
_pc_freeze(p::Union{BoomerAMG,GAMG}) = p.freeze

# NEW SECTION: solver type

struct XPETScSolver{PL,TM,TV,TK,SY} <: Distribute.AbstractDistributedSolver
    petsclib::PL
    A::TM
    b::TV
    x::TV
    ksp::TK
    n_owned::Int
    sync::SY           # device-wide sync (nothing on host): PETSc and XCALibre use separate streams
    setup_every::Int   # rebuild the PC every N solves (1 = every solve, PETSc default)
    nsolve::Base.RefValue{Int}
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
        comm=MPI.COMM_WORLD, petsc_options="", label="")
    part = dmesh.partition
    TF = _get_float(dmesh)
    petsclib = _petsclib(TF)
    # the same string configures PETSc's start-up and the Krylov solve; entries PETSc does not
    # recognise at one stage are consumed at the other. Start-up options apply on the FIRST call
    # only, since PETSc is initialised once per process.
    PETSc.initialize(petsclib; options=String.(split(petsc_options)))
    PI = petsclib.PetscInt
    A = _A(eqn)
    # device fields never fall back to host solves; a device-enabled PETSc is required
    device_solve = !(_nzval(A) isa Array)
    # backend ext declares its PETSc pairing (cuda/mpiaijcusparse, hip/mpiaijhipsparse)
    dev = device_solve ? Distribute.petsc_device_info(_nzval(A)) : nothing
    device_solve && !_petsc_has_pkg(petsclib, dev.pkg) && error(
        "PETScSolver: fields live on the GPU but this PETSc build has no $(dev.pkg) support, " *
        "and GPU runs are not supported on a host-only PETSc. PETSc_jll ships no GPU-enabled " *
        "library, so install a $(dev.pkg)-enabled PETSc and select it through MPIPreferences " *
        "and PETSc's own preferences in this project environment, or run on the CPU backend. " *
        "See the distributed simulations page of the documentation.")
    rowptr, colval = Vector(_rowptr(A)), Vector(_colval(A))
    n = part.n_owned
    N = MPI.Allreduce(n, +, comm)
    # owned rows are the contiguous CSR prefix, so the COO values are nzval[1:nnz_owned] in place;
    # ghost rows are garbage and never shipped
    nnz_owned = Int(rowptr[n+1]) - 1
    l2g = part.local_to_global
    coo_i = Vector{PI}(undef, nnz_owned)
    for r ∈ 1:n, k ∈ rowptr[r]:rowptr[r+1]-1
        coo_i[k] = l2g[r] - 1
    end
    coo_j = PI[l2g[colval[k]] - 1 for k ∈ 1:nnz_owned]
    Amat = LibPETSc.MatCreate(petsclib, comm)
    LibPETSc.MatSetSizes(petsclib, Amat, PI(n), PI(n), PI(N), PI(N))
    mt = device_solve ? dev.mat : "mpiaij"
    GC.@preserve mt LibPETSc.MatSetType(petsclib, Amat, Base.unsafe_convert(Cstring, mt))
    LibPETSc.MatSetPreallocationCOO(petsclib, Amat, LibPETSc.PetscCount(nnz_owned), coo_i, coo_j)
    sync = device_solve ? dev.sync : nothing
    _set_values!(petsclib, Amat, _nzval(A), sync)
    x, b = LibPETSc.MatCreateVecs(petsclib, Amat)
    curated = merge((; ksp_type=_ksp_type(setup.solver), pc_type=_pc_type(setup.preconditioner)),
        _pc_options(setup.preconditioner))
    raw = isempty(petsc_options) ? (;) : PETSc.parse_options(String.(split(petsc_options)))
    opts = merge(curated, raw)
    isnothing(opts.ksp_type) && error("no PETSc mapping for solver $(typeof(setup.solver)); " *
        "name one with petsc_options=\"-ksp_type ...\"")
    isnothing(opts.pc_type) && error("no PETSc mapping for preconditioner " *
        "$(typeof(setup.preconditioner)); name one with petsc_options=\"-pc_type ...\"")
    setup.preconditioner isa DILU && opts.pc_type == "bjacobi" && MPI.Comm_rank(comm) == 0 &&
        @warn "DILU has no PETSc equivalent; substituting per-rank block ILU(0) " *
            "(-pc_type bjacobi). Name another with petsc_options=\"-pc_type ...\"" maxlog=1
    # catches BoomerAMG and any "-pc_type hypre"/"-pc_hypre_type ..." passthrough
    if any(v -> occursin("hypre", string(v)), values(opts)) && !_petsc_has_pkg(petsclib, "hypre")
        error("PETScSolver: hypre requested but this PETSc build ($(petsclib.PetscScalar)) has " *
            "no hypre support. PETSc_jll carries hypre for Float64 only; other precisions need " *
            "a PETSc configured with --download-hypre. Use GAMG(), which needs no extra build, " *
            "or see the distributed simulations page of the documentation.")
    end
    ksp = PETSc.KSP(Amat; opts...)
    # tolerances mean what they mean to Krylov; `convergence` is the outer-loop target only
    (; atol, rtol) = setup
    LibPETSc.KSPSetTolerances(petsclib, ksp, TF(rtol), TF(atol),
        TF(-2), PI(setup.itmax)) # -2 = PETSC_DEFAULT (dtol)
    LibPETSc.KSPSetInitialGuessNonzero(petsclib, ksp, LibPETSc.PETSC_TRUE)
    extra = Base.structdiff(opts, (ksp_type=nothing, pc_type=nothing))
    MPI.Comm_rank(comm) == 0 && @info "PETSc solve [$label]: KSP=$(opts.ksp_type) " *
        "PC=$(opts.pc_type) atol=$(TF(atol)) rtol=$(TF(rtol)) itmax=$(setup.itmax)" *
        (isempty(extra) ? "" : " " * join(("$k=$v" for (k, v) ∈ pairs(extra)), " "))
    setup_every = _pc_freeze(setup.preconditioner)
    XPETScSolver(petsclib, Amat, b, x, ksp, n, sync, setup_every, Ref(0))
end

# NEW SECTION: assembly and solve

_sync(::Nothing) = nothing
_sync(f) = f()

# PETSc reads the values where they live (host or device) through its COO map; no staging copy
function _set_values!(petsclib, A, nzval, sync)
    _sync(sync)
    GC.@preserve nzval LibPETSc.MatSetValuesCOO(petsclib, A,
        reinterpret(Ptr{eltype(nzval)}, pointer(nzval)), LibPETSc.INSERT_VALUES)
end

function passemble!(s::XPETScSolver, eqn, partition; component=nothing)
    _set_values!(s.petsclib, s.A, _nzval(_A(eqn)), s.sync)
    PETSc.withlocalarray!(s.b; read=false, write=true) do arr
        copyto!(arr, view(_b(eqn, component), 1:s.n_owned))
    end
    s
end

# PETSc.jl hands back a device array for a device Vec, so both copies stay on the device
_copy_owned_in!(s, x) = PETSc.withlocalarray!(s.x; read=false, write=true) do arr
    copyto!(arr, view(x, 1:s.n_owned))
end

_copy_owned_out!(s, x) = PETSc.withlocalarray!(s.x; read=true, write=false) do arr
    copyto!(view(x, 1:s.n_owned), arr)
end

# rebuild the PC every `setup_every` solves; apply the frozen (cheap-to-apply) hierarchy in between.
# Krylov still uses the updated matrix, so it converges to the current system's solution.
function _maybe_freeze_pc!(s::XPETScSolver)
    s.setup_every <= 1 && return
    n = s.nsolve[]; s.nsolve[] = n + 1
    flag = (n % s.setup_every == 0) ? LibPETSc.PETSC_FALSE : LibPETSc.PETSC_TRUE
    LibPETSc.KSPSetReusePreconditioner(s.petsclib, s.ksp, flag)
end

function psolve!(s::XPETScSolver, x::AbstractVector)
    _maybe_freeze_pc!(s)
    _copy_owned_in!(s, x)
    _sync(s.sync)
    PETSc.solve!(s.x, s.ksp, s.b)
    _sync(s.sync)
    _copy_owned_out!(s, x)
    x
end

function psolve_transpose!(s::XPETScSolver, x::AbstractVector)
    _copy_owned_in!(s, x)
    _sync(s.sync)
    LibPETSc.KSPSolveTranspose(s.petsclib, s.ksp, s.b, s.x)
    _sync(s.sync)
    _copy_owned_out!(s, x)
    x
end

end # module
