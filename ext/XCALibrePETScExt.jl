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
_pc_type(::Union{DILU,ILU0GPU,IC0GPU}) = "bjacobi" # per-rank incomplete-factorisation blocks
_pc_type(::BoomerAMG) = "hypre" # PCHYPRE defaults to boomeramg; no transpose apply (SPD only)
_pc_type(::GAMG) = "gamg" # PETSc native aggregation AMG (SPD only)
_pc_type(p) = nothing

# curated PC kwargs -> PETSc options; each kwarg k=v becomes -pc_<prefix>_<k> v
_pc_options(p) = (;)
_pc_options(::IC0GPU) = (sub_pc_type="icc",)
_pc_options(p::BoomerAMG) =
    NamedTuple(Symbol("pc_hypre_boomeramg_$k") => v for (k, v) ∈ pairs(p.opts))
_pc_options(p::GAMG) =
    NamedTuple(Symbol("pc_gamg_$k") => v for (k, v) ∈ pairs(p.opts))

# serial PCs whose PETSc mapping is a relative, not the same method
_substitute(p) = nothing
_substitute(::Union{DILU,ILU0GPU}) = "per-rank block ILU(0)"
_substitute(::IC0GPU) = "per-rank block ICC(0)"

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

# narrowest index type that addresses the global system: MatMult is memory-bound, so 32-bit
# indices move a quarter fewer bytes per nonzero. Only preference-configured libs exist at
# runtime (wrappers are generated at precompile), so other precisions need their own env.
function _petsclib(TF, nnz_global)
    libs = filter(l -> l.PetscScalar == TF, PETSc.petsclibs)
    isempty(libs) && error("no PETSc library with PetscScalar=$TF (available: " *
        join(("$(l.PetscScalar)/$(l.PetscInt)" for l ∈ PETSc.petsclibs), ", ") *
        "); run in an env whose PETSc preference points at a $TF build")
    fits = filter(l -> nnz_global <= typemax(l.PetscInt), libs)
    isempty(fits) && error("the global matrix has $nnz_global nonzeros, more than any " *
        "$TF PETSc library's index type can address; use a 64-bit-index PETSc build")
    fits[argmin(map(l -> sizeof(l.PetscInt), fits))]
end

# a String reaches every solve; a NamedTuple keyed by equation label adds per-equation options
# after `all`. Labels are the ones the solvers pass to `wrap_eqn`.
const _OPTION_KEYS = (:all, :U, :p, :k, :omega, :T, :y)
_options_for(o::AbstractString, label) = String(o)
function _options_for(o::NamedTuple, label)
    bad = setdiff(keys(o), _OPTION_KEYS)
    isempty(bad) || error("petsc_options: unknown equation key(s) $(Tuple(bad)); " *
        "use $(_OPTION_KEYS)")
    strip(string(get(o, :all, ""), " ", isempty(label) ? "" : get(o, Symbol(label), "")))
end

# hypre's memory-location query reports device memory on every build (a CPU-only hypre maps device
# memory to the host), so the execution policy (HYPRE_EXEC_DEVICE = 1) is the real answer. Querying
# an uninitialised hypre creates its handle and breaks the later BoomerAMG creation (D83).
function _hypre_on_device(petsclib)
    lib = Base.Libc.Libdl.dlopen(petsclib.petsc_library)
    sym(s) = Base.Libc.Libdl.dlsym(lib, s; throw_error=false)
    q, isinit, init = sym(:HYPRE_GetExecutionPolicy), sym(:HYPRE_Initialized), sym(:HYPRE_Initialize)
    (q === nothing || isinit === nothing || init === nothing) && return false
    ccall(isinit, Cint, ()) == 0 && ccall(init, Cint, ())
    policy = Ref{Cint}(-1)
    ccall(q, Cint, (Ptr{Cint},), policy)
    policy[] == 1
end

# `use_gpu_aware_mpi` is a private PETSc symbol: a build without it must name the user's alternative
function _petsc_global(petsclib, sym)
    lib = Base.Libc.Libdl.dlopen(petsclib.petsc_library)
    p = Base.Libc.Libdl.dlsym(lib, sym; throw_error=false)
    p === nothing && error("PETScSolver: this PETSc build does not export `$sym`, so the " *
        "host-staged GPU communication path cannot be selected automatically; pass " *
        "-use_gpu_aware_mpi 0 in petsc_options, or use a CUDA-aware MPI")
    p
end

# PETSc aborts on first device use over a non-CUDA-aware MPI. Its exported flag is cleared rather
# than passing a start-up option, which is lost when PETSc was initialised before this call.
function _gpu_comm!(petsclib, opts, comm)
    aware = MPI.has_cuda()
    user_set = occursin("-use_gpu_aware_mpi", opts)
    if MPI.Comm_rank(comm) == 0
        if user_set
            @info "GPU communication: set by petsc_options" maxlog=1 _id=:gpu_comm
        elseif aware
            @info "GPU communication: CUDA-aware MPI, device buffers passed directly" maxlog=1 _id=:gpu_comm
        else
            @warn "GPU communication: MPI is not CUDA-aware, so inter-rank messages are staged " *
                "through host memory (slower). Solves still run on the GPU. Use a CUDA-aware MPI " *
                "to pass device buffers directly; some need it enabled at launch, e.g. Open MPI " *
                "with OMPI_MCA_opal_cuda_support=true." maxlog=1 _id=:gpu_comm
        end
    end
    (aware || user_set) && return nothing
    unsafe_store!(Ptr{Cint}(_petsc_global(petsclib, :use_gpu_aware_mpi)), Cint(0))
    nothing
end

function PETScSolver(eqn, dmesh::DistributedMesh, setup;
        comm=getfield(dmesh, :comm), petsc_options="", label="")
    petsc_options = _options_for(petsc_options, label)
    part = dmesh.partition
    TF = _get_float(dmesh)
    A = _A(eqn)
    rowptr, colval = Vector(_rowptr(A)), Vector(_colval(A))
    n = part.n_owned
    # owned rows are the contiguous CSR prefix, so the COO values are nzval[1:nnz_owned] in place;
    # ghost rows are garbage and never shipped
    nnz_owned = Int(rowptr[n+1]) - 1
    N, nnz_global = MPI.Allreduce([n, nnz_owned], +, comm)
    petsclib = _petsclib(TF, nnz_global)
    device_solve = !(_nzval(A) isa Array)
    # the same string configures PETSc's start-up and the Krylov solve; entries PETSc does not
    # recognise at one stage are consumed at the other. Start-up options apply on the FIRST call
    # only, since PETSc is initialised once per process.
    PETSc.initialize(petsclib; options=String.(split(petsc_options)))
    PI = petsclib.PetscInt
    # device fields never fall back to host solves; a device-enabled PETSc is required.
    # backend ext declares its PETSc pairing (cuda/mpiaijcusparse, hip/mpiaijhipsparse)
    dev = device_solve ? Distribute.petsc_device_info(_nzval(A)) : nothing
    device_solve && !_petsc_has_pkg(petsclib, dev.pkg) && error(
        "PETScSolver: fields live on the GPU but this PETSc build has no $(dev.pkg) support, " *
        "and GPU runs are not supported on a host-only PETSc. PETSc_jll ships no GPU-enabled " *
        "library, so install a $(dev.pkg)-enabled PETSc and select it through MPIPreferences " *
        "and PETSc's own preferences in this project environment, or run on the CPU backend. " *
        "See the distributed simulations page of the documentation.")
    device_solve && _gpu_comm!(petsclib, petsc_options, comm)
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
    # Krylov.jl's CG stops on sqrt(r'Mr), PETSc's natural norm; keyed on the resolved type so a
    # passthrough -ksp_type never inherits it
    opts.ksp_type == "cg" && !haskey(opts, :ksp_norm_type) &&
        (opts = merge(opts, (ksp_norm_type="natural",)))
    isnothing(opts.ksp_type) && error("no PETSc mapping for solver $(typeof(setup.solver)); " *
        "name one with petsc_options=\"-ksp_type ...\"")
    isnothing(opts.pc_type) && error("no PETSc mapping for preconditioner " *
        "$(typeof(setup.preconditioner)); name one with petsc_options=\"-pc_type ...\"")
    sub = _substitute(setup.preconditioner)
    !isnothing(sub) && opts.pc_type == "bjacobi" && MPI.Comm_rank(comm) == 0 &&
        @warn "$(nameof(typeof(setup.preconditioner))) has no PETSc equivalent; substituting " *
            "$sub (-pc_type bjacobi). Name another with petsc_options=\"-pc_type ...\"" maxlog=1 _id=
            nameof(typeof(setup.preconditioner))
    # catches BoomerAMG and any "-pc_type hypre"/"-pc_hypre_type ..." passthrough
    if any(v -> occursin("hypre", string(v)), values(opts)) && !_petsc_has_pkg(petsclib, "hypre")
        error("PETScSolver: hypre requested but this PETSc build ($(petsclib.PetscScalar)) has " *
            "no hypre support. PETSc_jll carries hypre for Float64 only; other precisions need " *
            "a PETSc configured with --download-hypre. Use GAMG(), which needs no extra build, " *
            "or see the distributed simulations page of the documentation.")
    end
    pc = setup.preconditioner
    if pc isa BoomerAMG && device_solve && !pc.device && !_hypre_on_device(petsclib)
        error("PETScSolver: BoomerAMG() with GPU fields needs a hypre built for the device, and " *
            "this PETSc's hypre runs on the host (or exposes no HYPRE_GetExecutionPolicy), which " *
            "crashes instead of erroring. Use GAMG() or Jacobi() for pressure, or " *
            "BoomerAMG(device=true) to skip this check. See the distributed simulations page of " *
            "the documentation.")
    end
    ksp = PETSc.KSP(Amat; opts...)
    # tolerances mean what they mean to Krylov.jl: rtol is relative to the warm-started initial
    # residual, not PETSc's default ||b||; `convergence` is the outer-loop target only
    (; atol, rtol) = setup
    LibPETSc.KSPSetTolerances(petsclib, ksp, TF(rtol), TF(atol),
        TF(-2), PI(setup.itmax)) # -2 = PETSC_DEFAULT (dtol)
    LibPETSc.KSPSetInitialGuessNonzero(petsclib, ksp, LibPETSc.PETSC_TRUE)
    LibPETSc.KSPConvergedDefaultSetUIRNorm(petsclib, ksp)
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
