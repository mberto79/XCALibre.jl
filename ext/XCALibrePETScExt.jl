module XCALibrePETScExt

using XCALibre, MPI, PETSc
using PETSc: LibPETSc
using XCALibre.Distribute
import XCALibre.Distribute: PETScSolver, passemble!, psolve!, psolve_transpose!, _parallel_partition
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

# NEW SECTION: signal dispositions

# PETSc's CUDA device init resets eleven signals to SIG_DFL, so Julia's safepoint faults would kill
# the process silently; the dispositions in force before are put back
const _JULIA_SIGNALS = Cint.((1, 3, 4, 5, 7, 8, 11, 13, 15, 23, 31)) # HUP QUIT ILL TRAP BUS FPE SEGV PIPE TERM URG SYS
const _SIGACTION_BYTES = 152

_sigaction_get(sig) = (buf = zeros(UInt8, _SIGACTION_BYTES);
    ccall(:sigaction, Cint, (Cint, Ptr{Cvoid}, Ptr{UInt8}), sig, C_NULL, buf); buf)

function _with_julia_signals(f)
    saved = map(_sigaction_get, _JULIA_SIGNALS)
    result = f()
    for (sig, act) ∈ zip(_JULIA_SIGNALS, saved)
        _sigaction_get(sig) == act && continue
        ccall(:sigaction, Cint, (Cint, Ptr{UInt8}, Ptr{Cvoid}), sig, act, C_NULL)
    end
    result
end

# NEW SECTION: solver type

struct XPETScSolver{PL,TM,TV,TK,SY,FI} <: Distribute.AbstractDistributedSolver
    petsclib::PL
    A::TM
    b::TV
    x::TV
    ksp::TK
    n_owned::Int
    sync::SY           # device-wide sync (nothing on host): PETSc and XCALibre use separate streams
    fill::FI           # host block scatter; nothing on device, where values go through the COO map
    place::Ptr{Cvoid}  # Vec(CUDA)PlaceArray: x and b own no storage and borrow the caller's arrays
    reset::Ptr{Cvoid}  # Vec(CUDA)ResetArray
    bptr::Base.RefValue{Ptr{Cvoid}} # b's array, set by passemble! and placed by the next solve
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
# an uninitialised hypre creates its handle and breaks the later BoomerAMG creation.
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

# one live solver per equation label per process: a new wrap of a label ends the previous run's
# solver, and every rank wraps in the same order, so the collective destroys match
const _LIVE = Dict{String,Any}()

function PETScSolver(eqn, dmesh::DistributedMesh, setup; label="", kwargs...)
    isempty(label) || _release!(pop!(_LIVE, label, nothing))
    s = _with_julia_signals(() -> _petsc_solver(eqn, dmesh, setup; label, kwargs...))
    isempty(label) || (_LIVE[label] = s)
    s
end

_release!(::Nothing) = nothing
function _release!(s)
    isnothing(s.ksp.opts) || PETSc.destroy(s.ksp.opts)
    foreach(PETSc.destroy, (s.ksp, s.A, s.x, s.b))
end

function _petsc_solver(eqn, dmesh::DistributedMesh, setup;
        comm=getfield(dmesh, :comm), petsc_options="", label="")
    petsc_options = _options_for(petsc_options, label)
    part = dmesh.partition
    TF = _get_float(dmesh)
    A = _A(eqn)
    rowptr, colval = _host(_rowptr(A)), _host(_colval(A))
    n = part.n_owned
    # owned rows are the contiguous CSR prefix, so the COO values are nzval[1:nnz_owned] in place;
    # ghost rows are garbage and never shipped
    nnz_owned = Int(rowptr[n+1]) - 1
    N, nnz_global = MPI.Allreduce([n, nnz_owned], +, comm)
    petsclib = _petsclib(TF, nnz_global)
    device_solve = !(_nzval(A) isa Array)
    # the same string configures PETSc's start-up and the Krylov solve; entries PETSc does not
    # recognise at one stage are consumed at the other. Start-up options apply on the first call
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
    sync = device_solve ? dev.sync : nothing
    Amat, fill = device_solve ?
        _coo_matrix(petsclib, comm, dev.mat, rowptr, colval, l2g, n, N, nnz_owned) :
        _split_matrix(petsclib, comm, _rowptr(A), _colval(A), _nzval(A), l2g, n, N)
    _set_values!(petsclib, Amat, _nzval(A), sync, fill)
    vec = device_solve ? dev.vec : _HOST_VEC
    x, b = (_vec_without_array(petsclib, comm, n, N, vec.create) for _ ∈ 1:2)
    curated = merge((; ksp_type=_ksp_type(setup.solver), pc_type=_pc_type(setup.preconditioner)),
        _pc_options(setup.preconditioner))
    raw = isempty(petsc_options) ? (;) : PETSc.parse_options(String.(split(petsc_options)))
    opts = merge(curated, raw)
    # PETSc's natural norm sqrt(r'Mr); serial solves measure rtol on the unpreconditioned
    # ||b - Ax|| instead (Solve.solve_system!), which "-ksp_norm_type unpreconditioned" matches.
    # Keyed on the resolved type so a passthrough -ksp_type never inherits it
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
    XPETScSolver(petsclib, Amat, b, x, ksp, n, sync, fill, _petsc_sym(petsclib, vec.place),
        _petsc_sym(petsclib, vec.reset), Ref(C_NULL), setup_every, Ref(0))
end

# NEW SECTION: assembly and solve

_sync(::Nothing) = nothing
_sync(f) = f()

_host(x) = x isa Array ? x : Array(x)

_petsc_sym(petsclib, name) = Base.Libc.Libdl.dlsym(Base.Libc.Libdl.dlopen(petsclib.petsc_library), name)

const _HOST_VEC = (create=:VecCreateMPIWithArray, place=:VecPlaceArray, reset=:VecResetArray)

function _vec_without_array(petsclib, comm, n, N, create)
    v = Ref{LibPETSc.CVec}(C_NULL)
    PI = petsclib.PetscInt
    _vec_create(_petsc_sym(petsclib, create), comm, PI(n), PI(N), v) == 0 ||
        error("PETSc: $create failed")
    LibPETSc.PetscVec(v[], petsclib)
end

for I ∈ (Int32, Int64)
    @eval _vec_create(f, comm, n::$I, N::$I, v) = ccall(f, Cint,
        (MPI.MPI_Comm, $I, $I, $I, Ptr{Cvoid}, Ptr{LibPETSc.CVec}), comm, one($I), n, N, C_NULL, v)
end

# host Ptr or device CuPtr alike: PETSc takes the owned prefix, which starts the array
_raw_ptr(x) = reinterpret(Ptr{Cvoid}, pointer(x))

_vec_call(f, v, p) = ccall(f, Cint, (LibPETSc.CVec, Ptr{Cvoid}), v, p) == 0 || error("PETSc: vector placement failed")
_vec_call(f, v) = ccall(f, Cint, (LibPETSc.CVec,), v) == 0 || error("PETSc: vector reset failed")

function _coo_matrix(petsclib, comm, mt, rowptr, colval, l2g, n, N, nnz_owned)
    PI = petsclib.PetscInt
    coo_i = Vector{PI}(undef, nnz_owned)
    for r ∈ 1:n, k ∈ rowptr[r]:rowptr[r+1]-1
        coo_i[k] = l2g[r] - 1
    end
    coo_j = PI[l2g[colval[k]] - 1 for k ∈ 1:nnz_owned]
    Amat = LibPETSc.MatCreate(petsclib, comm)
    LibPETSc.MatSetSizes(petsclib, Amat, PI(n), PI(n), PI(N), PI(N))
    GC.@preserve mt LibPETSc.MatSetType(petsclib, Amat, Base.unsafe_convert(Cstring, mt))
    LibPETSc.MatSetPreallocationCOO(petsclib, Amat, LibPETSc.PetscCount(nnz_owned), coo_i, coo_j)
    Amat, nothing
end

# owned rows list owned columns then ghosts, each ascending in global id (ghosts are sorted by
# owning rank, then id), which is exactly the order of PETSc's diagonal and off-diagonal blocks
struct _SplitFill{M,VI}
    Ad::M
    Ao::M
    rowptr::VI
    colval::VI
    n::Int
    nnz_o::Int
    get::Ptr{Cvoid}     # MatSeqAIJGetArrayWrite
    restore::Ptr{Cvoid} # MatSeqAIJRestoreArrayWrite
end

function _seqaij_nnz(petsclib, M)
    info = Ref{LibPETSc.MatInfo}()
    LibPETSc.MatGetInfo(petsclib, M, LibPETSc.MAT_LOCAL, info)
    Int(info[].nz_used)
end

# PETSc.jl's MatSeqAIJGetArrayWrite wrapper is broken (sizes by an undefined Vec), so call it direct
function _seqaij_array(fp, M, ::Type{T}) where T
    p = Ref{Ptr{T}}(C_NULL)
    ccall(fp, Cint, (Ptr{Cvoid}, Ptr{Ptr{T}}), M.ptr, p) == 0 || error("PETSc: MatSeqAIJ array access failed")
    p[]
end

function _split_matrix(petsclib, comm, rowptr, colval, nzval, l2g, n, N)
    PI = petsclib.PetscInt
    issorted(view(l2g, 1:n)) && issorted(view(l2g, n+1:length(l2g))) ||
        error("PETScSolver: owned and ghost cells must each be numbered in ascending global order")
    nnz_owned = Int(rowptr[n+1]) - 1
    Amat = LibPETSc.MatCreateMPIAIJWithArrays(petsclib, comm, PI(n), PI(n), PI(N), PI(N),
        PI[rowptr[r] - 1 for r ∈ 1:n+1], PI[l2g[colval[k]] - 1 for k ∈ 1:nnz_owned], nzval)
    LibPETSc.MatSetOption(petsclib, Amat, LibPETSc.MAT_NO_OFF_PROC_ENTRIES, LibPETSc.PETSC_TRUE)
    Ad, Ao, _ = LibPETSc.MatMPIAIJGetSeqAIJ(petsclib, Amat)
    nnz_o = count(k -> colval[k] > n, 1:nnz_owned)
    (_seqaij_nnz(petsclib, Ad), _seqaij_nnz(petsclib, Ao)) == (nnz_owned - nnz_o, nnz_o) ||
        error("PETScSolver: PETSc's diagonal/off-diagonal split does not match the local CSR")
    lib = Base.Libc.Libdl.dlopen(petsclib.petsc_library)
    Amat, _SplitFill(Ad, Ao, rowptr, colval, n, nnz_o,
        Base.Libc.Libdl.dlsym(lib, :MatSeqAIJGetArrayWrite), Base.Libc.Libdl.dlsym(lib, :MatSeqAIJRestoreArrayWrite))
end

# PETSc reads the values where they live (host or device) through its COO map; no staging copy
function _set_values!(petsclib, A, nzval, sync, ::Nothing)
    _sync(sync)
    GC.@preserve nzval LibPETSc.MatSetValuesCOO(petsclib, A,
        reinterpret(Ptr{eltype(nzval)}, pointer(nzval)), LibPETSc.INSERT_VALUES)
end

function _set_values!(petsclib, A, nzval::Vector{T}, sync, f::_SplitFill) where T
    ad = _seqaij_array(f.get, f.Ad, T)
    ao = f.nnz_o > 0 ? _seqaij_array(f.get, f.Ao, T) : Ptr{T}(C_NULL)
    kd = ko = 0
    @inbounds for r ∈ 1:f.n, k ∈ f.rowptr[r]:f.rowptr[r+1]-1
        if f.colval[k] <= f.n
            unsafe_store!(ad, nzval[k], kd += 1)
        else
            unsafe_store!(ao, nzval[k], ko += 1)
        end
    end
    _seqaij_array(f.restore, f.Ad, T)
    f.nnz_o > 0 && _seqaij_array(f.restore, f.Ao, T)
    LibPETSc.MatAssemblyBegin(petsclib, A, LibPETSc.MAT_FINAL_ASSEMBLY)
    LibPETSc.MatAssemblyEnd(petsclib, A, LibPETSc.MAT_FINAL_ASSEMBLY)
end

function passemble!(s::XPETScSolver, eqn, partition; component=nothing)
    _set_values!(s.petsclib, s.A, _nzval(_A(eqn)), s.sync, s.fill)
    s.bptr[] = _raw_ptr(_b(eqn, component))
    s
end

# x and b borrow the field and the right-hand side for the solve only; b's owner is the equation,
# which outlives every solve
function _with_placed(f, s::XPETScSolver, x)
    GC.@preserve x begin
        _vec_call(s.place, s.x, _raw_ptr(x))
        _vec_call(s.place, s.b, s.bptr[])
        try
            _sync(s.sync)
            f(s)
            _sync(s.sync)
        finally
            _vec_call(s.reset, s.x)
            _vec_call(s.reset, s.b)
        end
    end
    x
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
    t0 = time_ns()   # INVESTIGATION timer
    r = _with_placed(_ksp_solve, s, x)
    XCALibre.Solve.prof_add!("petsc KSP solve", t0, Int(LibPETSc.KSPGetIterationNumber(s.petsclib, s.ksp)))
    r
end

psolve_transpose!(s::XPETScSolver, x::AbstractVector) = _with_placed(_ksp_solve_transpose, s, x)

_ksp_solve(s) = PETSc.solve!(s.x, s.ksp, s.b)
_ksp_solve_transpose(s) = LibPETSc.KSPSolveTranspose(s.petsclib, s.ksp, s.b, s.x)

# NEW SECTION: parallel partitioning (MatPartitioning on the owned-row cell graph)

for T ∈ (Int32, Int64)
    @eval _ccall_int(f, obj, v::$T) = ccall(f, Cint, (Ptr{Cvoid}, $T), obj, v)
end

# PETSc.jl's MatPartitioningApply wrapper discards the output IS, so the partitioner is called direct
function _parallel_partition(dm::DistributedMesh, method, petsc_options)
    comm = getfield(dm, :comm)
    p = dm.partition
    (; cells, cell_neighbours) = dm.mesh
    n = p.n_owned
    l2g = p.local_to_global
    rowptr, cols = Int[0], Int[]
    for c ∈ 1:n
        append!(cols, sort!(unique(Int(l2g[cell_neighbours[j]]) - 1 for j ∈ cells[c].faces_range)))
        push!(rowptr, length(cols))
    end
    N, nnz = MPI.Allreduce([n, length(cols)], +, comm)
    petsclib = _petsclib(_get_float(dm), nnz)
    PETSc.initialize(petsclib; options=String.(split(petsc_options)))
    _petsc_has_pkg(petsclib, string(method)) || error("repartition: this PETSc build has no $method; " *
        "use a PETSc configured with it (conda-forge's petsc has parmetis and ptscotch)")
    PI, TS = petsclib.PetscInt, petsclib.PetscScalar
    A = LibPETSc.MatCreateMPIAIJWithArrays(petsclib, comm, PI(n), PI(n), PI(N), PI(N),
        PI.(rowptr), PI.(cols), ones(TS, length(cols)))
    sym(name) = _petsc_sym(petsclib, name)
    ok(err, what) = err == 0 || error("repartition: PETSc $what failed ($err)")
    part, is = Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)
    ok(ccall(sym(:MatPartitioningCreate), Cint, (MPI.MPI_Comm, Ptr{Ptr{Cvoid}}), comm, part), "MatPartitioningCreate")
    ok(ccall(sym(:MatPartitioningSetAdjacency), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), part[], A.ptr), "SetAdjacency")
    ok(ccall(sym(:MatPartitioningSetType), Cint, (Ptr{Cvoid}, Cstring), part[], string(method)), "SetType")
    ok(_ccall_int(sym(:MatPartitioningSetNParts), part[], PI(MPI.Comm_size(comm))), "SetNParts")
    ok(ccall(sym(:MatPartitioningSetFromOptions), Cint, (Ptr{Cvoid},), part[]), "SetFromOptions")
    ok(ccall(sym(:MatPartitioningApply), Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), part[], is), "MatPartitioningApply")
    idx = Ref{Ptr{Cvoid}}(C_NULL)
    ok(ccall(sym(:ISGetIndices), Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), is[], idx), "ISGetIndices")
    dest = Int.(copy(unsafe_wrap(Array, Ptr{PI}(idx[]), n)))
    ok(ccall(sym(:ISRestoreIndices), Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), is[], idx), "ISRestoreIndices")
    ok(ccall(sym(:ISDestroy), Cint, (Ptr{Ptr{Cvoid}},), is), "ISDestroy")
    ok(ccall(sym(:MatPartitioningDestroy), Cint, (Ptr{Ptr{Cvoid}},), part), "MatPartitioningDestroy")
    PETSc.destroy(A)
    dest
end

end # module
