module XCALibrePETScExt

using XCALibre, MPI, PETSc
using PETSc: LibPETSc
using XCALibre.Distribute
import XCALibre.Distribute: PETScSolver, passemble!, psolve!, psolve_transpose!, piterations,
    _parallel_partition
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval
import XCALibre.Mesh: _get_float

# NEW SECTION: PETSc.jl 0.4/0.5 compatibility

# 0.5 (Julia 1.12+) returns outputs instead of filling Refs and takes Strings for C strings;
# 0.4 serves Julia 1.10. Drop the 0.4 branch once Julia 1.12 is the floor.
const _V05 = pkgversion(PETSc) >= v"0.5"
const _destroy! = _V05 ? PETSc.destroy! : PETSc.destroy

if _V05
    _petsc_has_pkg(petsclib, pkg) = LibPETSc.PetscHasExternalPackage(petsclib, pkg)
    _mat_info(petsclib, M) = LibPETSc.MatGetInfo(petsclib, M, LibPETSc.MAT_LOCAL)
    _mat_set_type(petsclib, A, mt) = LibPETSc.MatSetType(petsclib, A, mt)
else
    _petsc_has_pkg(petsclib, pkg) =
        LibPETSc.PetscHasExternalPackage(petsclib, Vector{Int8}(codeunits(pkg * "\0")))
    _mat_info(petsclib, M) =
        (info = Ref{LibPETSc.MatInfo}(); LibPETSc.MatGetInfo(petsclib, M, LibPETSc.MAT_LOCAL, info); info[])
    _mat_set_type(petsclib, A, mt) =
        GC.@preserve mt LibPETSc.MatSetType(petsclib, A, Base.unsafe_convert(Cstring, mt))
end

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

# the KSP's context for _ksp_converged; vectors are raw handles (C_NULL until needed)
mutable struct _StopTest
    atol::Float64
    rtol::Float64
    target::Float64
    preset::Bool       # target set by the caller (MixedF32 corrections), not from r₀
    keep::Bool         # the target survives it = 0 (a BiCGStab retry continues the same solve)
    explicit::Bool     # ‖D⁻¹r‖ is formed by KSPBuildResidual
    rowscale::Bool     # the KSP solves D⁻¹A·x = D⁻¹b
    gate::Float64
    ratio::Float64     # ‖D⁻¹r‖ over the KSP's norm at the last explicit check
    dinv::LibPETSc.CVec  # 1/|diag(A)| of the unscaled matrix
    bs::LibPETSc.CVec    # D⁻¹b (:rowscale, :shell)
    w::LibPETSc.CVec
    t::LibPETSc.CVec
    p::LibPETSc.CVec     # D·x in the shell PC
    As::Ptr{Cvoid}       # D⁻¹A, the KSP's operator (:shell); A stays the PC's
    inner::Ptr{Cvoid}    # the configured PC, applied by the shell after D
    build::Ptr{Cvoid}    # KSPBuildResidual
    pmult::Ptr{Cvoid}    # VecPointwiseMult
    norm::Ptr{Cvoid}     # VecNorm
end

struct XPETScSolver{PL,TM,TV,TK,SY,FI} <: Distribute.AbstractDistributedSolver
    petsclib::PL
    A::TM
    b::TV
    x::TV
    ksp::TK
    n_owned::Int
    sync::SY           # device-wide sync (nothing on host): PETSc and XCALibre use separate streams
    fill::FI           # host block scatter, or the COO map on device
    place::Ptr{Cvoid}  # Vec(CUDA)PlaceArray: x and b own no storage and borrow the caller's arrays
    reset::Ptr{Cvoid}  # Vec(CUDA)ResetArray
    bptr::Base.RefValue{Ptr{Cvoid}} # b's array, set by passemble! and placed by the next solve
    setup_every::Int   # rebuild the PC every N solves (1 = every solve, PETSc default)
    nsolve::Base.RefValue{Int}
    name::String       # equation label and method, for reporting unconverged solves
    tol::NamedTuple{(:atol,:rtol),Tuple{Float64,Float64}}
    report::Bool       # warn on an unconverged solve; false when an outer loop reports instead
    root::Bool         # the reason is global, so one rank reports it
    rnorm::Base.RefValue{Float64} # ‖D⁻¹(b - Ax)‖₂ of the last unconverged solve
    nwarn::Base.RefValue{Int}     # unconverged solves measured; none after the warning's maxlog
    stop::_StopTest
end

# one PetscInt width is loaded per process, fixed by PETSc's "PetscInt" preference at precompile
# (Int64 unless the user ran PETSc.set_petscint!(Int32)); other precisions need their own env
function _petsclib(TF, nnz_global)
    libs = filter(l -> l.PetscScalar == TF, PETSc.petsclibs)
    isempty(libs) && error("no PETSc library with PetscScalar=$TF (available: " *
        join(("$(l.PetscScalar)/$(l.PetscInt)" for l ∈ PETSc.petsclibs), ", ") *
        "); run in an env whose PETSc preference points at a $TF build")
    fits = filter(l -> nnz_global <= typemax(l.PetscInt), libs)
    isempty(fits) && error("the global matrix has $nnz_global nonzeros, more than the loaded " *
        "$TF PETSc library's $(libs[1].PetscInt) indices can address; run " *
        "PETSc.set_petscint!(Int64) and restart Julia")
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
    isnothing(s.ksp.opts) || _destroy!(s.ksp.opts)
    foreach(_destroy!, (s.ksp, s.A, s.x, s.b))
    c = s.stop
    foreach(v -> _vec_destroy(s.petsclib, getfield(c, v)), (:dinv, :bs, :w, :t, :p))
    c.inner == C_NULL || _petsc_ok(s, :PCDestroy, Ref(c.inner))
    c.As == C_NULL || _petsc_ok(s, :MatDestroy, Ref(c.As))
end

# float_type/nzval: a MixedF32 solve selects the Float32 PETSc library and assembles
# from a Float32 copy of the matrix values; fields stay in the mesh's float type
function _petsc_solver(eqn, dmesh::DistributedMesh, setup;
        comm=getfield(dmesh, :comm), petsc_options="", label="", float_type=_get_float(dmesh),
        nzval=nothing, report=true)
    petsc_options = _options_for(petsc_options, label)
    part = dmesh.partition
    TF = float_type
    A = _A(eqn)
    nzval = something(nzval, _nzval(A))
    rowptr, colval = _host(_rowptr(A)), _host(_colval(A))
    n = part.n_owned
    # owned rows are the contiguous CSR prefix, so the COO values are nzval[1:nnz_owned] in place;
    # ghost rows are garbage and never shipped
    nnz_owned = Int(rowptr[n+1]) - 1
    N, nnz_global = MPI.Allreduce([n, nnz_owned], +, comm)
    petsclib = _petsclib(TF, nnz_global)
    device_solve = !(nzval isa Array)
    # the same string configures PETSc's start-up and the Krylov solve; entries PETSc does not
    # recognise at one stage are consumed at the other. Start-up options apply on the first call
    # only, since PETSc is initialised once per process.
    PETSc.initialize(petsclib; options=String.(split(petsc_options)))
    PI = petsclib.PetscInt
    # device fields never fall back to host solves; a device-enabled PETSc is required.
    # backend ext declares its PETSc pairing (cuda/mpiaijcusparse, hip/mpiaijhipsparse)
    dev = device_solve ? Distribute.petsc_device_info(nzval) : nothing
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
        _split_matrix(petsclib, comm, _rowptr(A), _colval(A), nzval, l2g, n, N)
    _set_values!(petsclib, Amat, nzval, sync, fill)
    vec = device_solve ? dev.vec : _HOST_VEC
    x, b = (_vec_without_array(petsclib, comm, n, N, vec.create) for _ ∈ 1:2)
    curated = merge((; ksp_type=_ksp_type(setup.solver), pc_type=_pc_type(setup.preconditioner)),
        _pc_options(setup.preconditioner))
    raw = isempty(petsc_options) ? (;) : PETSc.parse_options(String.(split(petsc_options)))
    opts = merge(curated, raw)
    stop_opts, mode = _stop_options(opts)
    opts = merge(stop_opts, opts)
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
    # tolerances mean what they mean on serial paths: rtol is relative to the warm-started initial
    # residual, not PETSc's default ||b||; `convergence` is the outer-loop target only. No dtol stop,
    # which serial solves lack. _StopTest replaces PETSc's test; the tolerances set here are for -ksp_view
    (; atol, rtol) = setup
    LibPETSc.KSPSetTolerances(petsclib, ksp, TF(rtol), TF(atol), floatmax(TF), PI(setup.itmax))
    LibPETSc.KSPSetInitialGuessNonzero(petsclib, ksp, LibPETSc.PETSC_TRUE)
    stop = _StopTest(petsclib, b, mode, atol, rtol)
    mode === :none || ccall(_petsc_sym(petsclib, :KSPSetConvergenceTest), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), ksp.ptr, _converged_ptr(PI, real(petsclib.PetscScalar)),
        pointer_from_objref(stop), C_NULL) == 0 || error("PETSc: KSPSetConvergenceTest failed")
    mode === :shell && _shell_pc!(petsclib, ksp, Amat, stop)
    extra = Base.structdiff(opts, (ksp_type=nothing, pc_type=nothing))
    MPI.Comm_rank(comm) == 0 && @info "PETSc solve [$label]: KSP=$(opts.ksp_type) " *
        "PC=$(opts.pc_type) atol=$(TF(atol)) rtol=$(TF(rtol)) itmax=$(setup.itmax)" *
        (isempty(extra) ? "" : " " * join(("$k=$v" for (k, v) ∈ pairs(extra)), " "))
    setup_every = _pc_freeze(setup.preconditioner)
    XPETScSolver(petsclib, Amat, b, x, ksp, n, sync, fill, _petsc_sym(petsclib, vec.place),
        _petsc_sym(petsclib, vec.reset), Ref(C_NULL), setup_every, Ref(0),
        XCALibre.Solve._solve_name(setup, label), (atol=Float64(atol), rtol=Float64(rtol)), report,
        MPI.Comm_rank(comm) == 0, Ref(0.0), Ref(0), stop)
end

# NEW SECTION: stopping rule

# every path stops when ‖D⁻¹(b - Ax)‖₂ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖₂ (PETSc's own test takes the max of the
# two), with the serial GMRES restart. How the KSP meets ‖D⁻¹r‖:
# :monitored  Jacobi, left-preconditioned: the KSP's norm is ‖D⁻¹r‖ (pc_jacobi_abs)
# :rowscale   PCs that commute with row scaling (block ILU, SOR): D⁻¹A·x = D⁻¹b, right-preconditioned
# :shell      bcgs/gmres with AMG or ICC PCs, which need A itself: the KSP's operator is a scaled copy D⁻¹A
#             and a shell PC applies P(A)·D, the serial right preconditioner
# :explicit   CG and CGS with another PC and passthrough types: the KSP's cheap norm predicts when
#             ‖D⁻¹r‖ from KSPBuildResidual (one MatMult, left-preconditioned) is worth computing
# :none       preonly or no norm: PETSc decides
# keyed on the resolved types; user side or norm options keep the KSP's norm and go :explicit
function _stop_options(opts)
    (; ksp_type, pc_type) = opts
    restart = ksp_type == "gmres" ? (; ksp_gmres_restart=XCALibre.Solve.GMRES_RESTART) : (;)
    jacobi = pc_type == "jacobi" ? (; pc_jacobi_abs="true") : (;)
    (ksp_type == "preonly" || get(opts, :ksp_norm_type, "") == "none") && return merge(restart, jacobi), :none
    own = !(haskey(opts, :ksp_pc_side) || haskey(opts, :ksp_norm_type))
    (own && ksp_type ∈ ("cg", "bcgs", "cgs", "gmres")) || return merge(restart, jacobi), :explicit
    pc_type == "jacobi" &&
        return merge(restart, jacobi, (; ksp_pc_side="left", ksp_norm_type="preconditioned")), :monitored
    ksp_type == "cg" && return (; ksp_norm_type="natural"), :explicit
    # cgs unwinds a right preconditioner over the whole iterate, warm start included
    ksp_type == "cgs" && return (; ksp_pc_side="left", ksp_norm_type="preconditioned"), :explicit
    right = merge(restart, (; ksp_pc_side="right", ksp_norm_type="unpreconditioned"))
    right, _commutes_with_rowscale(opts) ? :rowscale : :shell
end

# ILU(k)(D⁻¹A) = D⁻¹·ILU(k)(A) and SOR likewise; ICC, Cholesky and AMG need a symmetric matrix
_commutes_with_rowscale(opts) = opts.pc_type ∈ ("bjacobi", "asm", "ilu", "sor", "none") &&
    string(get(opts, :sub_pc_type, "ilu")) ∈ ("ilu", "sor", "none")

# the explicit check runs once the predicted ‖D⁻¹r‖ is within GATE of the target, then whenever it
# meets it; each check refreshes the prediction's ratio
const GATE = 10.0

function _StopTest(petsclib, b, mode, atol, rtol)
    dup(need) = need ? _vec_duplicate(petsclib, b).ptr : C_NULL
    explicit, shell = mode === :explicit, mode === :shell
    rowscale = shell || mode === :rowscale
    _StopTest(atol, rtol, 0.0, false, false, explicit, rowscale, GATE, 1.0,
        dup(explicit || rowscale), dup(rowscale), dup(explicit), dup(explicit), dup(shell), C_NULL, C_NULL,
        (_petsc_sym(petsclib, f) for f ∈ (:KSPBuildResidual, :VecPointwiseMult, :VecNorm))...)
end

# ‖D⁻¹r‖₂ from KSPBuildResidual (b - Ax, of Aᵀ in a transpose solve) or a PETSc error code
function _explicit_norm(c::_StopTest, ksp, ::Type{PR}) where PR
    V = Ref(c.w)
    err = ccall(c.build, Cint, (Ptr{Cvoid}, LibPETSc.CVec, LibPETSc.CVec, Ptr{LibPETSc.CVec}), ksp, c.t, c.w, V)
    err == 0 || return err, 0.0
    err = ccall(c.pmult, Cint, (LibPETSc.CVec, LibPETSc.CVec, LibPETSc.CVec), V[], V[], c.dinv)
    err == 0 || return err, 0.0
    val = Ref{PR}()
    err = ccall(c.norm, Cint, (LibPETSc.CVec, Cint, Ptr{PR}), V[], 1, val) # 1 = NORM_2
    err, Float64(val[])
end

function _ksp_converged(ksp, it, m, reason, ctx, ::Type{PR}) where PR
    c = unsafe_pointer_to_objref(ctx)::_StopTest
    m = Float64(m)
    r = m
    if c.explicit && isfinite(m)
        if it == 0
            c.gate = GATE
        elseif c.ratio*m > c.gate*c.target
            unsafe_store!(reason, Cint(LibPETSc.KSP_CONVERGED_ITERATING))
            return Cint(0)
        end
        err, r = _explicit_norm(c, ksp, PR)
        err == 0 || return err
        c.ratio = r/max(m, floatmin(Float64))
        it == 0 || (c.gate = 1.0)
    end
    it == 0 && !c.keep && (c.target = XCALibre.Solve._stop_target(c.atol, c.rtol, r))
    code = !isfinite(r) ? LibPETSc.KSP_DIVERGED_NANORINF :
        r > c.target ? LibPETSc.KSP_CONVERGED_ITERATING :
        r ≤ c.atol ? LibPETSc.KSP_CONVERGED_ATOL : LibPETSc.KSP_CONVERGED_RTOL
    unsafe_store!(reason, Cint(code))
    Cint(0)
end

for PI ∈ (Int32, Int64), PR ∈ (Float32, Float64)
    @eval begin
        _ksp_converged(ksp, it::$PI, m::$PR, reason, ctx) = _ksp_converged(ksp, it, m, reason, ctx, $PR)
        _converged_ptr(::Type{$PI}, ::Type{$PR}) =
            @cfunction(_ksp_converged, Cint, (Ptr{Cvoid}, $PI, $PR, Ptr{Cint}, Ptr{Cvoid}))
    end
end

# a MixedF32 correction stops at the refinement's own target, in place of one taken from its r₀
function Distribute.pset_tolerances!(s::XPETScSolver, atol, rtol, target)
    c = s.stop
    c.atol, c.rtol = atol, rtol
    c.preset = !isnothing(target)
    c.preset && (c.target = target)
    s
end

# the KSP's configured PC (built from A) moves behind a shell applying it after D; the KSP's operator
# becomes D⁻¹A, a copy refreshed by passemble!. Options are read at construction, so the swap holds
function _shell_pc!(petsclib, ksp, Amat, c)
    ok(f, args...) = _ptr_call(_petsc_sym(petsclib, f), args...) == 0 || error("PETSc: $f failed")
    inner, shell, As, comm = Ref(C_NULL), Ref(C_NULL), Ref(C_NULL), Ref{MPI.MPI_Comm}()
    ccall(_petsc_sym(petsclib, :MatDuplicate), Cint, (Ptr{Cvoid}, Cint, Ptr{Ptr{Cvoid}}), Amat.ptr, 1, As) == 0 ||
        error("PETSc: MatDuplicate failed") # 1 = MAT_COPY_VALUES
    ok(:KSPGetPC, ksp.ptr, inner)
    ok(:PetscObjectReference, inner[])
    ok(:PetscObjectGetComm, ksp.ptr, comm)
    ccall(_petsc_sym(petsclib, :PCCreate), Cint, (MPI.MPI_Comm, Ptr{Ptr{Cvoid}}), comm[], shell) == 0 ||
        error("PETSc: PCCreate failed")
    ccall(_petsc_sym(petsclib, :PCSetType), Cint, (Ptr{Cvoid}, Cstring), shell[], "shell") == 0 ||
        error("PETSc: PCSetType failed")
    ok(:PCShellSetContext, shell[], pointer_from_objref(c))
    PR = real(petsclib.PetscScalar)
    sym(f) = _petsc_sym(petsclib, f)
    _SHELL_OPS[PR][] = (context=sym(:PCShellGetContext), divide=sym(:VecPointwiseDivide), apply=sym(:PCApply),
        apply_transpose=sym(:PCApplyTranspose), setup=sym(:PCSetUp), reuse=sym(:PCSetReusePreconditioner))
    apply, apply_transpose, setup = _shell_callbacks(PR)
    ok(:PCShellSetApply, shell[], apply)
    ok(:PCShellSetApplyTranspose, shell[], apply_transpose)
    ok(:PCShellSetSetUp, shell[], setup)
    ok(:KSPSetPC, ksp.ptr, shell[])
    ok(:PCDestroy, shell) # the KSP holds it
    ok(:KSPSetOperators, ksp.ptr, As[], Amat.ptr)
    ok(:PCSetOperators, inner[], Amat.ptr, Amat.ptr)
    c.As, c.inner = As[], inner[]
    c
end

# callbacks see only the PC; a process loads one library per scalar type, whose symbols they use
const _SHELL_OPS = Dict(T => Ref((context=C_NULL, divide=C_NULL, apply=C_NULL, apply_transpose=C_NULL, setup=C_NULL,
    reuse=C_NULL)) for T ∈ (Float32, Float64))

_ptr_call(f, a) = ccall(f, Cint, (Ptr{Cvoid},), a)
_ptr_call(f, a, b) = ccall(f, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
_ptr_call(f, a, b, c) = ccall(f, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), a, b, c)

function _shell_context(pc, ops)
    ctx = Ref(C_NULL)
    ccall(ops.context, Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), pc, ctx)
    unsafe_pointer_to_objref(ctx[])::_StopTest
end

# y = P(D·x), D·x = x/dinv
function _shell_apply(pc, x, y, ops)
    c = _shell_context(pc, ops)
    err = _ptr_call(ops.divide, c.p, x, c.dinv)
    err == 0 ? _ptr_call(ops.apply, c.inner, c.p, y) : err
end

# y = D·Pᵀ(x)
function _shell_apply_transpose(pc, x, y, ops)
    c = _shell_context(pc, ops)
    err = _ptr_call(ops.apply_transpose, c.inner, x, c.p)
    err == 0 ? _ptr_call(ops.divide, y, c.p, c.dinv) : err
end

# PCApply sets the inner PC up again once A changes; it rebuilds only here, when the KSP (and its
# freeze count) sets up the shell
function _shell_setup(pc, ops)
    inner = _shell_context(pc, ops).inner
    ccall(ops.reuse, Cint, (Ptr{Cvoid}, Cint), inner, 0)
    err = ccall(ops.setup, Cint, (Ptr{Cvoid},), inner)
    ccall(ops.reuse, Cint, (Ptr{Cvoid}, Cint), inner, 1)
    err
end

for T ∈ (Float32, Float64)
    apply, transpose, setup = Symbol.((:_shell_apply_, :_shell_apply_transpose_, :_shell_setup_), T)
    @eval begin
        $apply(pc, x, y) = _shell_apply(pc, x, y, _SHELL_OPS[$T][])
        $transpose(pc, x, y) = _shell_apply_transpose(pc, x, y, _SHELL_OPS[$T][])
        $setup(pc) = _shell_setup(pc, _SHELL_OPS[$T][])
        _shell_callbacks(::Type{$T}) = (@cfunction($apply, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid})),
            @cfunction($transpose, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid})), @cfunction($setup, Cint, (Ptr{Cvoid},)))
    end
end

function _vec_duplicate(petsclib, v)
    w = Ref{LibPETSc.CVec}(C_NULL)
    ccall(_petsc_sym(petsclib, :VecDuplicate), Cint, (LibPETSc.CVec, Ptr{LibPETSc.CVec}), v, w) == 0 ||
        error("PETSc: VecDuplicate failed")
    LibPETSc.PetscVec(w[], petsclib)
end

_vec_destroy(petsclib, v) = v == C_NULL ||
    ccall(_petsc_sym(petsclib, :VecDestroy), Cint, (Ptr{LibPETSc.CVec},), Ref(v)) == 0 || error("PETSc: VecDestroy failed")

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
    _mat_set_type(petsclib, Amat, mt)
    LibPETSc.MatSetPreallocationCOO(petsclib, Amat, LibPETSc.PetscCount(nnz_owned), coo_i, coo_j)
    Amat, _COOFill(_petsc_sym(petsclib, :MatSetValuesCOO))
end

# PETSc.jl 0.5 wraps MatSetValuesCOO for host Vectors only; device values need the raw pointer
struct _COOFill
    set::Ptr{Cvoid}
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

_seqaij_nnz(petsclib, M) = Int(_mat_info(petsclib, M).nz_used)

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
function _set_values!(petsclib, A, nzval, sync, f::_COOFill)
    _sync(sync)
    GC.@preserve nzval ccall(f.set, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, LibPETSc.InsertMode),
        A.ptr, _raw_ptr(nzval), LibPETSc.INSERT_VALUES) == 0 || error("PETSc: MatSetValuesCOO failed")
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

passemble!(s::XPETScSolver, eqn, partition; component=nothing) =
    passemble!(s, _nzval(_A(eqn)), _b(eqn, component))

function passemble!(s::XPETScSolver, nzval::AbstractVector, b::AbstractVector)
    _set_values!(s.petsclib, s.A, nzval, s.sync, s.fill)
    c = s.stop
    (c.explicit || c.rowscale) && _inverse_abs_diagonal!(s)
    c.As == C_NULL || ccall(_petsc_sym(s.petsclib, :MatCopy), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint), s.A.ptr, c.As, 2) == 0 ||
        error("PETSc: MatCopy failed") # 2 = SAME_NONZERO_PATTERN
    c.rowscale && _petsc_ok(s, :MatDiagonalScale, _operator(s), c.dinv, C_NULL)
    s.bptr[] = _raw_ptr(b)
    s
end

_petsc_ok(s, f, args...) = _petsc_call(_petsc_sym(s.petsclib, f), args...) == 0 || error("PETSc: $f failed")
_petsc_call(f, A::Ptr, v, w) = ccall(f, Cint, (Ptr{Cvoid}, LibPETSc.CVec, Ptr{Cvoid}), A, v, w)
_petsc_call(f, A::Ptr, v::LibPETSc.PetscVec, w::LibPETSc.PetscVec) =
    ccall(f, Cint, (Ptr{Cvoid}, LibPETSc.CVec, LibPETSc.CVec), A, v, w)
_petsc_call(f, v) = ccall(f, Cint, (LibPETSc.CVec,), v)
_petsc_call(f, A::Ptr, v) = ccall(f, Cint, (Ptr{Cvoid}, LibPETSc.CVec), A, v)
_petsc_call(f, u, v, w) = ccall(f, Cint, (LibPETSc.CVec, LibPETSc.CVec, LibPETSc.CVec), u, v, w)

# the KSP's operator: D⁻¹A when row-scaled (A itself, or its copy under :shell)
_operator(s) = s.stop.As == C_NULL ? s.A.ptr : s.stop.As

# dinv = 1/|diag(A)| of the unscaled matrix
function _inverse_abs_diagonal!(s)
    dinv = s.stop.dinv
    _petsc_ok(s, :MatGetDiagonal, s.A.ptr, dinv)
    _petsc_ok(s, :VecAbs, dinv)
    _petsc_ok(s, :VecReciprocal, dinv)
end

# the vectors :monitored solves need only to measure an unconverged solve, kept once made
function _work_vectors!(s)
    c = s.stop
    c.dinv == C_NULL && (c.dinv = _vec_duplicate(s.petsclib, s.b).ptr)
    c.w == C_NULL && (c.w = _vec_duplicate(s.petsclib, s.b).ptr)
    c.t == C_NULL && (c.t = _vec_duplicate(s.petsclib, s.b).ptr)
    c
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
    _with_placed(_solve_with_retries, s, x)
    _report_unconverged(s)
    x
end

# a shadow-vector breakdown (BiCGStab) stops at the last finite iterate; solving again from
# there restarts with the current residual as the shadow, as the serial solves retry theirs. The
# retry keeps the first solve's target
function _solve_with_retries(s)
    c = s.stop
    c.rowscale && _petsc_ok(s, :VecPointwiseMult, c.bs, c.dinv, s.b)
    c.keep = c.preset
    _ksp_solve(s)
    for _ ∈ 1:2
        _broke_down(s) || break
        c.keep = true
        _ksp_solve(s)
    end
    _measure_unconverged!(s)
end

_unconverged(s) = s.report && Integer(LibPETSc.KSPGetConvergedReason(s.petsclib, s.ksp)) < 0

# while x and b are placed; collective, and the reason is the same on every rank. The norm costs a
# MatMult, so it stops once the warning has reached its maxlog
function _measure_unconverged!(s, transpose=false)
    _unconverged(s) || return false
    s.nwarn[] += 1
    s.rnorm[] = s.nwarn[] ≤ XCALibre.Solve.UNCONVERGED_MAXLOG ? _scaled_residual_norm(s, transpose) : NaN
    true
end

# KSPGetResidualNorm is the norm the KSP monitored; this is ‖D⁻¹(b - Ax)‖₂ (b - Aᵀx when transposed)
function _scaled_residual_norm(s::XPETScSolver, transpose)
    c = _work_vectors!(s)
    PR = real(s.petsclib.PetscScalar)
    val = Ref{PR}()
    if c.rowscale && !transpose # the operator is D⁻¹A
        _petsc_ok(s, :MatMult, _operator(s), s.x, c.w)
        _vec_aypx(_petsc_sym(s.petsclib, :VecAYPX), c.w, -one(PR), c.bs)
    else
        c.rowscale || c.explicit || _inverse_abs_diagonal!(s)
        if transpose && c.rowscale # Aᵀx = (D⁻¹A)ᵀ·Dx
            _petsc_ok(s, :VecPointwiseDivide, c.t, s.x, c.dinv)
            _petsc_ok(s, :MatMultTranspose, _operator(s), c.t, c.w)
        else
            _petsc_ok(s, transpose ? :MatMultTranspose : :MatMult, s.A.ptr, s.x, c.w)
        end
        _vec_aypx(_petsc_sym(s.petsclib, :VecAYPX), c.w, -one(PR), s.b)
        _petsc_ok(s, :VecPointwiseMult, c.w, c.w, c.dinv)
    end
    _vec_norm2(_petsc_sym(s.petsclib, :VecNorm), c.w, val) == 0 || error("PETSc: VecNorm failed")
    Float64(val[])
end

for T ∈ (Float32, Float64)
    @eval _vec_aypx(f, y, a::$T, x) = ccall(f, Cint, (LibPETSc.CVec, $T, LibPETSc.CVec), y, a, x)
    @eval _vec_norm2(f, v, val::Ref{$T}) = ccall(f, Cint, (LibPETSc.CVec, Cint, Ptr{$T}), v, 1, val) # 1 = NORM_2
end

# every negative KSPConvergedReason (iteration cap, divergence, breakdown, indefinite PC, NaN)
function _report_unconverged(s::XPETScSolver)
    (_unconverged(s) && s.root) || return nothing
    XCALibre.Solve._warn_unconverged(s.name, string(LibPETSc.KSPGetConvergedReason(s.petsclib, s.ksp)),
        piterations(s), s.rnorm[], "‖D⁻¹(b - Ax)‖₂", s.tol, (:unconverged_solve, s.name))
end

piterations(s::XPETScSolver) = Int(LibPETSc.KSPGetIterationNumber(s.petsclib, s.ksp))

_broke_down(s::XPETScSolver) = LibPETSc.KSPGetConvergedReason(s.petsclib, s.ksp) ∈
    (LibPETSc.KSP_DIVERGED_BREAKDOWN, LibPETSc.KSP_DIVERGED_BREAKDOWN_BICG)

psolve_transpose!(s::XPETScSolver, x::AbstractVector) =
    (_with_placed(_solve_transpose_measured, s, x); _report_unconverged(s); x)
_solve_transpose_measured(s) = (_ksp_solve_transpose(s); _measure_unconverged!(s, true))

# check=false: unconverged solves are reported by _report_unconverged, not by PETSc.jl 0.5
_ksp_solve(s) = _V05 ? PETSc.solve!(s.x, s.ksp, _rhs(s); check=false) : PETSc.solve!(s.x, s.ksp, _rhs(s))
_rhs(s) = s.stop.rowscale ? LibPETSc.PetscVec(s.stop.bs, s.petsclib) : s.b
# (D⁻¹A)ᵀ·y = b with y = Dx; the KSP's norm is then ‖b - Aᵀx‖, so ‖D⁻¹r‖ is checked explicitly
function _ksp_solve_transpose(s)
    c = s.stop
    c.keep = c.preset
    c.rowscale || return LibPETSc.KSPSolveTranspose(s.petsclib, s.ksp, s.b, s.x)
    _work_vectors!(s)
    _petsc_ok(s, :VecPointwiseDivide, s.x, s.x, c.dinv)
    c.explicit = true
    try
        LibPETSc.KSPSolveTranspose(s.petsclib, s.ksp, s.b, s.x)
    finally
        c.explicit = false
    end
    _petsc_ok(s, :VecPointwiseMult, s.x, s.x, c.dinv)
end

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
    _destroy!(A)
    dest
end

end # module
