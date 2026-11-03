# NEW SECTION: preconditioner × solve path

# `solver` is always the outer iteration (Krylov, or Richardson for AMG); `preconditioner` is the
# method. Every rule on what runs where lives here; PETSc option strings live in the PETSc ext.
struct Substitute
    name::String
end
struct Unsupported
    msg::String
end

const _F16_JACOBI = Unsupported("MixedF16 supports the Jacobi preconditioner only; use MixedF32")
const _AMG_MIXED = Unsupported("mixed precision does not apply to AMG; use AMG(coarse_storage=Float32) " *
    "for a single-precision hierarchy")
const _HYPRE_ONLY = Unsupported("BoomerAMG() runs hypre through PETSc; on a serial mesh call distribute(mesh) " *
    "to solve through PETSc in a single process, or use AMG(coarsening=...)")

# columns: serial full precision, serial MixedF32, serial MixedF16, distributed (PETSc)
const PRECONDITIONER_PATHS = (
    Jacobi       = (full=:same,      f32=:same,      f16=:same,       petsc=:same),
    NormDiagonal = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("Jacobi")),
    DILU         = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ILU(0)")),
    ILU0GPU      = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ILU(0)")),
    IC0GPU       = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ICC(0)")),
    AMG          = (full=:same,      f32=_AMG_MIXED, f16=_AMG_MIXED,  petsc=:same), # PETSc: GAMG, see _gamg_conflicts
    BoomerAMG    = (full=_HYPRE_ONLY, f32=_HYPRE_ONLY, f16=_HYPRE_ONLY, petsc=:same),
)

# outer iterations each path runs; the serial full-precision path is AMG's for AMG
const SOLVER_PATHS = (
    krylov = (Cg, Cgs, Bicgstab, Gmres),
    amg    = (Cg, Bicgstab, Richardson),
    mixed  = (Cg, Cgs, Bicgstab, Gmres),
    petsc  = (Cg, Cgs, Bicgstab, Gmres, Richardson),
)

_column(::FullPrecision) = :full
_column(::MixedF32) = :f32
_column(::MixedF16) = :f16

_solver_column(::FullPrecision, ::AMG) = :amg
_solver_column(::FullPrecision, pc) = :krylov
_solver_column(::AbstractMixedPrecision, pc) = :mixed

# types missing from the table (user-defined) run as named serially and need petsc_options in PETSc
_pc_rule(pc, col::Symbol) = begin
    name = nameof(typeof(pc))
    haskey(PRECONDITIONER_PATHS, name) && return PRECONDITIONER_PATHS[name][col]
    col === :petsc ? Unsupported("no PETSc mapping for preconditioner $name; name one with " *
        "petsc_options=\"-pc_type ...\"") : :same
end
_pc_rule(pc::AMG, col::Symbol) = col === :petsc && !isempty(_gamg_conflicts(pc)) ?
    Unsupported(join(_gamg_conflicts(pc), "; ")) : PRECONDITIONER_PATHS.AMG[col]

function _solver_rule(solver, col::Symbol)
    allowed = SOLVER_PATHS[col]
    any(S -> solver isa S, allowed) && return :same
    names = join(("$(nameof(S))()" for S ∈ allowed), ", ")
    Unsupported(col === :amg ? "AMG runs with solver = $names" :
        col === :mixed ? "mixed precision supports $names" :
        "solver $(nameof(typeof(solver)))() needs preconditioner=AMG(...)")
end

# serial path of a SolverSetup, then the distributed one when the precision reaches PETSc
_serial_rules(s) = (_solver_rule(s.solver, _solver_column(s.precision, s.preconditioner)),
    _pc_rule(s.preconditioner, _column(s.precision)))
_petsc_rules(s) = s.precision isa MixedF16 ? (Unsupported("MixedF16 has no distributed path"),) :
    (_solver_rule(s.solver, :petsc), _pc_rule(s.preconditioner, :petsc))

_unsupported(rules) = findfirst(r -> r isa Unsupported, rules)

# SolverSetup refuses a setting only when no path can run it
function _check_any_path(setup)
    serial = _serial_rules(setup)
    i = _unsupported(serial)
    isnothing(i) || isnothing(_unsupported(_petsc_rules(setup))) || throw(ArgumentError(serial[i].msg))
    setup
end

# serial check before a workspace is built: throws an Unsupported, warns once for a Substitute
function _check_serial(setup)
    rules = _serial_rules(setup)
    i = _unsupported(rules)
    isnothing(i) || throw(ArgumentError(rules[i].msg))
    _warn_substitute(setup.preconditioner, rules[2], "on a serial mesh")
    setup
end

_warn_substitute(pc, rule, where) = nothing
_warn_substitute(pc, rule::Substitute, where) =
    @warn "$(nameof(typeof(pc)))() is not available $where; substituting $(rule.name)" maxlog=1 _id=(
        nameof(typeof(pc)), where)

# NEW SECTION: AMG on PETSc

# on a distributed mesh AMG runs as PETSc's GAMG (options spelled in the PETSc ext); settings GAMG
# cannot express stop the run, settings of the native implementation only are listed and skipped
function _gamg_conflicts(a::AMG)
    c, s = a.coarsening, a.smoother
    msgs = String[]
    c isa RugeStuben && push!(msgs, "AMG(coarsening=RugeStuben()) has no PETSc counterpart; use BoomerAMG() " *
        "for classical AMG on a distributed mesh")
    c isa SmoothAggregation && c.smoother_weight != 4/3 && push!(msgs, "SmoothAggregation(smoother_weight) has no PETSc counterpart")
    c isa SmoothAggregation && !isnothing(c.near_nullspace) && push!(msgs, "SmoothAggregation(near_nullspace) has no PETSc counterpart")
    c isa Geometric && c.merge_levels != 1 && push!(msgs, "Geometric(merge_levels) has no PETSc counterpart")
    a.pre_sweeps == a.post_sweeps || push!(msgs, "PETSc smooths as often before as after; set pre_sweeps == post_sweeps")
    s isa AMGSOR && s.omega != 1 && push!(msgs, "PETSc's SOR smoother in GAMG takes omega = 1; use AMGGaussSeidel()")
    s isa AMGChebyshev && a.pre_sweeps != 1 && push!(msgs, "AMGChebyshev with PETSc takes pre_sweeps = post_sweeps = 1; raise degree instead")
    msgs
end

const _NATIVE_ONLY = (:scale_correction, :coarse_solve, :fuse_levels, :min_coarse_rows, :coarse_storage)

# native-implementation settings changed from their defaults, which PETSc does not use
_native_only_set(a::AMG) = (ref = AMG(coarsening=a.coarsening, smoother=a.smoother);
    [k for k ∈ _NATIVE_ONLY if getfield(a, k) != getfield(ref, k)])

# SolverSetup is built on every rank; Distribute points this at its is_root
const _LOG_RANK = Ref{Function}(() -> true)

# the outer iteration is `solver=`; AMG(mode=...) is the old spelling
function _resolve_amg(pc::AMG, solver)
    isnothing(pc.mode) || pc.mode isa typeof(solver) || !_LOG_RANK[]() ||
        @warn "AMG(mode=$(nameof(typeof(pc.mode)))()) is deprecated and ignored: the outer iteration " *
            "is SolverSetup's solver=$(nameof(typeof(solver)))()" maxlog=1
    solver isa Union{Cg,Bicgstab,Richardson} ? _amg_with_mode(pc, solver) : pc
end
_resolve_amg(pc, solver) = pc

# deprecated SolverSetup(solver=AMG(mode=...)): AMG becomes the preconditioner, its mode the solver
_deprecated_amg_solver(solver, preconditioner) = (solver, preconditioner)
function _deprecated_amg_solver(amg::AMG, preconditioner)
    solver = something(amg.mode, Cg())
    _LOG_RANK[]() && @warn "SolverSetup(solver=AMG(...)) is deprecated; use SolverSetup(solver=" *
        "$(nameof(typeof(solver)))(), preconditioner=AMG(...)). The `preconditioner` given is unused." maxlog=1
    return solver, _amg_with_mode(amg, solver)
end

# AMG keeps its own preconditioner, so the equation holds an inert one that nothing refreshes;
# BoomerAMG lives in PETSc, and its serial workspace check stops the run
_set_preconditioner(::FullPrecision, setup::SolverSetup{F,I,S1,S2,<:AMG}, eqn, args...) where {F,I,S1,S2} =
    _inert_preconditioner()
set_preconditioner(::Union{AMG,BoomerAMG}, eqn, args...) = _inert_preconditioner()

_workspace(::FullPrecision, setup::SolverSetup{F,I,S1,S2,<:AMG}, eqn, b) where {F,I,S1,S2} =
    _workspace(setup.preconditioner, b, _index_type(_A(eqn)))
