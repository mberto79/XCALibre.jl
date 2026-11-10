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
const _AMG_F16 = Unsupported("MixedF16 does not apply to AMG; MixedF32 stores its coarse levels in Float32")
const _HYPRE_ONLY = Unsupported("BoomerAMG() runs hypre through PETSc; on a serial mesh call distribute(mesh) " *
    "to solve through PETSc in a single process, or use AMG()")

# columns: serial full precision, serial MixedF32, serial MixedF16, distributed (PETSc)
const PRECONDITIONER_PATHS = (
    Jacobi       = (full=:same,      f32=:same,      f16=:same,       petsc=:same),
    NormDiagonal = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("Jacobi")),
    DILU         = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ILU(0)")),
    ILU0GPU      = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ILU(0)")),
    IC0GPU       = (full=:same,      f32=:same,      f16=_F16_JACOBI, petsc=Substitute("per-rank block ICC(0)")),
    AMG          = (full=:same,      f32=:same,      f16=_AMG_F16,    petsc=:same), # PETSc: GAMG, see _lower_gamg
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
_solver_column(::MixedF32, ::AMG) = :amg
_solver_column(::FullPrecision, pc) = :krylov
_solver_column(::AbstractMixedPrecision, pc) = :mixed

# types missing from the table (user-defined) run as named serially and need petsc_options in PETSc
_pc_rule(pc, col::Symbol) = begin
    name = nameof(typeof(pc))
    haskey(PRECONDITIONER_PATHS, name) && return PRECONDITIONER_PATHS[name][col]
    col === :petsc ? Unsupported("no PETSc mapping for preconditioner $name; name one with " *
        "petsc_options=\"-pc_type ...\"") : :same
end
_pc_rule(pc::AMG, col::Symbol) = col === :petsc && !isempty(last(_lower_gamg(pc))) ?
    Unsupported(join(last(_lower_gamg(pc)), "; ")) : PRECONDITIONER_PATHS.AMG[col]

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

# AMG on a distributed mesh is PETSc's GAMG. One function translates every setting into PETSc options
# or refuses it, so the two cannot drift apart; NativeAMG settings apply to serial meshes only.
function _lower_gamg(a::AMG, solver=Cg())
    refused = String[]
    refuse(msg) = push!(refused, msg)
    s = a.sweeps
    s.grow == 0 || refuse("Sweeps(grow) has no PETSc counterpart: GAMG fixes its level count during setup")
    opts = (pc_gamg_reuse_interpolation=true, pc_mg_cycle_type="v", pc_gamg_coarse_eq_limit=a.coarsest,
        pc_gamg_aggressive_coarsening=0)
    # GAMG aggregates on a symmetric graph; a non-symmetric matrix must be symmetrised first
    solver isa Cg || (opts = merge(opts, (pc_gamg_graph_symmetrize=true,)))
    opts = merge(opts, _gamg_coarsening(a.coarsening, refuse), _gamg_smoother(a.smoother, s, refuse))
    return opts, refused
end

function _gamg_coarsening(c::SmoothedAggregation, refuse)
    c.weight == oftype(c.weight, 4/3) || refuse("SmoothedAggregation(weight) has no PETSc counterpart")
    isnothing(c.near_nullspace) || refuse("SmoothedAggregation(near_nullspace) has no PETSc counterpart")
    (pc_gamg_agg_nsmooths=1, pc_gamg_threshold=c.threshold)
end
# GAMG forms its own (larger than pairwise) aggregates, so merges sets nothing there
_gamg_coarsening(c::Pairwise, refuse) = (pc_gamg_agg_nsmooths=0,)
_gamg_coarsening(::RugeStuben, refuse) =
    (refuse("RugeStuben() has no PETSc counterpart; use BoomerAMG() for classical AMG on a distributed mesh"); (;))

# down sweeps on every level; unequal up sweeps through a separate up smoother
_gamg_sweeps(s::Sweeps) = s.pre == s.post ? (mg_levels_ksp_max_it=s.pre,) :
    (pc_mg_distinct_smoothup=true, mg_levels_ksp_max_it=s.pre, mg_levels_up_ksp_max_it=s.post)

# Jacobi damps by weight/λmax per level; PETSc's Richardson scale is fixed, so weight/2 assumes λmax ≈ 2
_gamg_smoother(j::Jacobi, s, refuse) = merge((mg_levels_ksp_type="richardson", mg_levels_pc_type="jacobi",
    mg_levels_ksp_richardson_scale=j.weight / 2), _gamg_sweeps(s))
function _gamg_smoother(c::Chebyshev, s, refuse)
    s.pre == s.post == 1 || refuse("Chebyshev() with PETSc takes sweeps = 1; raise degree instead")
    (mg_levels_ksp_type="chebyshev", mg_levels_pc_type="jacobi", mg_levels_ksp_max_it=c.degree,
        mg_levels_ksp_chebyshev_esteig="0,$(c.lambda_scale/c.eig_ratio),0,$(c.lambda_scale)")
end
# PETSc's SOR sweeps within each rank
function _gamg_smoother(g::SweepSmoother, s, refuse)
    _sweep_omega(g) == 1 || refuse("PETSc's SOR smoother in GAMG takes omega = 1; use GaussSeidel() or SymGaussSeidel()")
    merge((mg_levels_ksp_type="richardson", mg_levels_pc_type="sor", mg_levels_pc_sor_its=1),
        _sor_order(_sweep_order(g)), _gamg_sweeps(s))
end
_sor_order(::_Forward) = (mg_levels_pc_sor_local_forward=true,)
_sor_order(::_Symmetric) = (mg_levels_pc_sor_local_symmetric=true,)

# settings changed from their defaults that PETSc does not use: NativeAMG and Pairwise(merges)
function _native_only(a::AMG)
    n, d = a.native, NativeAMG()
    changed = ["$k=$(getfield(n, k))" for k ∈ fieldnames(NativeAMG) if getfield(n, k) != getfield(d, k)]
    unused = isempty(changed) ? String[] : ["NativeAMG(" * join(changed, ", ") * ")"]
    _pairwise_unused(a.coarsening, unused)
    join(unused, " or ")
end
_pairwise_unused(c, unused) = unused
_pairwise_unused(c::Pairwise, unused) =
    c.merges == Pairwise().merges ? unused : push!(unused, "Pairwise(merges=$(c.merges))")

# hypre's options for BoomerAMG
_lower_hypre(b::BoomerAMG) = (pc_hypre_boomeramg_strong_threshold=b.threshold,
    pc_hypre_boomeramg_coarsen_type=_hypre_name(b.coarsen), pc_hypre_boomeramg_interp_type=_hypre_name(b.interpolation),
    pc_hypre_boomeramg_P_max=b.max_interp, pc_hypre_boomeramg_agg_nl=b.aggressive, pc_hypre_boomeramg_agg_num_paths=2,
    pc_hypre_boomeramg_relax_type_all=_hypre_name(b.smoother),
    pc_hypre_boomeramg_grid_sweeps_down=b.sweeps[1], pc_hypre_boomeramg_grid_sweeps_up=b.sweeps[2])
_hypre_name(::HMIS) = "HMIS"
_hypre_name(::PMIS) = "PMIS"
_hypre_name(::Falgout) = "Falgout"
_hypre_name(::ExtI) = "ext+i"
_hypre_name(::Classical) = "classical"
_hypre_name(::GaussSeidel) = "SOR/Jacobi"
_hypre_name(::SymGaussSeidel) = "symmetric-SOR/Jacobi"

# SolverSetup is built on every rank; Distribute points this at its is_root
const _LOG_RANK = Ref{Function}(() -> true)

# AMG(mode=...) of earlier releases: the outer iteration is SolverSetup's solver=
function _resolve_amg(pc::_AMGMode, solver, precision)
    _LOG_RANK[]() && @warn "AMG(mode=$(nameof(typeof(pc.mode)))()) is deprecated and ignored: the outer iteration " *
        "is SolverSetup's solver=$(nameof(typeof(solver)))()" maxlog=1
    _resolve_amg(pc.amg, solver, precision)
end
# MixedF32 keeps the AMG outer iteration in full precision and stores the coarse levels in Float32
_resolve_amg(pc::AMG, solver, precision) =
    (_check_amg_solver(pc, solver); precision isa MixedF32 ? _with_storage(pc, Float32) : pc)
_resolve_amg(pc, solver, precision) = pc

# SolverSetup(solver=AMG(...)) of earlier releases: AMG becomes the preconditioner, its mode the solver
_deprecated_amg_solver(solver, preconditioner) = (solver, preconditioner)
_deprecated_amg_solver(amg::AMG, preconditioner) = _deprecated_amg_solver(_AMGMode(Cg(), amg), preconditioner)
function _deprecated_amg_solver(s::_AMGMode, preconditioner)
    _LOG_RANK[]() && @warn "SolverSetup(solver=AMG(...)) is deprecated; use SolverSetup(solver=" *
        "$(nameof(typeof(s.mode)))(), preconditioner=AMG(...)). The `preconditioner` given is unused." maxlog=1
    return s.mode, s.amg
end

# AMG keeps its own preconditioner, so the equation holds an inert one that nothing refreshes;
# BoomerAMG lives in PETSc, and its serial workspace check stops the run
const _AMGSetup = SolverSetup{F,I,S1,S2,<:AMG} where {F,I,S1,S2}
_set_preconditioner(::FullPrecision, setup::_AMGSetup, eqn, args...) = _inert_preconditioner()
_set_preconditioner(::MixedF32, setup::_AMGSetup, eqn) = _inert_preconditioner()
_set_preconditioner(::MixedF32, setup::_AMGSetup, eqn, BCs, config) = _inert_preconditioner()
set_preconditioner(::Union{AMG,BoomerAMG}, eqn, args...) = _inert_preconditioner()

_workspace(::FullPrecision, setup::_AMGSetup, eqn, b) =
    _workspace(setup.preconditioner, b, _index_type(_A(eqn)); solver=setup.solver)
_workspace(::MixedF32, setup::_AMGSetup, eqn, b) =
    _workspace(setup.preconditioner, b, _index_type(_A(eqn)); solver=setup.solver)
