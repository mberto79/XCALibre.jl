export monitor_linear_solves!, stop_monitoring_linear_solves!
export reset_linear_solve_monitor!, note_solver_iteration!
export linear_solve_records, linear_solve_summary, linear_solve_report
export check_linear_convergence, linear_solve_monitor_active

# =============================================================================
#  Linear-solve convergence monitor  (verification gate G1)
# =============================================================================
#
#  WHY THIS EXISTS
#
#  `solve_system!` previously reported a stalled solve with
#
#      iterations == itmax && @warn "Maximum number of iterations reached!"
#
#  which is easy to lose in a long log, only fires on EXACT equality, and says
#  nothing about how far short the solve stopped. On the LH2 pipe case that
#  allowed 45 of 45 pressure solves to return iterates 3-4 orders of magnitude
#  short of tolerance, silently, for weeks - and every physical conclusion drawn
#  on top of those runs had to be discarded.
#
#  Until a run can state that every linear solve converged, "the case diverged"
#  and "the pressure was never actually solved" are indistinguishable. This
#  monitor makes that statement checkable.
#
#  WHAT IS MEASURED
#
#  The TRUE residual ||b - A*x||_2 recomputed from the returned solution, not
#  the recurrence estimate the Krylov method carries internally - those drift,
#  and a drifted estimate is exactly how a stalled solve reports success. It is
#  compared against the same target the solver was given,
#
#      target = atol + rtol*||b||_2
#
#  A solve is CONVERGED when residual <= target, and AT ITMAX when it used its
#  full iteration budget. The two are different failures: at-itmax means the
#  budget was too small, unconverged-below-itmax means the method stalled or the
#  preconditioner is inadequate.
#
#  COST
#
#  One extra sparse mat-vec per solve, and only while active. The monitor is OFF
#  by default and `record_linear_solve!` returns on its first line when inactive,
#  so a production run pays nothing.
# =============================================================================

"""
    LinearSolveRecord

One linear solve. Fields carry the raw numbers; use [`linear_solve_report`](@ref)
for a readable summary.

- `label`      -- solved field, recovered from `config.solvers` (see notes below)
- `outer`      -- outer iteration/time step, as set by [`note_solver_iteration!`](@ref)
- `n`          -- number of unknowns
- `iterations` -- Krylov iterations used
- `itmax`      -- iteration budget the solve was given
- `residual`   -- `||b - A*x||_2`, recomputed from the returned solution
- `b_norm`     -- `||b||_2`
- `target`     -- `atol + rtol*||b||_2`, the criterion the solver was given
"""
struct LinearSolveRecord
    label::Symbol
    outer::Int
    n::Int
    iterations::Int
    itmax::Int
    residual::Float64
    b_norm::Float64
    target::Float64
end

"""True when the solve used its whole iteration budget."""
at_itmax(r::LinearSolveRecord) = r.iterations >= r.itmax

"""
True when the returned solution meets the tolerance the solver was given, OR
when there was nothing to solve.

The second clause matters. When `||b|| <= target` the zero vector already
satisfies the tolerance, so the system is trivially solved and the reported
residual is round-off noise about a null right-hand side - which happens
routinely for a velocity component in a quiescent field. Counting those as
failures would make the gate fire loudest on exactly the cases that are most at
rest.
"""
is_converged(r::LinearSolveRecord) = r.residual <= r.target || r.b_norm <= r.target

"""How far short the solve stopped, as a multiple of its target."""
shortfall(r::LinearSolveRecord) =
    is_converged(r) ? min(1.0, r.target > 0 ? r.residual/r.target : 1.0) :
    (r.target > 0 ? r.residual/r.target : Inf)

"""
    is_negligible(r, b_peak, negligible) -> Bool

True when the error this solve could have introduced is negligible **on the scale
the field actually operates at over the run**, `b_peak` being the largest `||b||`
seen for that field.

### Why an absolute `atol` cannot do this job

A field's right-hand side is not a fixed quantity. In the rung-0.4 hydrostatic
column the momentum RHS peaks at 5.1e-2 during the startup transient and falls to
2e-6 once the column settles, because a fluid at rest has nothing to solve. Any
absolute tolerance is either far above the working scale (masking real failures)
or far below the quiescent floor (flagging solves of a null system). Setting it
correctly for one mesh moved the problem to the next mesh three times running.

Judging `residual` against `b_peak` is scale-free and needs no tuning per case.
It is deliberately applied at summary level rather than per record, because
`b_peak` is a property of the whole run and is not known when a record is written.
"""
is_negligible(r::LinearSolveRecord, b_peak, negligible) =
    b_peak > 0 && r.residual <= negligible*b_peak

"""Relative residual `||b - A*x||/||b||` - scale-free, unlike `residual`."""
relative_residual(r::LinearSolveRecord) =
    r.b_norm > 0 ? r.residual/r.b_norm : (r.residual == 0 ? 0.0 : Inf)

# The relative tolerance the caller's (atol, rtol) actually amounts to on this
# system. Below ~1e-13 it is asking for more than double precision can deliver on
# anything but a trivial matrix, and a "failure" then says more about the request
# than about the solver.
implied_rtol(r::LinearSolveRecord) = r.b_norm > 0 ? r.target/r.b_norm : Inf

mutable struct LinearSolveMonitor
    active::Bool
    verbose::Bool
    outer::Int
    max_records::Int
    truncated::Bool
    records::Vector{LinearSolveRecord}
end

const LINEAR_SOLVE_MONITOR =
    LinearSolveMonitor(false, false, 0, 500_000, false, LinearSolveRecord[])

"""
    linear_solve_monitor_active() -> Bool

Whether linear solves are currently being recorded.
"""
linear_solve_monitor_active() = LINEAR_SOLVE_MONITOR.active

"""
    monitor_linear_solves!(; verbose=false, max_records=500_000)

Start recording every linear solve. Clears any previous records.

`verbose = true` prints a line for each solve that fails to converge or reaches
`itmax`, as it happens - useful when a run dies before a report can be taken.

Costs one extra sparse mat-vec per solve. Off by default.

```julia
monitor_linear_solves!()
residuals = run!(model, config)
linear_solve_report()
@test check_linear_convergence()
```
"""
function monitor_linear_solves!(; verbose=false, max_records=500_000)
    m = LINEAR_SOLVE_MONITOR
    m.active = true
    m.verbose = verbose
    m.max_records = max_records
    reset_linear_solve_monitor!()
    return nothing
end

"""
    stop_monitoring_linear_solves!()

Stop recording. Records already collected are kept and remain queryable.
"""
function stop_monitoring_linear_solves!()
    LINEAR_SOLVE_MONITOR.active = false
    return nothing
end

"""
    reset_linear_solve_monitor!()

Discard all records and reset the outer-iteration counter, leaving the active
state unchanged.
"""
function reset_linear_solve_monitor!()
    m = LINEAR_SOLVE_MONITOR
    empty!(m.records)
    m.outer = 0
    m.truncated = false
    return nothing
end

"""
    note_solver_iteration!(i)

Tag subsequent records with outer iteration/time step `i`, so a report can say
*which* step the solves failed in rather than only how many failed. Flow solvers
call this at the top of their time loop; it is a no-op when not monitoring.
"""
@inline function note_solver_iteration!(i)
    m = LINEAR_SOLVE_MONITOR
    m.active || return nothing
    m.outer = i
    return nothing
end

"""
    linear_solve_records() -> Vector{LinearSolveRecord}

Every solve recorded since monitoring started, in the order they ran.
"""
linear_solve_records() = LINEAR_SOLVE_MONITOR.records

# -----------------------------------------------------------------------------
#  Label recovery
# -----------------------------------------------------------------------------
#
#  `solve_system!` is handed the `SolverSetup` the caller pulled out of
#  `config.solvers`, so the user-facing field name can be recovered by matching
#  it back against that NamedTuple - no call site needs to pass a name.
#
#  `SolverSetup` is an immutable struct, so `===` is structural: two fields
#  configured identically are indistinguishable. That is reported honestly as a
#  joined label (`Symbol("U|alpha")`) rather than guessed at, because silently
#  attributing a stalled solve to the wrong field is worse than saying it could
#  be either.

function _auto_label(setup, config, component)
    matched = Symbol[]
    for (name, s) in pairs(config.solvers)
        s === setup && push!(matched, name)
    end
    base = if isempty(matched)
        :unknown
    elseif length(matched) == 1
        @inbounds matched[1]
    else
        Symbol(join(matched, "|"))
    end
    return _label_component(base, component)
end

_label_component(base::Symbol, ::Nothing) = base
_label_component(base::Symbol, ::XDir) = Symbol(base, :_x)
_label_component(base::Symbol, ::YDir) = Symbol(base, :_y)
_label_component(base::Symbol, ::ZDir) = Symbol(base, :_z)
_label_component(base::Symbol, ::Any) = base

# -----------------------------------------------------------------------------
#  Recording
# -----------------------------------------------------------------------------

# `R` and `Fx` are the equation's residual scratch arrays. They are reused here
# and then immediately overwritten by `residual()` in the caller, so borrowing
# them costs no allocation and clobbers nothing that is read later.
@kernel inbounds=true function _true_residual!(
    R, Fx, @Const(rowptr), @Const(colval), @Const(nzval), @Const(values), @Const(b))
    i = @index(Global)
    Ax = zero(eltype(R))
    for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        Ax += nzval[nzi]*values[colval[nzi]]
    end
    bi = b[i]
    R[i]  = (bi - Ax)^2
    Fx[i] = bi*bi
end

"""
    record_linear_solve!(phiEqn, setup, component, config, iterations)

Record one solve. Returns immediately when monitoring is off.

Call AFTER the solution has been copied back into `result`, and BEFORE
`residual()` - both use the same scratch arrays, and this ordering leaves
`residual()`'s own values intact.
"""
function record_linear_solve!(phiEqn, setup, component, config, iterations)
    m = LINEAR_SOLVE_MONITOR
    m.active || return nothing

    if length(m.records) >= m.max_records
        m.truncated || @warn "Linear-solve monitor hit max_records; further solves are not recorded." maxlog=1
        m.truncated = true
        return nothing
    end

    (; itmax, atol, rtol) = setup
    (; R, Fx, A) = phiEqn.equation
    b = _b(phiEqn, component)
    # Via the equation's field, not `result`: on the vector path `result` is
    # already the component ScalarField while `component` is still an `XDir`, and
    # only the (VectorField, XDir) pairing has a `get_values` method. This is the
    # same route `residual()` takes.
    values = get_values(get_phi(phiEqn), component)
    (; backend, workgroup) = config.hardware

    ndrange = length(values)
    kernel! = _true_residual!(_setup(backend, workgroup, ndrange)...)
    kernel!(R, Fx, _rowptr(A), _colval(A), _nzval(A), values, b)

    res = sqrt(Float64(sum(R)))
    bn  = sqrt(Float64(sum(Fx)))
    target = Float64(atol) + Float64(rtol)*bn

    label = _auto_label(setup, config, component)
    rec = LinearSolveRecord(label, m.outer, ndrange, Int(iterations), Int(itmax),
                            res, bn, target)
    push!(m.records, rec)

    if m.verbose && (!is_converged(rec) || at_itmax(rec))
        why = at_itmax(rec) ? "at itmax" : "stalled"
        @info "linear solve did not converge ($why)" field=label step=m.outer iterations=rec.iterations residual=res target=target
    end
    return nothing
end

# -----------------------------------------------------------------------------
#  Reporting
# -----------------------------------------------------------------------------

"""
Default for `negligible` in [`linear_solve_summary`](@ref) and
[`check_linear_convergence`](@ref): a residual below this fraction of the field's
peak `||b||` is treated as contributing nothing. 1e-6 is six orders below the
working scale and still many orders above where a genuine stall lives - the LH2
pipe case stalled at a *relative* residual near unity.
"""
const NEGLIGIBLE_RESIDUAL_FRACTION = 1.0e-6

"""
    linear_solve_summary(; negligible=NEGLIGIBLE_RESIDUAL_FRACTION) -> NamedTuple

Aggregate over all records:

- `n_solves`, `n_at_itmax`, `n_unconverged`, `n_negligible`
- `worst_shortfall`  -- largest `residual/target` among solves that were NOT excused
- `worst_field`      -- which field produced it
- `first_failure`    -- the first unexcused non-converged record, or `nothing`
- `by_field`         -- per-field counts, plus that field's peak `||b||`

`n_unconverged == 0` is gate G1. `n_negligible` counts solves that missed their
stated tolerance but whose residual is below `negligible` times the field's peak
`||b||` - see [`is_negligible`](@ref). They are excused, and counted, so that
excusing them stays visible rather than silent.
"""
function linear_solve_summary(; negligible=NEGLIGIBLE_RESIDUAL_FRACTION)
    recs = LINEAR_SOLVE_MONITOR.records
    n = length(recs)
    labels = unique(r.label for r in recs)

    # The field's working scale: the largest RHS it presented over the whole run.
    peak = Dict(l => maximum((r.b_norm for r in recs if r.label === l); init=0.0)
                for l in labels)

    ok(r) = is_converged(r) || is_negligible(r, peak[r.label], negligible)

    n_itmax = count(at_itmax, recs)
    n_bad = count(!ok, recs)
    n_excused = count(r -> !is_converged(r) && ok(r), recs)

    worst = 0.0
    worst_field = :none
    for r in recs
        ok(r) && continue
        sf = shortfall(r)
        if sf > worst
            worst = sf
            worst_field = r.label
        end
    end

    first_failure = nothing
    for r in recs
        if !ok(r)
            first_failure = r
            break
        end
    end

    by_field = NamedTuple(
        l => (
            n_solves      = count(r -> r.label === l, recs),
            n_at_itmax    = count(r -> r.label === l && at_itmax(r), recs),
            n_unconverged = count(r -> r.label === l && !ok(r), recs),
            n_negligible  = count(r -> r.label === l && !is_converged(r) && ok(r), recs),
            max_iterations = maximum((r.iterations for r in recs if r.label === l); init=0),
            b_peak = peak[l],
            worst_shortfall = maximum((shortfall(r) for r in recs
                                       if r.label === l && !ok(r)); init=0.0),
            worst_rel_residual = maximum((relative_residual(r) for r in recs if r.label === l); init=0.0),
            min_implied_rtol = minimum((implied_rtol(r) for r in recs if r.label === l); init=Inf),
        ) for l in labels)

    return (n_solves=n, n_at_itmax=n_itmax, n_unconverged=n_bad,
            n_negligible=n_excused,
            worst_shortfall=worst, worst_field=worst_field,
            first_failure=first_failure, by_field=by_field,
            negligible=negligible,
            truncated=LINEAR_SOLVE_MONITOR.truncated)
end

"""
    linear_solve_report(io=stdout)

Print a per-field table of solve counts, iterations and convergence, and a
verdict line for gate G1.
"""
function linear_solve_report(io::IO=stdout)
    s = linear_solve_summary()
    if s.n_solves == 0
        println(io, "No linear solves recorded. Call `monitor_linear_solves!()` before `run!`.")
        return nothing
    end

    println(io, "\nLinear solve convergence (gate G1)")
    println(io, "-"^78)
    @printf(io, "%-12s %7s %6s %7s %7s %9s %11s %11s\n",
            "field", "solves", "itmax", "unconv.", "neglig.", "max iters",
            "peak |b|", "worst r/tol")
    for (name, v) in pairs(s.by_field)
        @printf(io, "%-12s %7d %6d %7d %7d %9d %11.2e %11.3g\n",
                name, v.n_solves, v.n_at_itmax, v.n_unconverged, v.n_negligible,
                v.max_iterations, v.b_peak, v.worst_shortfall)
    end
    println(io, "-"^78)
    @printf(io, "%-12s %7d %6d %7d %7d %9s %11s %11.3g\n",
            "TOTAL", s.n_solves, s.n_at_itmax, s.n_unconverged, s.n_negligible,
            "", "", s.worst_shortfall)

    s.truncated && println(io, "\n! record buffer was full - counts are a lower bound")

    if any(l -> occursin('|', String(l)), keys(s.by_field))
        println(io, "\n! a joined label (\"a|b\") means those fields were configured with")
        println(io, "  identical `SolverSetup`s and cannot be told apart. Give one of them a")
        println(io, "  distinguishable setting (e.g. a different `itmax`) to separate them.")
    end

    if s.n_negligible > 0
        @printf(io, "\n  %d solve(s) missed their stated tolerance but left a residual below\n",
                s.n_negligible)
        @printf(io, "  %.0e of their field's peak |b|, so they contributed nothing and are\n",
                s.negligible)
        println(io, "  excused. Counted here rather than hidden - see `is_negligible`.")
    end

    if s.n_unconverged == 0
        println(io, "\nG1 PASS - every linear solve reached its tolerance, or left a")
        println(io, "residual negligible on its field's working scale.")
    else
        f = s.first_failure
        println(io, "\nG1 FAIL - $(s.n_unconverged) of $(s.n_solves) solves did not reach tolerance.")
        @printf(io, "  first failure: field %s at step %d, %d iterations (itmax %d)\n",
                f.label, f.outer, f.iterations, f.itmax)
        @printf(io, "                 residual %.4g against target %.4g  (%.3gx short)\n",
                f.residual, f.target, shortfall(f))
        if s.n_at_itmax > 0
            println(io, "  $(s.n_at_itmax) solves used their whole iteration budget - raise `itmax`,")
            println(io, "  but a solve stalling far short of tolerance is a preconditioner problem,")
            println(io, "  not a budget one. On stretched near-wall cells Jacobi is not sufficient.")
        else
            println(io, "  no solve hit `itmax`, so the budget is adequate and the method returned")
            println(io, "  early - check the convergence criteria and the preconditioner.")
        end
        # A tolerance that is unreachable in double precision is a property of the
        # request, not of the solver, and says nothing about the physics.
        tight = [(k, v) for (k, v) in pairs(s.by_field)
                 if v.n_unconverged > 0 && v.min_implied_rtol < 1e-13]
        if !isempty(tight)
            println(io, "\n  NOTE - `atol + rtol*|b|` amounts to a RELATIVE tolerance below 1e-13 for:")
            for (k, v) in tight
                @printf(io, "    %-12s implied rtol %.1e, worst relative residual %.1e\n",
                        k, v.min_implied_rtol, v.worst_rel_residual)
            end
            println(io, "  That is at or below the double-precision floor for a non-trivial system,")
            println(io, "  so this is far more likely an over-tight request than a stalled solve.")
            println(io, "  Raise `atol`/`rtol` to something reachable and re-run before reading")
            println(io, "  anything physical into the failure.")
        end
    end
    println(io)
    return nothing
end

"""
    check_linear_convergence(; slack=10.0, allow_at_itmax=false, verbose=true) -> Bool

Gate G1 as a boolean, for use in a test:

```julia
@test check_linear_convergence()
```

Returns `false` when any recorded solve missed its tolerance by more than
`slack`, or - when `allow_at_itmax = false` - when any solve used its whole
iteration budget. Also returns `false` when nothing was recorded, since a gate
that passes vacuously is worse than no gate.

### Why `slack` is not 1

A Krylov method stops on its own recurrence residual, which is an estimate. The
true `||b - A*x||` recomputed from the returned solution can differ from it by a
small factor - measured at 2.3x for a Bicgstab velocity solve on the rung-0.4
column, on a solve that was otherwise perfectly healthy at a relative residual of
4e-10. Failing the gate on that would train everyone to ignore it.

The failure this gate exists to catch is not a factor of two. On the LH2 pipe
case Jacobi returned pressure iterates that were **three to four orders of
magnitude** short, at a relative residual near unity. A default `slack` of 10
passes recurrence drift and still fires many orders of magnitude before that.

Set `slack = 1` for a strict check; the report always prints the exact numbers
either way, so nothing is hidden by the default.
"""
function check_linear_convergence(; slack=10.0, negligible=NEGLIGIBLE_RESIDUAL_FRACTION,
                                   allow_at_itmax=false, verbose=true)
    s = linear_solve_summary(negligible=negligible)
    if s.n_solves == 0
        verbose && @warn "check_linear_convergence: no solves recorded - was `monitor_linear_solves!()` called before `run!`?"
        return false
    end
    ok = s.n_unconverged == 0 && s.worst_shortfall <= slack &&
         (allow_at_itmax || s.n_at_itmax == 0)
    verbose && !ok && linear_solve_report()
    return ok
end
