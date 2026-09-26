# INVESTIGATION ONLY (branch HM/threads-perf-investigation, not for merging): wall-clock
# time per SIMPLE phase and per linear-solve component, to see which parts of the
# threaded path stop scaling. Off unless PROF_ON[] is true.

const PROF_ON = Ref(false)
const PROF_T = Dict{String,Float64}()     # seconds per label
const PROF_N = Dict{String,Int}()         # calls (or Krylov iterations) per label

prof_reset!() = (empty!(PROF_T); empty!(PROF_N); nothing)

@inline function prof_add!(label, t0, n=1)
    PROF_ON[] || return
    PROF_T[label] = get(PROF_T, label, 0.0) + (time_ns() - t0)/1e9
    PROF_N[label] = get(PROF_N, label, 0) + n
    nothing
end

macro prof(label, ex)
    quote
        local t0 = time_ns()
        local val = $(esc(ex))
        prof_add!($(esc(label)), t0)
        val
    end
end

# INVESTIGATION switches for the threaded-divergence tests (read at run time):
#   XCAL_STOCKSTOP=1     Krylov.jl's stock stop (M-norm relative test, as before the fix)
#   XCAL_UPRECON_EACH=1  refresh the Jacobi preconditioner before the Uy and Uz solves too
# and a per-solve log, on while SOLVELOG[] holds an open IO: field, iterations,
# Krylov.jl status, true ||b-Ax||_2 before and after (one extra SpMV each).
stockstop() = get(ENV, "XCAL_STOCKSTOP", "0") == "1"
uprecon_each() = get(ENV, "XCAL_UPRECON_EACH", "0") == "1"

const SOLVELOG = Ref{Any}(nothing)
const SOLVE_NAMES = IdDict{Any,String}()      # field => label, filled by the driver
const SOLVELOG_HEADER = "field,iterations,status,l2_0,l2_end,ratio"

function _solvelog!(eqn, result, solver, values, b, config, l2_0)
    l2 = _residual_norm(eqn, values, b, config)
    status = replace(string(solver.stats.status), "," => ";")
    println(SOLVELOG[], join((get(SOLVE_NAMES, result, "?"), Krylov.iteration_count(solver),
        status, l2_0, l2, l2/l2_0), ","))
    nothing
end
