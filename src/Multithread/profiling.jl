# Opt-in phase timers (branch HM/KOmega-profiling). Disabled by default: an instrumented
# call site costs one Ref load when XCPROF[] is false.

export XCPROF, XCPROF_EQN, xcprof!, xcprof_reset!, xcprof_report, @xcprof, @xcprof_eqn

const XCPROF = Ref(false)
const XCPROF_EQN = Ref(:x) # set by the caller so shared phases are attributed per equation
const XCPROF_TIME = Dict{Symbol,Float64}()
const XCPROF_CALLS = Dict{Symbol,Int}()
const XCPROF_UNITS = Dict{Symbol,Int}() # e.g. Krylov iterations

# Phases are recorded as `<eqn>_<phase>`, so the same instrumented function called from the
# U, p, k and omega paths is attributed to each of them separately.
@inline function xcprof!(phase::Symbol, seconds::Float64, units::Int=0)
    XCPROF[] || return nothing
    label = Symbol(XCPROF_EQN[], :_, phase)
    XCPROF_TIME[label] = get(XCPROF_TIME, label, 0.0) + seconds
    XCPROF_CALLS[label] = get(XCPROF_CALLS, label, 0) + 1
    XCPROF_UNITS[label] = get(XCPROF_UNITS, label, 0) + units
    nothing
end

# Names the equation that the enclosing block's phases belong to.
macro xcprof_eqn(sym, expr)
    quote
        if XCPROF[]
            local prev = XCPROF_EQN[]
            XCPROF_EQN[] = $(esc(sym))
            local val = try
                $(esc(expr))
            finally
                XCPROF_EQN[] = prev
            end
            val
        else
            $(esc(expr))
        end
    end
end

# Times `expr` and files it under `phase`. The expression runs exactly once either way.
macro xcprof(phase, expr)
    quote
        if XCPROF[]
            local t0 = time_ns()
            local val = $(esc(expr))
            xcprof!($(esc(phase)), (time_ns() - t0)*1e-9)
            val
        else
            $(esc(expr))
        end
    end
end

xcprof_reset!() = (empty!(XCPROF_TIME); empty!(XCPROF_CALLS); empty!(XCPROF_UNITS); nothing)

function xcprof_report(io::IO=stdout; iterations::Int=1, total::Float64=0.0)
    isempty(XCPROF_TIME) && return println(io, "xcprof: nothing recorded")
    recorded = sum(values(XCPROF_TIME))
    tot = total > 0 ? total : recorded
    rows = sort!(collect(XCPROF_TIME), by=last, rev=true)
    println(io, rpad("phase", 30), lpad("ms/iter", 10), lpad("% wall", 9),
            lpad("calls/it", 10), lpad("units/it", 10))
    for (k, v) in rows
        println(io, rpad(String(k), 30),
            lpad(round(1000v/iterations, digits=3), 10),
            lpad(round(100v/tot, digits=2), 9),
            lpad(round(XCPROF_CALLS[k]/iterations, digits=2), 10),
            lpad(round(XCPROF_UNITS[k]/iterations, digits=2), 10))
    end
    println(io, rpad("SUM(instrumented)", 30), lpad(round(1000recorded/iterations, digits=3), 10),
            lpad(round(100recorded/tot, digits=2), 9))
    println(io, rpad("WALL", 30), lpad(round(1000tot/iterations, digits=3), 10))
    nothing
end
