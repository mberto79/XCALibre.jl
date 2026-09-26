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
