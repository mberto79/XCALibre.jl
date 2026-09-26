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

# INVESTIGATION: partition-invariant checksums of each linear system, written while
# CKSUMLOG[] holds an open IO. Sums over the owned rows 1:n of diag(A), |offdiag(A)|, b,
# b^2, x, x^2; the distributed path reduces them over ranks before writing.
const CKSUMLOG = Ref{Any}(nothing)
const CKSUM_HEADER = "field,when,diag,absoff,bsum,bnorm,xsum,xnorm"

function _cksum_local(A, b, x, n)
    rp, cv, nz = _rowptr(A), _colval(A), _nzval(A)
    s = zeros(6)
    @inbounds for i ∈ 1:n
        for k ∈ rp[i]:(rp[i + 1] - 1)
            cv[k] == i ? (s[1] += nz[k]) : (s[2] += abs(nz[k]))
        end
        s[3] += b[i]; s[4] += b[i]^2; s[5] += x[i]; s[6] += x[i]^2
    end
    s
end

_cksum_write(name, when, s) = println(CKSUMLOG[],
    join((name, when, s[1], s[2], s[3], sqrt(s[4]), s[5], sqrt(s[6])), ","))

# INVESTIGATION: XCAL_DUMP=<field>:<k> writes the k-th solve of <field> (CSR, b, x0) to
# system_<field>_<k>.bin before it is solved (format of norm_compare.jl)
const DUMP_COUNT = Dict{String,Int}()
function _maybe_dump(name, A, b, x)
    spec = get(ENV, "XCAL_DUMP", "")
    isempty(spec) && return
    f, k = split(spec, ":"); k = parse(Int, k)
    f == name || return
    c = DUMP_COUNT[name] = get(DUMP_COUNT, name, 0) + 1
    c == k || return
    rp, cv, nz = _rowptr(A), _colval(A), _nzval(A)
    open("system_$(name)_$(k).bin", "w") do io
        write(io, Int64(length(b)), Int64(length(nz)))
        write(io, Int64.(rp), Int64.(cv), Float64.(nz), Float64.(b), Float64.(x))
    end
end
