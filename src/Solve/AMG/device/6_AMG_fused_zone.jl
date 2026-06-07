# Phase 7 (G3): single-launch fused top-zone GATE (convergence pre-check, CPU, no kernels).
#
# G3's hypothesis: collapse the top 2-3 V-cycle levels (smooth+restrict+coarse+prolong) into ONE
# kernel launch to eliminate launch/inter-level-traffic/occupancy overhead. The binding risk is NOT
# the KA mechanics (5a proved those) but CONVERGENCE: a single launch has no cross-workgroup sync, so
# the in-kernel fine smoother is forced to read off-aggregate neighbours from a STALE global x. For a
# preconditioner cycle x starts at 0 and is never synced mid-kernel, so off-aggregate stays 0 through
# BOTH pre- and post-smooth — i.e. the fused smoother is exactly aggregate-BLOCK-Jacobi (diagonal A
# block per aggregate). That is a strictly weaker operator than the materialized full-Jacobi V-cycle
# and MAY cost CG iterations.
#
# This file measures that iteration cost on the CPU with the existing host hierarchy BEFORE any kernel
# is written: model the top `fuse_depth` levels with block-Jacobi (the exact in-kernel operator), keep
# the materialized hierarchy below intact (integration model B), and drive it through preconditioned CG
# (NOT stationary Richardson — findings flag that as a weak diagnostic that diverges on F1 even when CG
# converges). Gate: if block-smoothing raises CG iters by more than ~20-30%, net time-to-solution loses
# (the fused zone removes <~5% of cycle time) and the kernel is not worth building — keep materialized B/C.

# NEW SECTION: aggregate-block-Jacobi sweep (the exact in-kernel fused smoother)

# Weighted-Jacobi sweep that skips off-aggregate neighbours (agg[j] != agg[i]) — i.e. solves against
# the per-aggregate diagonal block of A only. agg maps a (natural-order) row to its aggregate id.
function _host_block_jacobi_sweep!(xnew, xold, b, A, invdiag, omega, agg)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); T = eltype(xnew)
    @inbounds for i in 1:_m(A)
        gi = agg[i]; sigma = zero(T)
        for p in rp[i]:(rp[i + 1] - 1)
            j = cv[p]; (j == i || agg[j] != gi) && continue
            sigma += nz[p] * xold[j]
        end
        xnew[i] = (one(T) - omega) * xold[i] + omega * invdiag[i] * (b[i] - sigma)
    end
    return xnew
end

# NEW SECTION: host V-cycle with a block-smoothed top zone (integration model B)

# Full-hierarchy host V-cycle identical to _host_ml_vcycle, except levels 1..fuse_depth use the
# block-Jacobi sweep (the fused-kernel operator) instead of full Jacobi. fuse_depth=0 reproduces the
# materialized baseline exactly. aggs[l] is the level-l aggregation (natural order).
function _host_ml_vcycle_block(operators, Ps, omegas, invdiags, aggs, coarse_solve, b, pre, post,
                               fuse_depth)
    L = length(operators); M = L - 1; T = eltype(b)
    xs = [zeros(T, _m(operators[l])) for l in 1:L]
    rhss = Vector{Vector{T}}(undef, L); rhss[1] = copy(b)
    for l in 1:M
        Al = operators[l]; blk = l <= fuse_depth
        x = zeros(T, _m(Al)); tmp = similar(x)
        for _ in 1:pre
            blk ? _host_block_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l], aggs[l]) :
                  _host_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l])
            x, tmp = tmp, x
        end
        xs[l] = x
        r = rhss[l] .- _host_csr_matvec(Al, x)
        rhss[l + 1] = Ps[l]' * r
    end
    xs[L] = coarse_solve(rhss[L])
    for l in M:-1:1
        Al = operators[l]; blk = l <= fuse_depth
        x = xs[l]; tmp = similar(x)
        x .+= Ps[l] * xs[l + 1]
        for _ in 1:post
            blk ? _host_block_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l], aggs[l]) :
                  _host_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l])
            x, tmp = tmp, x
        end
        xs[l] = x
    end
    return xs[1]
end

# NEW SECTION: host preconditioned CG (the predictive convergence metric)

# Standard PCG with a caller-supplied SPD preconditioner apply M⁻¹ (the V-cycle). x0 = 0. Returns
# iterations to ‖r‖ <= rtol·‖b‖ and the final relative residual.
function _host_pcg(A, b, apply_M; itmax::Int=500, rtol=1e-8)
    T = eltype(b); n = length(b)
    x = zeros(T, n); r = copy(b)                      # x0 = 0 -> r0 = b
    bnorm = max(norm(b), eps(T))
    z = apply_M(r); p = copy(z); rz = dot(r, z)
    it = 0
    while it < itmax && norm(r) > rtol * bnorm
        it += 1
        q = _host_csr_matvec(A, p)
        alpha = rz / max(dot(p, q), eps(T))
        @. x += alpha * p
        @. r -= alpha * q
        norm(r) <= rtol * bnorm && break
        z = apply_M(r); rz_new = dot(r, z)
        beta = rz_new / max(rz, eps(T))
        @. p = z + beta * p
        rz = rz_new
    end
    return (iters=it, final_rel=norm(r) / bnorm)
end

# NEW SECTION: the G3 gate driver

# Measure CG iterations of the block-smoothed-top-zone V-cycle vs the materialized baseline, for
# fuse_depth = 0..min(max_depth, M). fuse_depth=0 is the materialized reference (full Jacobi). Returns
# per-depth iters + the ratio to baseline. ratio > ~1.2-1.3 means the fused kernel cannot win net
# time-to-solution (it only removes <~5% of cycle time) -> do NOT build it, keep materialized B/C.
function g3_block_smoother_gate(A, merge_levels::Integer; pre::Int=2, post::Int=2, max_coarse::Integer=64,
                                max_depth::Int=3, itmax::Int=500, rtol=1e-8)
    h = _build_mf_ml(A, merge_levels, CPU(); pre=pre, post=post, max_coarse=max_coarse, fused_top=0)
    T = h.T; n = h.st.n
    csolve = r -> h.coarse_fac \ r
    Am = h.operators[1]
    b = rand(T, n)
    M = h.M
    depths = collect(0:min(max_depth, M))
    iters = Int[]; finals = Float64[]
    for d in depths
        apply_M = r -> _host_ml_vcycle_block(h.operators, h.Ps, h.omegas, h.invdiags, h.aggs,
                                             csolve, r, pre, post, d)
        res = _host_pcg(Am, b, apply_M; itmax=itmax, rtol=rtol)
        push!(iters, res.iters); push!(finals, res.final_rel)
    end
    base = iters[1]
    ratios = [it / base for it in iters]
    return (depths=depths, iters=iters, ratios=ratios, finals=finals,
            levels=length(h.operators), n=n, coarse_n=h.st.coarse_n)
end
