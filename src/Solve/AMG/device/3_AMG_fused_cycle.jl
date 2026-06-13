# Phase 5b: matrix-free 2-grid V-cycle on the macro layout. The finest-level transfers P0/R0
# are NEVER materialized — restriction is an aggregate reduction, prolongation an aggregate
# broadcast, both keyed off agg_offsets with the Geometric 1/sqrt(w) normalization. The fine
# smoother is the reference weighted-Jacobi (reads off-aggregate neighbours from global x, exact
# — no race, x_old is a separate buffer). Only the small coupled coarse operator Ac=R·A·P (n_macro)
# is materialized. This is the first USEFUL cycle (5a was 1x1 block-diagonal, no smoother): it is
# the first thing that exercises the 1/sqrt(w) normalization (5a's 1x1 solve cancelled it) and it
# captures the bulk of the hierarchy VRAM win (finest P0/R0 are the largest transfer operators).

# NEW SECTION: matrix-free transfer kernels (no P/R stored)

# Restriction rc[g] = (Σ_{k in aggregate g} r[k]) / sqrt(w_g). One thread per macro-aggregate.
@kernel function _amg_mf_restrict_kernel!(rc, @Const(r), @Const(agg_offsets), @Const(inv_sqrt_w))
    g = @index(Global)
    T = eltype(rc)
    lo = Int(agg_offsets[g]); hi = Int(agg_offsets[g + 1]) - 1
    s = zero(T)
    @inbounds for k in lo:hi
        s += r[k]
    end
    @inbounds rc[g] = s * inv_sqrt_w[g]
end

# Prolongation x[k] += xc[g(k)] / sqrt(w_{g(k)}). One thread per fine row (additive correction).
@kernel function _amg_mf_prolong_kernel!(x, @Const(xc), @Const(row_macro), @Const(inv_sqrt_w))
    k = @index(Global)
    @inbounds begin
        g = Int(row_macro[k])
        x[k] += xc[g] * inv_sqrt_w[g]
    end
end

# NEW SECTION: host independent oracle (sparse reference P0/R, plain loops) — validates the kernel
# path AND the 1/sqrt(w) consistency across restrict, prolong, and the RAP-built Ac.

# Weighted-Jacobi sweep matching _amg_jacobi_step_kernel! (residual form, full-row sum).
function _host_jacobi_sweep!(xnew, xold, b, A, invdiag, omega)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); T = eltype(xnew)
    @inbounds for i in 1:_m(A)
        sigma = zero(T)
        for p in rp[i]:(rp[i + 1] - 1)
            sigma += nz[p] * xold[cv[p]]
        end
        xnew[i] = xold[i] + omega * invdiag[i] * (b[i] - sigma)
    end
    return xnew
end

# Full host 2-grid V-cycle (x init 0): pre-smooth, residual, sparse restrict R·r, dense coarse
# solve Ac\rc, sparse prolong P0·xc, post-smooth. Uses the REFERENCE sparse transfers so a wrong
# 1/sqrt(w) in the kernel block formulas surfaces as a mismatch.
function _host_2grid_cycle(Am, P0, R, Acsc, b, invdiag, omega, pre, post)
    n = _m(Am); T = eltype(b)
    x = zeros(T, n); tmp = zeros(T, n)
    for _ in 1:pre
        _host_jacobi_sweep!(tmp, x, b, Am, invdiag, omega); x, tmp = tmp, x
    end
    r = b .- _host_csr_matvec(Am, x)
    xc = Acsc \ (R * r)
    x .+= P0 * xc
    for _ in 1:post
        _host_jacobi_sweep!(tmp, x, b, Am, invdiag, omega); x, tmp = tmp, x
    end
    return x
end

# NEW SECTION: reusable matrix-free 2-grid cycle operator (the building block 5d wires in)

# Per-aggregate 1/sqrt(w) and the permuted row->macro map (the implicit P/R, never a sparse matrix).
function _mf_transfer_factors(agg_offsets, n_macro, n, ::Type{T}) where {T}
    ao = Vector{Int}(agg_offsets)
    inv_sqrt_w = T[one(T) / sqrt(T(ao[g + 1] - ao[g])) for g in 1:n_macro]
    row_macro = Vector{Int32}(undef, n)
    @inbounds for g in 1:n_macro, k in ao[g]:(ao[g + 1] - 1)
        row_macro[k] = Int32(g)
    end
    return inv_sqrt_w, row_macro
end

# Device-resident state for the MF 2-grid cycle, built once (frozen structure). Holds A_perm and the
# matrix-free transfer factors on device, plus the prefactored coupled coarse operator on host (the
# gather/scatter backslash is a placeholder for the device coarse solve wired in a later 5b step).
mutable struct MF2GridState{T, MA, VI, VT, VID, FAC, B}
    A::MA; agg_offsets::VI; inv_sqrt_w::VT; row_macro::VI
    invdiag::VT; diag_index::VID
    coarse_fac::FAC          # lu(Acsc) on host
    cell_perm::Vector{Int32} # permuted position -> original cell (un/permute)
    n::Int; n_macro::Int; omega::T; pre::Int; post::Int; backend::B
    x::VT; tmp::VT; r::VT; rc::VT  # scratch (device)
end

function _build_mf_2grid_state(A, merge_levels::Integer, backend; pre::Int=1, post::Int=1,
                               omega_nominal=4/3)
    Am = _amg_matrix(A)
    n = _m(Am); T = eltype(_nzval(Am))
    agg, P0, _ = build_prolongation(Am, Geometric(merge_levels=Int(merge_levels)))
    R = sparse(P0')
    Acsc = R * _csr_to_csc(Am) * P0
    _, invdiag = _diag_inverse(Am)
    lambda = _estimate_lambda_max(Am, invdiag)
    omega = min(T(omega_nominal), T(2) - eps(T)) / T(lambda)

    lay = build_macro_layout(A, merge_levels)
    agg == Vector{Int}(lay.fine_to_macro) || error("layout aggregation != reference aggregation")
    nm = lay.n_macro
    inv_sqrt_w, row_macro = _mf_transfer_factors(lay.agg_offsets, nm, n, T)
    _, invdiag_perm = _diag_inverse(lay.A_perm)
    diag_index_perm = _diag_index(lay.A_perm)

    dev(v) = backend isa CPU ? copy(v) : Adapt.adapt(backend, v)
    A_dev = AMGMatrixCSR(dev(_rowptr(lay.A_perm)), dev(_colval(lay.A_perm)), dev(_nzval(lay.A_perm)), n, n)
    return MF2GridState(A_dev, dev(lay.agg_offsets), dev(inv_sqrt_w), dev(row_macro),
                        dev(invdiag_perm), dev(diag_index_perm), lu(Acsc), Vector{Int32}(lay.cell_perm),
                        n, nm, omega, pre, post, backend,
                        dev(zeros(T, n)), dev(zeros(T, n)), dev(zeros(T, n)), dev(zeros(T, nm)))
end

# Apply ONE matrix-free 2-grid V-cycle (x init 0) to a device rhs, writing the correction into a fresh
# device vector. rhs is the residual when used as a preconditioner; the cycle solves A·dx ≈ rhs.
function mf_2grid_cycle(st::MF2GridState, rhs_dev)
    T = eltype(st.inv_sqrt_w); bk = st.backend; n = st.n; nm = st.n_macro; wg = 256
    rp, cv, nz = _rowptr(st.A), _colval(st.A), _nzval(st.A)
    _fill_device!(bk, st.x, zero(T))
    xcur, xtmp = st.x, st.tmp
    for _ in 1:st.pre
        _launch_amg_kernel!(bk, wg, _amg_jacobi_step_kernel!, n, xtmp, xcur, rhs_dev, rp, cv, nz, st.invdiag, st.omega)
        xcur, xtmp = xtmp, xcur
    end
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, st.r, rp, cv, nz, xcur, rhs_dev)
    _launch_amg_kernel!(bk, wg, _amg_mf_restrict_kernel!, nm, st.rc, st.r, st.agg_offsets, st.inv_sqrt_w)
    KernelAbstractions.synchronize(bk)
    xc_h = st.coarse_fac \ Array(st.rc)
    xcd = bk isa CPU ? T.(xc_h) : Adapt.adapt(bk, T.(xc_h))
    _launch_amg_kernel!(bk, wg, _amg_mf_prolong_kernel!, n, xcur, xcd, st.row_macro, st.inv_sqrt_w)
    for _ in 1:st.post
        _launch_amg_kernel!(bk, wg, _amg_jacobi_step_kernel!, n, xtmp, xcur, rhs_dev, rp, cv, nz, st.invdiag, st.omega)
        xcur, xtmp = xtmp, xcur
    end
    KernelAbstractions.synchronize(bk)
    return xcur
end

_fill_device!(::CPU, x, v) = (fill!(x, v); x)
_fill_device!(backend, x, v) = (fill!(x, v); KernelAbstractions.synchronize(backend); x)

# Net VRAM the MF path saves vs materializing P0/R0: erase both transfers (n nz each: T value +
# Int32 col) but ADD the implicit-transfer metadata (row_macro Int32 + inv_sqrt_w). Ac is materialized
# either way, so it is not part of the delta.
function _mf_vram_saved_bytes(n, n_macro, ::Type{T}) where {T}
    erased = 2 * n * (sizeof(T) + sizeof(Int32))
    added = n * sizeof(Int32) + n_macro * sizeof(T)
    return erased - added
end

# NEW SECTION: 5b validation drivers

# Correctness: ONE MF 2-grid cycle on the device (permuted space) vs an INDEPENDENT host oracle
# (reference sparse P0/R, plain loops, original space). Both share aggregation/Ac/lambda/omega so this
# isolates the matrix-free transfer + Jacobi kernels (incl. 1/sqrt(w) and the permutation). relerr ~eps
# confirms those. It is BLIND to the shared omega/lambda/coarse-solve — see fused_2grid_convergence.
function fused_2grid_cycle_spike(A, merge_levels::Integer, backend; pre::Int=1, post::Int=1,
                                 omega_nominal=4/3)
    Am = _amg_matrix(A)
    n = _m(Am); T = eltype(_nzval(Am))
    _, P0, _ = build_prolongation(Am, Geometric(merge_levels=Int(merge_levels)))
    R = sparse(P0'); Acsc = R * _csr_to_csc(Am) * P0
    _, invdiag = _diag_inverse(Am)
    lambda = _estimate_lambda_max(Am, invdiag)
    omega = min(T(omega_nominal), T(2) - eps(T)) / T(lambda)
    b = rand(T, n)
    x_oracle = _host_2grid_cycle(Am, P0, R, Acsc, b, invdiag, omega, pre, post)

    st = _build_mf_2grid_state(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal)
    b_perm = Vector{T}(undef, n)
    @inbounds for k in 1:n
        b_perm[k] = b[st.cell_perm[k]]
    end
    bd = st.backend isa CPU ? b_perm : Adapt.adapt(st.backend, b_perm)
    x_perm = Array(mf_2grid_cycle(st, bd))
    x_dev = Vector{T}(undef, n)
    @inbounds for k in 1:n
        x_dev[st.cell_perm[k]] = x_perm[k]
    end
    relerr = maximum(abs.(x_dev .- x_oracle)) / max(maximum(abs.(x_oracle)), eps(T))
    return (relerr=relerr, n=n, n_macro=st.n_macro, pre=pre, post=post,
            vram_saved_bytes=_mf_vram_saved_bytes(n, st.n_macro, T),
            ac_materialized_bytes=length(Acsc.nzval) * (sizeof(T) + sizeof(Int32)))
end

# Convergence (orthogonal to the oracle, which can't see omega/lambda/coarse-solve quality): use the
# MF 2-grid cycle as a stationary preconditioner — x_{k+1}=x_k + cycle(b - A x_k) — and check it
# actually drives ‖r‖ to tol with a sane per-cycle reduction factor (<1; ~0.1-0.3 for 2-grid Poisson).
# A wrong lambda/omega passes the oracle but would stall or diverge here.
function fused_2grid_convergence(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                                 itmax::Int=50, rtol=1e-8, omega_nominal=4/3)
    st = _build_mf_2grid_state(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal)
    T = eltype(st.inv_sqrt_w); n = st.n; bk = st.backend; wg = 256
    rp, cv, nz = _rowptr(st.A), _colval(st.A), _nzval(st.A)
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    b = dev(rand(T, n)); x = dev(zeros(T, n)); res = dev(zeros(T, n))
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
    KernelAbstractions.synchronize(bk)
    r0 = norm(Array(res)); rn = r0; factors = Float64[]; it = 0
    while it < itmax && rn > rtol * r0
        it += 1
        dx = mf_2grid_cycle(st, res)
        _launch_amg_kernel!(bk, wg, _amg_add_kernel!, n, x, dx)
        _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
        KernelAbstractions.synchronize(bk)
        rprev = rn; rn = norm(Array(res))
        push!(factors, rn / rprev)
    end
    return (converged=(rn <= rtol * r0), iters=it, final_rel=rn / r0,
            mean_factor=isempty(factors) ? NaN : sum(factors) / length(factors),
            last_factor=isempty(factors) ? NaN : factors[end], n=n, n_macro=st.n_macro)
end
