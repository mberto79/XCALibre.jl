# Phase 5b (Option B): full-hierarchy matrix-free V-cycle. Extends the validated RESULT-1 2-grid
# transfers (device/3_AMG_fused_cycle.jl) to EVERY level: P_l/R_l are NEVER materialized at any level
# — restriction is a per-aggregate contiguous reduction, prolongation an aggregate broadcast, both
# with the Geometric 1/sqrt(w) normalization. Only the coarse operators A_l stay materialized (built
# once by Galerkin RAP on host, permuted into aggregate-contiguous order, Int32 indices, VRAM-resident).
# This is the headline VRAM win: on F1 1.68M it erases ALL P+R = 115.8 MB (vs RESULT-1's finest-only
# 53.7 MB). Deterministic (contiguous reduction, no atomics) so it validates ~eps vs a host oracle.
# Coarsest solve is a host gather/scatter backslash placeholder (device coarse solve is a later step).

# NEW SECTION: position-mapped matrix-free transfer kernels (multilevel: child rows are permuted)

# Restriction rc[coarse_pos[g]] = (Σ_{k in aggregate g} r[k]) / sqrt(w_g). One thread per aggregate.
# coarse_pos maps aggregate id g (this level) to the child level's permuted row position.
@kernel function _amg_mf_restrict_pos_kernel!(rc, @Const(r), @Const(agg_offsets),
                                              @Const(inv_sqrt_w), @Const(coarse_pos))
    g = @index(Global)
    T = eltype(rc)
    lo = Int(agg_offsets[g]); hi = Int(agg_offsets[g + 1]) - 1
    s = zero(T)
    @inbounds for k in lo:hi
        s += r[k]
    end
    @inbounds rc[Int(coarse_pos[g])] = s * inv_sqrt_w[g]
end

# Prolongation x[k] += xc[coarse_pos[g(k)]] / sqrt(w). One thread per fine row (additive correction).
@kernel function _amg_mf_prolong_pos_kernel!(x, @Const(xc), @Const(row_macro),
                                             @Const(inv_sqrt_w), @Const(coarse_pos))
    k = @index(Global)
    @inbounds begin
        g = Int(row_macro[k])
        x[k] += xc[Int(coarse_pos[g])] * inv_sqrt_w[g]
    end
end

# NEW SECTION: matrix-free Galerkin coarse-operator application (top-k fused zone)
# A_l·x = R_{l-1}…R_1 · A_0 · P_1…P_{l-1}: prolong x up to the fine grid, ONE fine SpMV, restrict back.
# This applies any coarse operator EXACTLY using only the materialized fine A_0 + the matrix-free
# transfer factors — so A_1…A_k are never stored (the coarse-operator VRAM win). Cost: one fine SpMV
# per coarse apply (slower), so it is gated to the top `fused_top` levels where n is large.

# Overwrite prolongation y[k] = xc[coarse_pos[g(k)]] / sqrt(w) (the chain needs set, not the += variant).
@kernel function _amg_mf_prolong_set_kernel!(x, @Const(xc), @Const(row_macro),
                                             @Const(inv_sqrt_w), @Const(coarse_pos))
    k = @index(Global)
    @inbounds begin
        g = Int(row_macro[k])
        x[k] = xc[Int(coarse_pos[g])] * inv_sqrt_w[g]
    end
end

# r[i] = b[i] - r[i] (turn a stored A·x into the residual b - A·x in place).
@kernel function _amg_bmar_kernel!(r, @Const(b))
    i = @index(Global)
    @inbounds r[i] = b[i] - r[i]
end

# Weighted-Jacobi correction x[i] += omega·invdiag[i]·r[i] (matrix-free: r is b - A·x precomputed).
@kernel function _amg_jacobi_mf_kernel!(x, @Const(r), @Const(invdiag), omega)
    i = @index(Global)
    @inbounds x[i] += omega * invdiag[i] * r[i]
end

# NEW SECTION: per-level device state + multilevel state

# One transfer level: permuted operator A (aggregate-contiguous rows, Int32 CSR), its smoother data,
# and the matrix-free transfer factors. coarse_pos/row_macro/inv_sqrt_w implicitly encode P_l/R_l.
mutable struct MFLevel{MA, VI, VT, VID, T}
    A::MA                # A_perm_l (device CSR, Int32 indices)
    invdiag::VT; diag_index::VID; omega::T
    agg_offsets::VI      # len n_coarse+1: aggregate g owns permuted rows agg_offsets[g]:agg_offsets[g+1]-1
    inv_sqrt_w::VT       # len n_coarse: 1/sqrt(aggregate size)
    coarse_pos::VI       # len n_coarse: aggregate id -> child permuted row position
    row_macro::VI        # len n: permuted row -> aggregate id
    n::Int; n_coarse::Int
    x::VT; tmp::VT; r::VT; rhs::VT; sc::VT  # scratch (device); sc = Ac for scale_correction
end

mutable struct MFMLState{T, LV, FAC, B, VT, CI, HS, HT}
    levels::LV               # Vector{MFLevel} (transfer levels 1..M, finest first)
    coarse_fac::FAC          # lu(coarsest A) on host (host-LU fallback path)
    coarse_n::Int
    coarse_rhs::VT; coarse_x::VT  # coarsest device buffers (natural order)
    coarse_inv::CI           # device dense inverse for on-device GEMV (nothing -> host LU path)
    coarse_rhs_h::HS; coarse_x_h::HS    # reusable host solve buffers (factorization eltype TF)
    coarse_rhs_hT::HT; coarse_x_hT::HT  # reusable host transfer buffers (type T; alias *_h when T==TF)
    cell_perm::Vector{Int32} # finest permuted position -> original cell
    n::Int; pre::Int; post::Int; omega_nominal::T; backend::B
    fused_top::Int           # top coarse levels (2..fused_top+1) applied matrix-free (A_l not stored)
    coarse_max_rows::Int     # coarsest <= this -> device dense-inverse GEMV, else host LU
    scale_correction::Bool   # GAMG energy-min coarse correction (AMGSolver mode only; fused_top==0)
end

# NEW SECTION: zero-alloc coarsest solve (device dense-inverse GEMV or reusable-buffer host LU)

# Device dense inverse for an on-device GEMV coarse solve (no per-cycle host sync) when the coarsest
# is small enough; else `nothing` -> host LU fallback. Inverse computed in FP64 (accurate, no FP32
# pivot fragility) then stored at the cycle type T. Mirrors reference OnDevice(max_rows). pinv on
# singular. Returns the device-resident inverse (or host Matrix for a CPU backend).
function _build_coarse_dense_inv(coarse_csc, backend, ::Type{T}, max_rows::Integer) where {T}
    n = size(coarse_csc, 1)
    (n == 0 || n > max_rows) && return nothing
    Adense = Matrix(coarse_csc)
    Minv = try
        inv(Adense)
    catch err
        err isa LinearAlgebra.SingularException || rethrow(err)
        pinv(Adense)
    end
    return backend isa CPU ? T.(Minv) : Adapt.adapt(backend, T.(Minv))
end

# Reusable host buffers for the host-LU coarse solve. TF (factorization eltype) may differ from the
# cycle type T (UMFPACK promotes Float32 -> Float64), so transfer buffers are kept at T and alias the
# solve buffers when TF==T (no extra allocation, no per-cycle conversion in the common case).
function _coarse_host_buffers(coarse_fac, ::Type{T}, n::Integer) where {T}
    TF = eltype(coarse_fac)
    rhs_h = zeros(TF, n); x_h = zeros(TF, n)
    rhs_hT = TF === T ? rhs_h : zeros(T, n)
    x_hT = TF === T ? x_h : zeros(T, n)
    return rhs_h, x_h, rhs_hT, x_hT
end

# Coarsest solve, zero-alloc per cycle. GEMV branch: on-device dense inverse (no host sync). LU branch:
# the one surviving host round-trip, but through reusable buffers + in-place ldiv! (UMFPACK caches its
# workspace in the factorization) so it allocates nothing that scales with coarse_n.
function _mf_coarse_solve!(st::MFMLState)
    if st.coarse_inv !== nothing
        mul!(st.coarse_x, st.coarse_inv, st.coarse_rhs)
    else
        KernelAbstractions.synchronize(st.backend)
        copyto!(st.coarse_rhs_hT, st.coarse_rhs)
        st.coarse_rhs_h === st.coarse_rhs_hT || (st.coarse_rhs_h .= st.coarse_rhs_hT)
        ldiv!(st.coarse_x_h, st.coarse_fac, st.coarse_rhs_h)
        st.coarse_x_hT === st.coarse_x_h || (st.coarse_x_hT .= st.coarse_x_h)
        copyto!(st.coarse_x, st.coarse_x_hT)
    end
    return st.coarse_x
end

# Build the materialized A_l hierarchy by Galerkin RAP (host), stopping at max_coarse. operators[l]
# are natural-order AMGMatrixCSR; Ps[l]/aggs[l] are the reference normalized transfers/aggregation.
function _mf_ml_hierarchy(Am, merge_levels::Integer, max_coarse::Integer)
    operators = [Am]; Ps = []; aggs = Vector{Vector{Int}}()
    A_cur = Am
    while _m(A_cur) > max_coarse
        agg, P, _ = build_prolongation(A_cur, Geometric(merge_levels=Int(merge_levels)))
        nc = maximum(agg)
        nc >= _m(A_cur) && break              # no further coarsening possible
        A_next = _amg_matrix(sparse(P') * _csr_to_csc(A_cur) * P)
        push!(aggs, agg); push!(Ps, P); push!(operators, A_next)
        A_cur = A_next
    end
    return operators, Ps, aggs
end

# Build the device-resident multilevel matrix-free state (and return the host hierarchy + per-level
# omega/invdiag so an independent oracle can be built from the SAME operators).
# coarse_storage (default = finest T) sets the precision of levels 2..M + coarsest buffers; level 1
# (A_0, the "fused matrix-free" finest part) is ALWAYS built at the finest type T. Operators are built
# in T (accurate host RAP) then each level's device arrays are downcast to its target type.
function _build_mf_ml(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                      omega_nominal=4/3, max_coarse::Integer=64, fused_top::Integer=0,
                      coarse_max_rows::Integer=512, scale_correction::Bool=false,
                      coarse_storage=nothing)
    Am = _amg_matrix(A); T = eltype(_nzval(Am))
    TC = coarse_storage === nothing ? T : coarse_storage  # levels >=2 + coarsest storage type
    operators, Ps, aggs = _mf_ml_hierarchy(Am, merge_levels, max_coarse)
    L = length(operators); M = L - 1
    M >= 1 || error("matrix too small to coarsen at max_coarse=$max_coarse")

    # Per-level permutation (aggregate-contiguous) + child permutation for coarse_pos.
    cps = Vector{Vector{Int32}}(undef, M); cpis = Vector{Vector{Int32}}(undef, M)
    aos = Vector{Vector{Int32}}(undef, M)
    A_perms = Vector{Any}(undef, M); pvms = Vector{Vector{Int32}}(undef, M)
    for l in 1:M
        nc = maximum(aggs[l])
        cp, cpi, ao = _macro_permutation(aggs[l], nc)
        cps[l] = cp; cpis[l] = cpi; aos[l] = ao
        A_perms[l], pvms[l] = _permuted_operator(operators[l], cp, cpi)
    end

    omegas = Vector{T}(undef, M); invdiags_nat = Vector{Vector{T}}(undef, M)
    for l in 1:M
        _, invd = _diag_inverse(operators[l])
        invdiags_nat[l] = invd
        lambda = _estimate_lambda_max(operators[l], invd)
        omegas[l] = min(T(omega_nominal), T(2) - eps(T)) / T(lambda)
    end

    fused_top = clamp(Int(fused_top), 0, M - 1)  # level 1 (fine A_0) is never matrix-free
    dev(v) = backend isa CPU ? copy(v) : Adapt.adapt(backend, v)
    empty_csr(n) = AMGMatrixCSR(dev(Int32[]), dev(Int32[]), dev(T[]), n, n)
    levels = MFLevel[]
    for l in 1:M
        Tl = l == 1 ? T : TC                                  # finest stays T; coarse levels at TC
        Ap = A_perms[l]; n = _m(Ap); nc = maximum(aggs[l])
        inv_sqrt_w, row_macro = _mf_transfer_factors(aos[l], nc, n, Tl)
        coarse_pos = l < M ? copy(cpis[l + 1]) : Int32.(1:nc)  # coarsest child is natural order
        _, invdiag_perm = _diag_inverse(Ap)
        is_mf = 2 <= l <= fused_top + 1                        # apply A_l matrix-free -> don't store it
        Adev = is_mf ? empty_csr(n) :
               AMGMatrixCSR(dev(_rowptr(Ap)), dev(_colval(Ap)), dev(Tl.(_nzval(Ap))), n, n)
        diag_index_perm = is_mf ? dev(Int[]) : dev(_diag_index(Ap))
        sc = scale_correction ? dev(zeros(Tl, n)) : dev(Tl[])  # Ac scratch (matfree levels too)
        push!(levels, MFLevel(Adev, dev(Tl.(invdiag_perm)), diag_index_perm, Tl(omegas[l]),
                              dev(aos[l]), dev(inv_sqrt_w), dev(coarse_pos), dev(row_macro),
                              n, nc, dev(zeros(Tl, n)), dev(zeros(Tl, n)), dev(zeros(Tl, n)), dev(zeros(Tl, n)), sc))
    end

    coarse_n = _m(operators[L])
    coarse_csc = _csr_to_csc(operators[L])
    coarse_fac = lu(coarse_csc)
    coarse_inv = _build_coarse_dense_inv(coarse_csc, backend, TC, coarse_max_rows)
    rhs_h, x_h, rhs_hT, x_hT = _coarse_host_buffers(coarse_fac, TC, coarse_n)
    st = MFMLState(levels, coarse_fac, coarse_n, dev(zeros(TC, coarse_n)), dev(zeros(TC, coarse_n)),
                   coarse_inv, rhs_h, x_h, rhs_hT, x_hT,
                   Vector{Int32}(cps[1]), _m(Am), pre, post, T(omega_nominal), backend, fused_top,
                   Int(coarse_max_rows), scale_correction)
    return (st=st, operators=operators, Ps=Ps, omegas=omegas, invdiags=invdiags_nat,
            coarse_fac=coarse_fac, T=T, A_perms=A_perms, pvms=pvms, aos=aos, cps=cps,
            cpis=cpis, aggs=aggs, M=M)
end

# k weighted-Jacobi sweeps on lv.A with rhs, result guaranteed in lv.x (ping-pong + copy if odd).
function _mf_smooth!(lv::MFLevel, rhs, k::Int, bk, wg)
    rp, cv, nz = _rowptr(lv.A), _colval(lv.A), _nzval(lv.A)
    src, dst = lv.x, lv.tmp
    for _ in 1:k
        _launch_amg_kernel!(bk, wg, _amg_jacobi_step_kernel!, lv.n, dst, src, rhs, rp, cv, nz,
                            lv.invdiag, lv.omega)
        src, dst = dst, src
    end
    src === lv.x || copyto!(lv.x, src)
    return lv.x
end

_mf_is_matfree(st::MFMLState, l::Int) = 2 <= l <= st.fused_top + 1

# out_l <- A_l·x_l EXACTLY, matrix-free Galerkin: prolong x_l up to the fine grid, ONE fine SpMV on
# the materialized A_0, restrict back to level l. A_l is never stored. cy/cz are finest-sized scratch.
function _mf_apply_operator!(out_l, st::MFMLState, l::Int, x_l, cy, cz, bk, wg)
    nl = st.levels[l].n
    cur, oth = cy, cz
    copyto!(view(cur, 1:nl), view(x_l, 1:nl))
    for j in (l - 1):-1:1                                   # prolong chain level l -> fine
        lv = st.levels[j]
        _launch_amg_kernel!(bk, wg, _amg_mf_prolong_set_kernel!, lv.n, oth, cur,
                            lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
        cur, oth = oth, cur
    end
    A1 = st.levels[1].A                                     # the one materialized fine SpMV
    _launch_amg_kernel!(bk, wg, _amg_csr_matvec_kernel!, st.levels[1].n, oth,
                        _rowptr(A1), _colval(A1), _nzval(A1), cur)
    cur, oth = oth, cur
    for j in 1:(l - 1)                                      # restrict chain fine -> level l
        lv = st.levels[j]
        _launch_amg_kernel!(bk, wg, _amg_mf_restrict_pos_kernel!, lv.n_coarse, oth, cur,
                            lv.agg_offsets, lv.inv_sqrt_w, lv.coarse_pos)
        cur, oth = oth, cur
    end
    copyto!(view(out_l, 1:nl), view(cur, 1:nl))
    return out_l
end

# k matrix-free weighted-Jacobi sweeps at level l (x += omega·invdiag·(rhs - A_l·x)), A_l never stored.
function _mf_smooth_matfree!(st::MFMLState, l::Int, rhs, k::Int, cy, cz, bk, wg)
    lv = st.levels[l]
    for _ in 1:k
        _mf_apply_operator!(lv.r, st, l, lv.x, cy, cz, bk, wg)
        _launch_amg_kernel!(bk, wg, _amg_bmar_kernel!, lv.n, lv.r, rhs)
        _launch_amg_kernel!(bk, wg, _amg_jacobi_mf_kernel!, lv.n, lv.x, lv.r, lv.invdiag, lv.omega)
    end
    return lv.x
end

# GAMG scale_correction up-sweep. Replaces plain x += P·xc with the energy-minimising x += sf·c,
# sf=(r_l·c)/(c·Ac). r_l = lv.r, the down-sweep post-pre-smoothing residual (still valid: lv.x is
# untouched between the down-sweep residual and this call), so no residual recompute is paid.
# Matrix-free levels get Ac via the Galerkin chain (_mf_apply_operator!, one fine SpMV).
function _mf_coarse_correction_scaled!(st::MFMLState, l::Int, child_x, cy, cz, bk, wg)
    lv = st.levels[l]
    T = eltype(lv.x)
    c = lv.tmp
    _launch_amg_kernel!(bk, wg, _amg_mf_prolong_set_kernel!, lv.n, c, child_x,
                        lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
    if _mf_is_matfree(st, l)
        _mf_apply_operator!(lv.sc, st, l, c, cy, cz, bk, wg)                                      # Ac
    else
        rp, cv, nz = _rowptr(lv.A), _colval(lv.A), _nzval(lv.A)
        _launch_amg_kernel!(bk, wg, _amg_csr_matvec_kernel!, lv.n, lv.sc, rp, cv, nz, c)          # Ac
        # level 1's lv.r aliases the matfree chain scratch cz; with fused levels above it is stale by
        # the time the up-sweep reaches l=1, so recompute (cz is free here: deeper levels are done)
        l == 1 && st.fused_top > 0 &&
            _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, lv.n, lv.r, rp, cv, nz, lv.x, lv.rhs)
    end
    sf = _amg_scale_factor(T(dot(lv.r, c)), T(dot(c, lv.sc)))
    _launch_amg_kernel!(bk, wg, _amg_axpy_kernel!, lv.n, lv.x, sf, c)
    return lv.x
end

# Apply ONE V-cycle (x init 0) to a device rhs (finest permuted order). The top `fused_top` coarse
# levels are applied matrix-free (A_l not stored); the rest stay materialized; coarsest = host
# backslash. Down: smooth, residual, restrict. Up: prolong + post-smooth. Returns finest correction.
function mf_ml_cycle(st::MFMLState, rhs_dev)
    bk = st.backend; wg = 256; M = length(st.levels); T = eltype(st.coarse_rhs)
    cy = st.levels[1].tmp; cz = st.levels[1].r  # finest-sized chain scratch (free once level 1 is done)
    lv1 = st.levels[1]; copyto!(lv1.rhs, rhs_dev)
    # down sweep
    for l in 1:M
        lv = st.levels[l]
        _fill_device!(bk, lv.x, zero(T))
        if _mf_is_matfree(st, l)
            _mf_smooth_matfree!(st, l, lv.rhs, st.pre, cy, cz, bk, wg)
            _mf_apply_operator!(lv.r, st, l, lv.x, cy, cz, bk, wg)
            _launch_amg_kernel!(bk, wg, _amg_bmar_kernel!, lv.n, lv.r, lv.rhs)  # lv.r = rhs - A_l x
        else
            _mf_smooth!(lv, lv.rhs, st.pre, bk, wg)
            rp, cv, nz = _rowptr(lv.A), _colval(lv.A), _nzval(lv.A)
            _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, lv.n, lv.r, rp, cv, nz, lv.x, lv.rhs)
        end
        child_rhs = l < M ? st.levels[l + 1].rhs : st.coarse_rhs
        _launch_amg_kernel!(bk, wg, _amg_mf_restrict_pos_kernel!, lv.n_coarse, child_rhs, lv.r,
                            lv.agg_offsets, lv.inv_sqrt_w, lv.coarse_pos)
    end
    # coarsest solve (zero-alloc: device GEMV or reusable-buffer host LU)
    _mf_coarse_solve!(st)
    # up sweep
    for l in M:-1:1
        lv = st.levels[l]
        child_x = l < M ? st.levels[l + 1].x : st.coarse_x
        if st.scale_correction
            _mf_coarse_correction_scaled!(st, l, child_x, cy, cz, bk, wg)
        else
            _launch_amg_kernel!(bk, wg, _amg_mf_prolong_pos_kernel!, lv.n, lv.x, child_x,
                                lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
        end
        # Coarse levels post-smooth against lv.r (the pre-smoothing residual), matching the reference
        # _cycle! rhs aliasing (level.rhs is overwritten by _residual! at levels >= 2). Empirically a
        # stronger cycle: F1 standalone 151 -> 121 iters, == reference; Cg iters unchanged (89 == 89).
        post_rhs = l == 1 ? lv.rhs : lv.r
        if _mf_is_matfree(st, l)
            copyto!(lv.tmp, post_rhs)  # _mf_smooth_matfree! uses lv.r as scratch; avoid aliasing
            _mf_smooth_matfree!(st, l, lv.tmp, st.post, cy, cz, bk, wg)
        else
            _mf_smooth!(lv, post_rhs, st.post, bk, wg)
        end
    end
    KernelAbstractions.synchronize(bk)
    return st.levels[1].x
end

# NEW SECTION: independent host oracle (reference sparse P_l / A_l, natural order, plain loops)

function _host_ml_vcycle(operators, Ps, omegas, invdiags, coarse_solve, b, pre, post;
                         scale_correction::Bool=false)
    L = length(operators); M = L - 1; T = eltype(b)
    xs = [zeros(T, _m(operators[l])) for l in 1:L]
    rhss = Vector{Vector{T}}(undef, L); rhss[1] = copy(b)
    for l in 1:M
        Al = operators[l]
        x = zeros(T, _m(Al)); tmp = similar(x)
        for _ in 1:pre
            _host_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l]); x, tmp = tmp, x
        end
        xs[l] = x
        r = rhss[l] .- _host_csr_matvec(Al, x)
        rhss[l + 1] = Ps[l]' * r
    end
    xs[L] = coarse_solve(rhss[L])
    for l in M:-1:1
        Al = operators[l]
        x = xs[l]; tmp = similar(x)
        r_pre = rhss[l] .- _host_csr_matvec(Al, x)           # x still holds the pre-smoothed iterate
        c = Ps[l] * xs[l + 1]
        if scale_correction
            Ac = _host_csr_matvec(Al, c)
            sf = _amg_scale_factor(T(dot(r_pre, c)), T(dot(c, Ac)))
            x .+= sf .* c
        else
            x .+= c
        end
        post_rhs = l == 1 ? rhss[l] : r_pre                  # mirror mf_ml_cycle / reference aliasing
        for _ in 1:post
            _host_jacobi_sweep!(tmp, x, post_rhs, Al, invdiags[l], omegas[l]); x, tmp = tmp, x
        end
        xs[l] = x
    end
    return xs[1]
end

# Net VRAM saved by erasing ALL P_l/R_l vs materializing them. The reference stores P AND R as full
# CSRs (AMGLevel has both), so erased counts both nzval+colval AND their rowptr arrays (P has n+1, R
# has nc+1). Added = the implicit-transfer metadata (row_macro n, inv_sqrt_w nc, coarse_pos nc,
# agg_offsets nc+1). A_l are materialized either way so excluded from the delta. NOTE the gross win is
# structurally small under Geometric: unsmoothed piecewise-constant P has ONE nonzero per row, so
# P-as-CSR ≈ row_macro+inv_sqrt_w already — the bulk of operator VRAM lives in the coarse A_l (kept).
function _mf_ml_vram_saved_bytes(Ps, ::Type{T}) where {T}
    erased = 0; added = 0
    for l in eachindex(Ps)
        nnzP = nnz(Ps[l]); n = size(Ps[l], 1); nc = size(Ps[l], 2)
        erased += nnzP * (sizeof(T) + sizeof(Int32)) + (n + 1) * sizeof(Int32)    # P_l (CSR)
        erased += nnzP * (sizeof(T) + sizeof(Int32)) + (nc + 1) * sizeof(Int32)   # R_l (CSR, =Pᵀ)
        added += n * sizeof(Int32) + nc * (sizeof(T) + 2 * sizeof(Int32)) + (nc + 1) * sizeof(Int32)
    end
    return erased - added
end

# NEW SECTION: 5b Option-B validation drivers

# Correctness: ONE full-hierarchy MF V-cycle (device, permuted) vs the independent host oracle
# (reference sparse P_l, natural order). Both share operators/omega/coarse-solve so this isolates the
# matrix-free transfers + permutation + 1/sqrt(w) across ALL levels. relerr ~eps confirms them.
function mf_ml_cycle_spike(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                           omega_nominal=4/3, max_coarse::Integer=64, coarse_max_rows::Integer=512,
                           scale_correction::Bool=false)
    h = _build_mf_ml(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, coarse_max_rows=coarse_max_rows,
                     scale_correction=scale_correction)
    st = h.st; T = h.T; n = st.n
    b = rand(T, n)
    # oracle coarse solve must match the state's mechanism (GEMV inv vs LU) for ~eps parity
    Minv_h = st.coarse_inv === nothing ? nothing : Array(st.coarse_inv)
    csolve = Minv_h === nothing ? (r -> h.coarse_fac \ r) : (r -> Minv_h * r)
    x_oracle = _host_ml_vcycle(h.operators, h.Ps, h.omegas, h.invdiags, csolve, b, pre, post;
                               scale_correction=scale_correction)
    b_perm = Vector{T}(undef, n)
    @inbounds for k in 1:n
        b_perm[k] = b[st.cell_perm[k]]
    end
    bd = backend isa CPU ? b_perm : Adapt.adapt(backend, b_perm)
    x_perm = Array(mf_ml_cycle(st, bd))
    x_dev = Vector{T}(undef, n)
    @inbounds for k in 1:n
        x_dev[st.cell_perm[k]] = x_perm[k]
    end
    relerr = maximum(abs.(x_dev .- x_oracle)) / max(maximum(abs.(x_oracle)), eps(T))
    return (relerr=relerr, n=n, levels=length(st.levels) + 1, pre=pre, post=post,
            vram_saved_bytes=_mf_ml_vram_saved_bytes(h.Ps, T))
end

# Convergence (orthogonal to the oracle): use the MF V-cycle as a stationary preconditioner and check
# it drives ‖r‖ to tol with a sane per-cycle factor (<1). A wrong omega/coarse-solve passes the oracle
# but stalls/diverges here.
function mf_ml_convergence(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                           itmax::Int=80, rtol=1e-8, omega_nominal=4/3, max_coarse::Integer=64,
                           fused_top::Integer=0, scale_correction::Bool=false)
    h = _build_mf_ml(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, fused_top=fused_top, scale_correction=scale_correction)
    st = h.st; T = h.T; n = st.n; bk = st.backend; wg = 256
    Afine = st.levels[1].A; rp, cv, nz = _rowptr(Afine), _colval(Afine), _nzval(Afine)
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    # work in the finest PERMUTED space (A_perm consistent with the cycle)
    b = dev(rand(T, n)); x = dev(zeros(T, n)); res = dev(zeros(T, n))
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
    KernelAbstractions.synchronize(bk)
    r0 = norm(Array(res)); rn = r0; factors = Float64[]; it = 0
    while it < itmax && rn > rtol * r0
        it += 1
        dx = mf_ml_cycle(st, res)
        _launch_amg_kernel!(bk, wg, _amg_add_kernel!, n, x, dx)
        _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
        KernelAbstractions.synchronize(bk)
        rprev = rn; rn = norm(Array(res))
        push!(factors, rn / rprev)
    end
    return (converged=(rn <= rtol * r0), iters=it, final_rel=rn / r0,
            mean_factor=isempty(factors) ? NaN : sum(factors) / length(factors),
            last_factor=isempty(factors) ? NaN : factors[end], n=n, levels=length(st.levels) + 1)
end

# NEW SECTION: G2 zero-alloc validation

# G2: per-cycle host allocation after the zero-alloc coarse handoff. The old path allocated O(coarse_n)
# every cycle (Array() + backslash + T.() + adapt); the new path reuses buffers / GEMVs on device.
# Success = a1==a2 (constant, no growth) and small/independent of coarse_n. KA kernel launches add a
# fixed host allocation that cannot be removed, so we report the bytes, not assert a literal zero.
function mf_ml_cycle_allocs(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                            omega_nominal=4/3, max_coarse::Integer=64, fused_top::Integer=0,
                            coarse_max_rows::Integer=512)
    h = _build_mf_ml(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, fused_top=fused_top, coarse_max_rows=coarse_max_rows)
    st = h.st; T = h.T; n = st.n
    rhs = backend isa CPU ? rand(T, n) : Adapt.adapt(backend, rand(T, n))
    mf_ml_cycle(st, rhs)                                # warmup (compile + cuBLAS handle)
    a1 = @allocated mf_ml_cycle(st, rhs)
    a2 = @allocated mf_ml_cycle(st, rhs)
    return (host_bytes=(a1, a2), constant=(a1 == a2), coarse_n=st.coarse_n,
            branch=(st.coarse_inv === nothing ? :host_lu : :gemv), levels=length(st.levels) + 1)
end

# NEW SECTION: top-k matrix-free Galerkin validation + VRAM accounting

# Device bytes of the coarse operators A_l (l=2..fused_top+1) that the top-k path does NOT store.
function _mf_topk_vram_erased_bytes(operators, fused_top, ::Type{T}) where {T}
    bytes = 0
    for l in 2:(fused_top + 1)
        l > length(operators) && break
        nnzA = length(_nzval(operators[l])); n = _m(operators[l])
        bytes += nnzA * (sizeof(T) + sizeof(Int32)) + (n + 1) * sizeof(Int32)
    end
    return bytes
end

# Correctness of the top-k matrix-free path: ONE cycle with fused_top=k (A_2..A_{k+1} applied via the
# Galerkin chain) vs fused_top=0 (those A_l materialized). The chain computes the EXACT operator, so
# the two must agree to ~eps (slightly looser, ~1e-13, from longer fp accumulation). A mismatch means
# a chain composition bug (normalization / coarse_pos ordering). Same A/permutation in both states.
function mf_ml_topk_error(A, merge_levels::Integer, backend, fused_top::Integer; pre::Int=2,
                          post::Int=2, omega_nominal=4/3, max_coarse::Integer=64)
    hk = _build_mf_ml(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top)
    h0 = _build_mf_ml(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=0)
    T = hk.T; n = hk.st.n
    b = rand(T, n)
    bd = backend isa CPU ? b : Adapt.adapt(backend, b)
    xk = Array(mf_ml_cycle(hk.st, bd))
    x0 = Array(mf_ml_cycle(h0.st, bd))
    relerr = maximum(abs.(xk .- x0)) / max(maximum(abs.(x0)), eps(T))
    return (relerr=relerr, n=n, levels=length(hk.st.levels) + 1, fused_top=hk.st.fused_top,
            vram_erased_bytes=_mf_topk_vram_erased_bytes(hk.operators, hk.st.fused_top, T))
end
