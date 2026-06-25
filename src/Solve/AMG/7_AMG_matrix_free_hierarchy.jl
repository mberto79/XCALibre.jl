# Full-hierarchy matrix-free V-cycle: P_l/R_l never materialized (per-aggregate reduce/broadcast with
# 1/sqrt(w)); only coarse operators A_l stay materialized (Galerkin RAP, permuted, Int32 indices).

# NEW SECTION: operator permutation + transfer factors (host build, used by the hierarchy + refresh)

_amg_index_type(n::Integer) = n <= typemax(Int32) ? Int32 : Int64

_fill_device!(::CPU, x, v) = (fill!(x, v); x)
_fill_device!(backend, x, v) = (fill!(x, v); KernelAbstractions.synchronize(backend); x)

# Group fine cells by aggregate id into contiguous blocks (counting sort); returns permutation
# (new->old, old->new) and CSR-style aggregate offsets.
function aggregate_permutation(fine_to_macro, n_macro)
    n = length(fine_to_macro)
    counts = zeros(Int, n_macro)
    @inbounds for i in 1:n
        counts[fine_to_macro[i]] += 1
    end
    agg_offsets = Vector{Int32}(undef, n_macro + 1)
    agg_offsets[1] = 1
    @inbounds for m in 1:n_macro
        agg_offsets[m + 1] = agg_offsets[m] + counts[m]
    end
    cell_perm = Vector{Int32}(undef, n)
    cell_perm_inv = Vector{Int32}(undef, n)
    pos = Vector{Int}(agg_offsets[1:n_macro])
    @inbounds for i in 1:n
        m = fine_to_macro[i]
        k = pos[m]; pos[m] += 1
        cell_perm[k] = i
        cell_perm_inv[i] = k
    end
    return cell_perm, cell_perm_inv, agg_offsets
end

# Build Pc·A·Pcᵀ (symmetric permutation) in CSR + the slot map back to the original nzval (value-only
# refresh restreams through it). Int32 indices: the MF cycle drives A_perm through KA kernels, not cuSPARSE.
function permute_operator(A, cell_perm, cell_perm_inv)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A)
    n = _m(A); nnz = length(nz); T = eltype(nz)
    TI = _amg_index_type(max(n, nnz + 1))
    new_rowptr = Vector{TI}(undef, n + 1); new_rowptr[1] = 1
    new_colval = Vector{TI}(undef, nnz)
    new_nzval = Vector{T}(undef, nnz)
    perm_value_map = Vector{Int32}(undef, nnz)
    cols = Int[]; slots = Int[]
    slot = 1
    @inbounds for k in 1:n
        old_i = cell_perm[k]
        empty!(cols); empty!(slots)
        for p in rp[old_i]:(rp[old_i + 1] - 1)
            push!(cols, cell_perm_inv[cv[p]]); push!(slots, p)
        end
        for o in sortperm(cols)
            new_colval[slot] = cols[o]
            new_nzval[slot] = nz[slots[o]]
            perm_value_map[slot] = slots[o]
            slot += 1
        end
        new_rowptr[k + 1] = slot
    end
    return AMGMatrixCSR(new_rowptr, new_colval, new_nzval, n, n), perm_value_map
end

# Per-aggregate 1/sqrt(w) and the permuted row->aggregate map (the implicit P/R, never a sparse matrix).
function transfer_factors(agg_offsets, n_macro, n, ::Type{T}) where {T}
    ao = Vector{Int}(agg_offsets)
    inv_sqrt_w = T[one(T) / sqrt(T(ao[g + 1] - ao[g])) for g in 1:n_macro]
    row_macro = Vector{Int32}(undef, n)
    @inbounds for g in 1:n_macro, k in ao[g]:(ao[g + 1] - 1)
        row_macro[k] = Int32(g)
    end
    return inv_sqrt_w, row_macro
end

# NEW SECTION: position-mapped matrix-free transfer kernels (multilevel: child rows are permuted)

# Restriction rc[coarse_pos[g]] = (Σ_{k in aggregate g} r[k]) / sqrt(w_g). One thread per aggregate.
# coarse_pos maps aggregate id g (this level) to the child level's permuted row position.
@kernel function _amg_matrix_free_restrict_kernel!(rc, @Const(r), @Const(agg_offsets),
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
@kernel function _amg_matrix_free_prolong_add_kernel!(x, @Const(xc), @Const(row_macro),
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
@kernel function _amg_matrix_free_prolong_set_kernel!(x, @Const(xc), @Const(row_macro),
                                             @Const(inv_sqrt_w), @Const(coarse_pos))
    k = @index(Global)
    @inbounds begin
        g = Int(row_macro[k])
        x[k] = xc[Int(coarse_pos[g])] * inv_sqrt_w[g]
    end
end

# r[i] = b[i] - r[i] (turn a stored A·x into the residual b - A·x in place).
@kernel function _amg_residual_in_place_kernel!(r, @Const(b))
    i = @index(Global)
    @inbounds r[i] = b[i] - r[i]
end

# Weighted-Jacobi correction x[i] += omega·invdiag[i]·r[i] (matrix-free: r is b - A·x precomputed).
@kernel function _amg_jacobi_correction_kernel!(x, @Const(r), @Const(invdiag), omega)
    i = @index(Global)
    @inbounds x[i] += omega * invdiag[i] * r[i]
end

# NEW SECTION: per-level device state + multilevel state

# One transfer level: permuted operator A (aggregate-contiguous rows, Int32 CSR), its smoother data,
# and the matrix-free transfer factors. coarse_pos/row_macro/inv_sqrt_w implicitly encode P_l/R_l.
mutable struct MatrixFreeLevel{MA, VI, VT, VID, T}
    A::MA                # A_perm_l (device CSR, Int32 indices)
    invdiag::VT; diag_index::VID; omega::T
    agg_offsets::VI      # len n_coarse+1: aggregate g owns permuted rows agg_offsets[g]:agg_offsets[g+1]-1
    inv_sqrt_w::VT       # len n_coarse: 1/sqrt(aggregate size)
    coarse_pos::VI       # len n_coarse: aggregate id -> child permuted row position
    row_macro::VI        # len n: permuted row -> aggregate id
    n::Int; n_coarse::Int
    x::VT; tmp::VT; r::VT; rhs::VT; sc::VT  # scratch (device); sc = Ac for scale_correction
end

# Device matrix-free hierarchy AND the workspace.hierarchy on the matrix-free path (absorbs the former
# MFGreenfield wrapper: refresh_plan/cell_perm_device/residual_permuted + the outer-loop fields the
# materialised AMGHierarchy carries). coarse_fac/coarse_inv/refresh_plan are Ref{Any} so the empty and
# built hierarchies share one concrete type (workspace.hierarchy reassignment after build type-checks).
mutable struct MatrixFreeHierarchy{T, LV, B, VT, VR, VI, HS, HT} <: AbstractAMGHierarchy
    levels::LV               # Vector{MatrixFreeLevel} (transfer levels 1..M, finest first)
    coarse_fac::Base.RefValue{Any}  # lu(coarsest A) on host (host-LU fallback path)
    coarse_n::Int
    coarse_rhs::VT; coarse_x::VT  # coarsest device buffers (natural order, coarse storage type)
    coarse_inv::Base.RefValue{Any}  # device dense inverse for on-device GEMV (nothing -> host LU path)
    coarse_rhs_h::HS; coarse_x_h::HS    # reusable host solve buffers (factorization eltype TF)
    coarse_rhs_hT::HT; coarse_x_hT::HT  # reusable host transfer buffers (type T; alias *_h when T==TF)
    cell_perm::Vector{Int32} # finest permuted position -> original cell (host)
    cell_perm_device::VI     # cell_perm on device (gather/scatter the outer residual/correction)
    residual_permuted::VR    # device scratch: residual gathered into permuted order (working type T)
    n::Int; pre::Int; post::Int; omega_nominal::T; backend::B
    fused_top::Int           # top coarse levels (2..fused_top+1) applied matrix-free (A_l not stored)
    coarse_max_rows::Int     # coarsest <= this -> device dense-inverse GEMV, else host LU
    scale_correction::Bool   # GAMG energy-min coarse correction (AMGSolver mode only; fused_top==0)
    refresh_plan::Base.RefValue{Any}  # MatrixFreeRefreshPlan (frozen-sparsity device refresh); nothing until built
    nrows::Int; nnz::Int     # pattern guard: rebuild on a change
    is_symmetric::Bool       # Cg gate reads it
    last_cycle_factor::Float64
    workgroup::Int           # gather/scatter launch workgroup (cycle internals fix wg=256)
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

# Typed-but-unbuilt matrix-free hierarchy, mirroring _empty_hierarchy: same concrete type as a built
# one (placeholder lu gives the host-buffer eltype TF; empty device/host buffers match build's types) so
# update! can reassign workspace.hierarchy after the first build. Populated on first update! call.
function _empty_matrix_free_hierarchy(backend, ::Type{T}, ::Type{TS}=T) where {T,TS}
    da(v) = backend isa CPU ? copy(v) : Adapt.adapt(backend, v)
    placeholder = lu(sparse([1], [1], [one(T)], 1, 1))
    rhs_h, x_h, rhs_hT, x_hT = _coarse_host_buffers(placeholder, TS, 0)
    return MatrixFreeHierarchy(MatrixFreeLevel[], Ref{Any}(nothing), 0,
                   da(zeros(TS, 0)), da(zeros(TS, 0)), Ref{Any}(nothing),
                   rhs_h, x_h, rhs_hT, x_hT,
                   Int32[], da(Int32[]), da(zeros(T, 0)),
                   0, 2, 2, T(4 / 3), backend, 0, 512, false,
                   Ref{Any}(nothing), 0, 0, true, 0.0, 256)
end

_amg_empty_hierarchy(::MatrixFreeAMG, backend, ::Type{T}, ::Type{TS}) where {T,TS} =
    _empty_matrix_free_hierarchy(backend, T, TS)

# Coarsest solve, zero-alloc per cycle. GEMV branch: on-device dense inverse (no host sync). LU branch:
# the one surviving host round-trip, but through reusable buffers + in-place ldiv! (UMFPACK caches its
# workspace in the factorization) so it allocates nothing that scales with coarse_n.
function solve_coarsest_level!(st::MatrixFreeHierarchy)
    coarse_inv = st.coarse_inv[]
    if coarse_inv !== nothing
        mul!(st.coarse_x, coarse_inv, st.coarse_rhs)
    else
        KernelAbstractions.synchronize(st.backend)
        copyto!(st.coarse_rhs_hT, st.coarse_rhs)
        st.coarse_rhs_h === st.coarse_rhs_hT || (st.coarse_rhs_h .= st.coarse_rhs_hT)
        ldiv!(st.coarse_x_h, st.coarse_fac[], st.coarse_rhs_h)
        st.coarse_x_hT === st.coarse_x_h || (st.coarse_x_hT .= st.coarse_x_h)
        copyto!(st.coarse_x, st.coarse_x_hT)
    end
    return st.coarse_x
end

# Build the materialized A_l hierarchy by Galerkin RAP (host), stopping at max_coarse. operators[l]
# are natural-order AMGMatrixCSR; Ps[l]/aggs[l] are the reference normalized transfers/aggregation.
function build_galerkin_operators(Am, merge_levels::Integer, max_coarse::Integer)
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
function build_matrix_free_hierarchy(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                      omega_nominal=4/3, max_coarse::Integer=64, fused_top::Integer=0,
                      coarse_max_rows::Integer=512, scale_correction::Bool=false,
                      coarse_storage=nothing)
    Am = _amg_matrix(A); T = eltype(_nzval(Am))
    TC = coarse_storage === nothing ? T : coarse_storage  # levels >=2 + coarsest storage type
    operators, Ps, aggs = build_galerkin_operators(Am, merge_levels, max_coarse)
    L = length(operators); M = L - 1
    M >= 1 || error("matrix too small to coarsen at max_coarse=$max_coarse")

    # Per-level permutation (aggregate-contiguous) + child permutation for coarse_pos.
    cps = Vector{Vector{Int32}}(undef, M); cpis = Vector{Vector{Int32}}(undef, M)
    aos = Vector{Vector{Int32}}(undef, M)
    A_perms = Vector{Any}(undef, M); pvms = Vector{Vector{Int32}}(undef, M)
    for l in 1:M
        nc = maximum(aggs[l])
        cp, cpi, ao = aggregate_permutation(aggs[l], nc)
        cps[l] = cp; cpis[l] = cpi; aos[l] = ao
        A_perms[l], pvms[l] = permute_operator(operators[l], cp, cpi)
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
    levels = MatrixFreeLevel[]
    for l in 1:M
        Tl = l == 1 ? T : TC                                  # finest stays T; coarse levels at TC
        Ap = A_perms[l]; n = _m(Ap); nc = maximum(aggs[l])
        inv_sqrt_w, row_macro = transfer_factors(aos[l], nc, n, Tl)
        coarse_pos = l < M ? copy(cpis[l + 1]) : Int32.(1:nc)  # coarsest child is natural order
        _, invdiag_perm = _diag_inverse(Ap)
        is_mf = 2 <= l <= fused_top + 1                        # apply A_l matrix-free -> don't store it
        Adev = is_mf ? empty_csr(n) :
               AMGMatrixCSR(dev(_rowptr(Ap)), dev(_colval(Ap)), dev(Tl.(_nzval(Ap))), n, n)
        diag_index_perm = is_mf ? dev(Int[]) : dev(_diag_index(Ap))
        sc = scale_correction ? dev(zeros(Tl, n)) : dev(Tl[])  # Ac scratch (matfree levels too)
        push!(levels, MatrixFreeLevel(Adev, dev(Tl.(invdiag_perm)), diag_index_perm, Tl(omegas[l]),
                              dev(aos[l]), dev(inv_sqrt_w), dev(coarse_pos), dev(row_macro),
                              n, nc, dev(zeros(Tl, n)), dev(zeros(Tl, n)), dev(zeros(Tl, n)), dev(zeros(Tl, n)), sc))
    end

    coarse_n = _m(operators[L])
    coarse_csc = _csr_to_csc(operators[L])
    coarse_fac = lu(coarse_csc)
    coarse_inv = _build_coarse_dense_inv(coarse_csc, backend, TC, coarse_max_rows)
    rhs_h, x_h, rhs_hT, x_hT = _coarse_host_buffers(coarse_fac, TC, coarse_n)
    n = _m(Am)
    st = MatrixFreeHierarchy(levels, Ref{Any}(coarse_fac), coarse_n,
                   dev(zeros(TC, coarse_n)), dev(zeros(TC, coarse_n)), Ref{Any}(coarse_inv),
                   rhs_h, x_h, rhs_hT, x_hT,
                   Vector{Int32}(cps[1]), dev(cps[1]), dev(zeros(T, n)),
                   n, pre, post, T(omega_nominal), backend, fused_top, Int(coarse_max_rows), scale_correction,
                   Ref{Any}(nothing), n, length(_nzval(Am)), _is_symmetric(Am), 0.0, 256)
    return (st=st, operators=operators, Ps=Ps, omegas=omegas, invdiags=invdiags_nat,
            coarse_fac=coarse_fac, T=T, A_perms=A_perms, pvms=pvms, aos=aos, cps=cps,
            cpis=cpis, aggs=aggs, M=M)
end

# k weighted-Jacobi sweeps on lv.A with rhs, result guaranteed in lv.x (ping-pong + copy if odd).
function smooth_level!(lv::MatrixFreeLevel, rhs, k::Int, bk, wg)
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

is_fused_level(st::MatrixFreeHierarchy, l::Int) = 2 <= l <= st.fused_top + 1

# out_l <- A_l·x_l EXACTLY, matrix-free Galerkin: prolong x_l up to the fine grid, ONE fine SpMV on
# the materialized A_0, restrict back to level l. A_l is never stored. cy/cz are finest-sized scratch.
function apply_fused_operator!(out_l, st::MatrixFreeHierarchy, l::Int, x_l, cy, cz, bk, wg)
    nl = st.levels[l].n
    cur, oth = cy, cz
    copyto!(view(cur, 1:nl), view(x_l, 1:nl))
    for j in (l - 1):-1:1                                   # prolong chain level l -> fine
        lv = st.levels[j]
        _launch_amg_kernel!(bk, wg, _amg_matrix_free_prolong_set_kernel!, lv.n, oth, cur,
                            lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
        cur, oth = oth, cur
    end
    A1 = st.levels[1].A                                     # the one materialized fine SpMV
    _launch_amg_kernel!(bk, wg, _amg_csr_matvec_kernel!, st.levels[1].n, oth,
                        _rowptr(A1), _colval(A1), _nzval(A1), cur)
    cur, oth = oth, cur
    for j in 1:(l - 1)                                      # restrict chain fine -> level l
        lv = st.levels[j]
        _launch_amg_kernel!(bk, wg, _amg_matrix_free_restrict_kernel!, lv.n_coarse, oth, cur,
                            lv.agg_offsets, lv.inv_sqrt_w, lv.coarse_pos)
        cur, oth = oth, cur
    end
    copyto!(view(out_l, 1:nl), view(cur, 1:nl))
    return out_l
end

# k matrix-free weighted-Jacobi sweeps at level l (x += omega·invdiag·(rhs - A_l·x)), A_l never stored.
function smooth_fused_level!(st::MatrixFreeHierarchy, l::Int, rhs, k::Int, cy, cz, bk, wg)
    lv = st.levels[l]
    for _ in 1:k
        apply_fused_operator!(lv.r, st, l, lv.x, cy, cz, bk, wg)
        _launch_amg_kernel!(bk, wg, _amg_residual_in_place_kernel!, lv.n, lv.r, rhs)
        _launch_amg_kernel!(bk, wg, _amg_jacobi_correction_kernel!, lv.n, lv.x, lv.r, lv.invdiag, lv.omega)
    end
    return lv.x
end

# GAMG scale_correction up-sweep. Replaces plain x += P·xc with the energy-minimising x += sf·c,
# sf=(r_l·c)/(c·Ac). r_l = lv.r, the down-sweep post-pre-smoothing residual (still valid: lv.x is
# untouched between the down-sweep residual and this call), so no residual recompute is paid.
# Matrix-free levels get Ac via the Galerkin chain (apply_fused_operator!, one fine SpMV).
function apply_scaled_coarse_correction!(st::MatrixFreeHierarchy, l::Int, child_x, cy, cz, bk, wg)
    lv = st.levels[l]
    T = eltype(lv.x)
    c = lv.tmp
    _launch_amg_kernel!(bk, wg, _amg_matrix_free_prolong_set_kernel!, lv.n, c, child_x,
                        lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
    if is_fused_level(st, l)
        apply_fused_operator!(lv.sc, st, l, c, cy, cz, bk, wg)                                      # Ac
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
function matrix_free_cycle!(st::MatrixFreeHierarchy, rhs_dev)
    bk = st.backend; wg = 256; M = length(st.levels); T = eltype(st.coarse_rhs)
    cy = st.levels[1].tmp; cz = st.levels[1].r  # finest-sized chain scratch (free once level 1 is done)
    lv1 = st.levels[1]; copyto!(lv1.rhs, rhs_dev)
    # down sweep
    for l in 1:M
        lv = st.levels[l]
        _fill_device!(bk, lv.x, zero(T))
        if is_fused_level(st, l)
            smooth_fused_level!(st, l, lv.rhs, st.pre, cy, cz, bk, wg)
            apply_fused_operator!(lv.r, st, l, lv.x, cy, cz, bk, wg)
            _launch_amg_kernel!(bk, wg, _amg_residual_in_place_kernel!, lv.n, lv.r, lv.rhs)  # lv.r = rhs - A_l x
        else
            smooth_level!(lv, lv.rhs, st.pre, bk, wg)
            rp, cv, nz = _rowptr(lv.A), _colval(lv.A), _nzval(lv.A)
            _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, lv.n, lv.r, rp, cv, nz, lv.x, lv.rhs)
        end
        child_rhs = l < M ? st.levels[l + 1].rhs : st.coarse_rhs
        _launch_amg_kernel!(bk, wg, _amg_matrix_free_restrict_kernel!, lv.n_coarse, child_rhs, lv.r,
                            lv.agg_offsets, lv.inv_sqrt_w, lv.coarse_pos)
    end
    # coarsest solve (zero-alloc: device GEMV or reusable-buffer host LU)
    solve_coarsest_level!(st)
    # up sweep
    for l in M:-1:1
        lv = st.levels[l]
        child_x = l < M ? st.levels[l + 1].x : st.coarse_x
        if st.scale_correction
            apply_scaled_coarse_correction!(st, l, child_x, cy, cz, bk, wg)
        else
            _launch_amg_kernel!(bk, wg, _amg_matrix_free_prolong_add_kernel!, lv.n, lv.x, child_x,
                                lv.row_macro, lv.inv_sqrt_w, lv.coarse_pos)
        end
        # Coarse levels post-smooth against lv.r (the pre-smoothing residual), matching the reference
        # _cycle! rhs aliasing (level.rhs is overwritten by _residual! at levels >= 2). Empirically a
        # stronger cycle: F1 standalone 151 -> 121 iters, == reference; Cg iters unchanged (89 == 89).
        post_rhs = l == 1 ? lv.rhs : lv.r
        if is_fused_level(st, l)
            copyto!(lv.tmp, post_rhs)  # smooth_fused_level! uses lv.r as scratch; avoid aliasing
            smooth_fused_level!(st, l, lv.tmp, st.post, cy, cz, bk, wg)
        else
            smooth_level!(lv, post_rhs, st.post, bk, wg)
        end
    end
    KernelAbstractions.synchronize(bk)
    return st.levels[1].x
end
