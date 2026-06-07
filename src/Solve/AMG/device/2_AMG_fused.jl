# Phase 5: matrix-free fused multi-level V-cycle (one workgroup per macro-aggregate). The fused
# zone does fine smoothing + restriction + coarse-correct + prolongation WITHOUT materializing
# P/R/Ac — the operators live implicitly in @localmem/registers, reading only the permuted fine
# operator A_perm. This file currently holds the 5a feasibility spike: the core transfer mechanics
# (localmem staging, barrier-separated stages, leading-thread per-aggregate coarse solve, broadcast
# prolongation) validated against a materialized reference doing the identical aggregate-local math.

# NEW SECTION: dynamic index types (user request 2026-06-06; rationale in AMG_findings.md)

# Smallest SIGNED index type addressing n entries. Signed because materialized coarse levels go
# through cuSPARSE (wants Cint); on <=8GB hardware n,nnz << typemax(Int32) so unsigned global
# would buy no usable range, only risk the cuSPARSE path. Use for GLOBAL operator indices.
_amg_index_type(n::Integer) = n <= typemax(Int32) ? Int32 : Int64

# Smallest UNSIGNED type for LOCAL within-aggregate indices (key off MAX aggregate size, not the
# nominal workgroup, so an oversized aggregate can't overflow). cuSPARSE-free here, so unsigned is
# safe and shrinks the local index footprint. Returns a Type at runtime — bake it into the stored
# eltype at the host init boundary; never let it flow unguarded into a kernel inner loop.
function _amg_local_index_type(max_aggregate_size::Integer)
    max_aggregate_size <= typemax(UInt8)  ? UInt8  :
    max_aggregate_size <= typemax(UInt16) ? UInt16 :
    max_aggregate_size <= typemax(UInt32) ? UInt32 : UInt64
end

# Fused workgroup size = next power of two >= max aggregate size (== 2^merge_levels for binary
# face-pairing). Powers of two keep the (future) binary-tree reduction exact.
_amg_fused_workgroup(max_aggregate_size::Integer) = 1 << ceil(Int, log2(max(max_aggregate_size, 1)))

_max_aggregate_size(agg_offsets) = maximum(diff(Vector{Int}(agg_offsets)))

# NEW SECTION: 5a fused aggregate-local 2-grid correction (feasibility spike)

# One workgroup per macro-aggregate. dx <- P (Ac \ (R r)) with R = normalized piecewise-constant
# restriction (1/sqrt(w)), Ac = R·A_perm·P the 1x1 aggregate-local Galerkin operator, computed
# matrix-free from the in-aggregate coefficients (columns inside the aggregate's contiguous range).
# KA body-split rule: locals are dropped across @synchronize — every stage recomputes from @index.
@kernel function _amg_fused_2grid_kernel!(
    dx::AbstractVector{T}, @Const(rowptr), @Const(colval), @Const(nzval),
    @Const(agg_offsets), @Const(r), ::Val{W}
) where {T, W}
    rs = @localmem T (W,)   # residual staging / restriction input
    as = @localmem T (W,)   # per-row in-aggregate coefficient sums (for Ac)
    cs = @localmem T (1,)    # coarse error correction (broadcast scalar)

    # Stage 1: stage residual and each active row's in-aggregate coefficient sum into shared mem.
    g = @index(Group, Linear)
    ℓ = @index(Local, Linear)
    lo = Int(agg_offsets[g])
    hi = Int(agg_offsets[g + 1]) - 1
    w = hi - lo + 1
    if ℓ <= w
        row = lo + ℓ - 1
        rs[ℓ] = r[row]
        s = zero(T)
        for p in rowptr[row]:(rowptr[row + 1] - 1)
            c = colval[p]
            (lo <= c <= hi) && (s += nzval[p])
        end
        as[ℓ] = s
    else
        rs[ℓ] = zero(T)
        as[ℓ] = zero(T)
    end
    @synchronize

    # Stage 2: leading thread solves the 1x1 coarse operator and writes the broadcast correction.
    g = @index(Group, Linear)
    ℓ = @index(Local, Linear)
    lo = Int(agg_offsets[g])
    hi = Int(agg_offsets[g + 1]) - 1
    w = hi - lo + 1
    if ℓ == 1
        sumr = zero(T)
        sumA = zero(T)
        for k in 1:w
            sumr += rs[k]
            sumA += as[k]
        end
        sw = sqrt(T(w))
        r_coarse = sumr / sw                       # R r
        Ac = sumA / T(w)                           # R A P (1x1)
        cs[1] = abs(Ac) > eps(T) ? r_coarse / Ac : zero(T)
    end
    @synchronize

    # Stage 3: prolongate the coarse correction back to the fine cells (register broadcast of cs).
    g = @index(Group, Linear)
    ℓ = @index(Local, Linear)
    lo = Int(agg_offsets[g])
    hi = Int(agg_offsets[g + 1]) - 1
    w = hi - lo + 1
    if ℓ <= w
        dx[lo + ℓ - 1] = cs[1] / sqrt(T(w))
    end
end

# Launch helper: exactly one workgroup of W threads per macro-aggregate.
function fused_2grid_correction!(dx, A_perm, agg_offsets, r, backend, ::Val{W}) where {W}
    n_macro = length(agg_offsets) - 1
    kernel! = _amg_fused_2grid_kernel!(backend, W, n_macro * W)
    kernel!(dx, _rowptr(A_perm), _colval(A_perm), _nzval(A_perm), agg_offsets, r, Val(W))
    KernelAbstractions.synchronize(backend)
    return dx
end

# Materialized ground truth: identical aggregate-local 2-grid math, plain host loops (deterministic
# reference for the spike — validates the kernel mechanics, NOT multigrid convergence).
function _fused_2grid_reference(A_perm, agg_offsets, r)
    rp = _rowptr(A_perm); cv = _colval(A_perm); nz = _nzval(A_perm)
    T = eltype(nz)
    dx = zeros(T, _m(A_perm))
    @inbounds for g in 1:(length(agg_offsets) - 1)
        lo = Int(agg_offsets[g]); hi = Int(agg_offsets[g + 1]) - 1; w = hi - lo + 1
        sumr = zero(T); sumA = zero(T)
        for row in lo:hi
            sumr += r[row]
            for p in rp[row]:(rp[row + 1] - 1)
                c = cv[p]
                (lo <= c <= hi) && (sumA += nz[p])
            end
        end
        sw = sqrt(T(w))
        Ac = sumA / T(w)
        ec = abs(Ac) > eps(T) ? (sumr / sw) / Ac : zero(T)
        pf = ec / sw
        for row in lo:hi
            dx[row] = pf
        end
    end
    return dx
end

# INDEPENDENT validation: compare the fused kernel against an oracle built from the REFERENCE's own
# machinery (not a hand-rolled twin) — reference normalized P0 (build_prolongation), sparse Galerkin
# RAP, and a diagonal (Jacobi) coarse solve == the kernel's per-aggregate 1x1 block solve. This
# discriminates a real algorithmic error in the in-aggregate coefficient set / transfer composition,
# which the twin-vs-twin spike cannot. Returns (relerr, agg_match): agg_match guards that the layout
# aggregation equals the reference aggregation (else the comparison is meaningless).
function fused_2grid_oracle_error(A, merge_levels::Integer, backend)
    Am = _amg_matrix(A)
    agg, P0, _ = build_prolongation(Am, Geometric(merge_levels=Int(merge_levels)))
    R = sparse(P0')
    Ac = R * _csr_to_csc(Am) * P0
    lay = build_macro_layout(A, merge_levels)
    agg_match = agg == Vector{Int}(lay.fine_to_macro)
    n = _m(Am); T = eltype(_nzval(Am))
    r_perm = rand(T, n)
    W = _amg_fused_workgroup(_max_aggregate_size(lay.agg_offsets))
    dev(x) = backend isa CPU ? x : Adapt.adapt(backend, x)
    A_dev = AMGMatrixCSR(dev(_rowptr(lay.A_perm)), dev(_colval(lay.A_perm)), dev(_nzval(lay.A_perm)),
                         lay.A_perm.m, lay.A_perm.n)
    dxd = dev(zeros(T, n))
    fused_2grid_correction!(dxd, A_dev, dev(lay.agg_offsets), dev(r_perm), backend, Val(W))
    dx_k = Array(dxd)
    dx_korig = similar(dx_k); r_orig = similar(r_perm)
    @inbounds for k in 1:n
        dx_korig[Int(lay.cell_perm[k])] = dx_k[k]
        r_orig[Int(lay.cell_perm[k])] = r_perm[k]
    end
    dx_o = P0 * ((R * r_orig) ./ diag(Ac))
    relerr = maximum(abs.(dx_korig .- dx_o)) / max(maximum(abs.(dx_o)), eps(T))
    return (relerr=relerr, agg_match=agg_match, W=W)
end

# Spike driver: build the macro layout, run the fused kernel and a HAND-ROLLED twin of the same
# aggregate-local math on the same permuted residual. relerr=0 here proves CPU/GPU determinism + no
# transcription typo only — NOT algorithmic correctness. Use fused_2grid_oracle_error for the latter.
function fused_2grid_spike(A, merge_levels::Integer, backend)
    lay = build_macro_layout(A, merge_levels)
    W = _amg_fused_workgroup(_max_aggregate_size(lay.agg_offsets))
    A_perm = lay.A_perm
    T = eltype(_nzval(A_perm))
    r = rand(T, _m(A_perm))
    dx_ref = _fused_2grid_reference(A_perm, lay.agg_offsets, r)

    dev(x) = backend isa CPU ? x : Adapt.adapt(backend, x)
    A_dev = AMGMatrixCSR(dev(_rowptr(A_perm)), dev(_colval(A_perm)), dev(_nzval(A_perm)),
                         A_perm.m, A_perm.n)
    ao = dev(lay.agg_offsets); rd = dev(r); dxd = dev(zeros(T, _m(A_perm)))
    fused_2grid_correction!(dxd, A_dev, ao, rd, backend, Val(W))
    dx = Array(dxd)
    relerr = maximum(abs.(dx .- dx_ref)) / max(maximum(abs.(dx_ref)), eps(T))
    return (W=W, n_macro=lay.n_macro, relerr=relerr)
end
