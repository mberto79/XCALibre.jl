# Phase 4 experiment: SELL-P (Sliced ELLPACK) fine-format SpMV vs the reference scalar-CSR
# kernel (prompt §3). Standalone + isolated — NOT wired into the AMG module. Adopt ONLY if it
# beats CSR on the real FVM fixtures; else keep CSR and record why (saved "report when it helps"
# rule). Both kernels are one-thread-per-row; the only difference is memory layout:
#   CSR  : row-major nzval -> a warp's 32 threads read 32 scattered row-starts (uncoalesced).
#   SELL-P: slice(=warp=32 rows) column-major, pad to slice-max row -> lane=local row, so for a
#           fixed column j the 32 lanes read 32 CONSECUTIVE addresses (coalesced). Int32 cols.
# Low row-length variance (see findings) makes SELL-P pad overhead tiny; this isolates whether
# coalescing alone wins.

using XCALibre
using SparseArrays, LinearAlgebra, Statistics
using XCALibre.Solve: _rowptr, _colval, _nzval, _amg_matrix, _m
using KernelAbstractions
import Adapt
import Adapt: adapt
using CUDA

# SELL-P (slice height = 32). col padding -> 1 (valid index), val padding -> 0. Int32 cols when
# n fits 32-bit. slice_ptr[s]:slice_ptr[s+1]-1 holds slice s, stored column-major (w_s*h entries).
struct SELLP{Tv,Ti,VTv,VTi}
    n::Int
    slice::Int
    slice_ptr::VTi   # length nslices+1, 1-based offsets into val/col
    col::VTi         # padded column indices (Ti), slice-column-major
    val::VTv         # padded values (Tv), slice-column-major
end
Adapt.@adapt_structure SELLP

function build_sellp(A; slice=32)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); n = _m(A); Tv = eltype(nz)
    Ti = n <= typemax(Int32) ? Int32 : Int64
    nslices = cld(n, slice)
    slice_ptr = Vector{Ti}(undef, nslices + 1); slice_ptr[1] = 1
    @inbounds for s in 1:nslices
        lo = (s-1)*slice + 1; hi = min(s*slice, n)
        wmax = 0
        for i in lo:hi; wmax = max(wmax, rp[i+1] - rp[i]); end
        slice_ptr[s+1] = slice_ptr[s] + wmax * slice
    end
    total = slice_ptr[end] - 1
    col = ones(Ti, total)            # padding -> column 1
    val = zeros(Tv, total)           # padding -> 0
    @inbounds for s in 1:nslices
        lo = (s-1)*slice + 1; hi = min(s*slice, n)
        base = slice_ptr[s]
        for i in lo:hi
            r = i - lo                # 0-based local row (= lane)
            j = 0
            for p in rp[i]:(rp[i+1]-1)
                off = base + j*slice + r
                col[off] = cv[p]; val[off] = nz[p]
                j += 1
            end
        end
    end
    return SELLP{Tv,Ti,typeof(val),typeof(slice_ptr)}(n, slice, slice_ptr, col, val)
end

@kernel function _sellp_matvec_kernel!(y, @Const(slice_ptr), @Const(col), @Const(val), @Const(x), n::Int, slice::Int)
    i = @index(Global)
    @inbounds if i <= n
        s = (i - 1) ÷ slice
        r = (i - 1) % slice
        base = slice_ptr[s + 1]
        stop = slice_ptr[s + 2]
        acc = zero(eltype(y))
        off = base + r
        while off < stop
            acc += val[off] * x[col[off]]
            off += slice
        end
        y[i] = acc
    end
end

@kernel function _csr_matvec_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x))
    i = @index(Global)
    acc = zero(eltype(y))
    @inbounds for p in rowptr[i]:(rowptr[i+1]-1)
        acc += nzval[p] * x[colval[p]]
    end
    @inbounds y[i] = acc
end

_gpu_time(f, reps) = (f(); CUDA.synchronize(); t0=time_ns(); for _ in 1:reps; f(); end; CUDA.synchronize(); (time_ns()-t0)/1e9/reps)

# Compare SELL-P vs scalar-CSR SpMV on `A` (CUDA). Returns timings + correctness + bandwidth.
function sellp_vs_csr(A; slice=32, workgroup=256, reps=200)
    n = _m(A); Tv = eltype(_nzval(A))
    rp = Int32.(collect(_rowptr(A))); cv = Int32.(collect(_colval(A))); nz = collect(_nzval(A))
    S = build_sellp(A; slice=slice)
    x_h = rand(Tv, n)
    # correctness (host)
    y_csr_h = zeros(Tv, n)
    @inbounds for i in 1:n, p in rp[i]:(rp[i+1]-1); y_csr_h[i] += nz[p]*x_h[cv[p]]; end
    err = maximum(abs.(sellp_matvec_host(S, x_h) .- y_csr_h))
    # device
    d_rp=CUDA.CuVector(rp); d_cv=CUDA.CuVector(cv); d_nz=CUDA.CuVector(nz); d_x=CUDA.CuVector(x_h)
    d_sp=CUDA.CuVector(S.slice_ptr); d_sc=CUDA.CuVector(S.col); d_sv=CUDA.CuVector(S.val)
    d_y=CUDA.CuVector(zeros(Tv,n))
    be = CUDABackend()
    csr! = _csr_matvec_kernel!(be, workgroup, n)
    sel! = _sellp_matvec_kernel!(be, workgroup, n)
    # cuSPARSE CSR mul! = the ACTUAL reference GPU AMG SpMV (ext/XCALibre_CUDAExt.jl:207)
    Acu = CUDA.CUSPARSE.CuSparseMatrixCSR(d_rp, d_cv, d_nz, (n, n))
    t_csr = _gpu_time(()->csr!(d_y,d_rp,d_cv,d_nz,d_x), reps)
    t_sel = _gpu_time(()->sel!(d_y,d_sp,d_sc,d_sv,d_x,n,slice), reps)
    t_cusp = _gpu_time(()->mul!(d_y, Acu, d_x), reps)
    nnz = length(nz); sp_store = length(S.val)
    pad = sp_store/nnz - 1
    bw_sel = (sp_store*(sizeof(Tv)+4)+n*sizeof(Tv))/t_sel/1e9
    bw_cusp = (nnz*(sizeof(Tv)+4)+n*sizeof(Tv))/t_cusp/1e9
    return (; n, nnz, sp_store, pad, err, t_naive=t_csr, t_cusp, t_sel,
            speedup_vs_cusparse=t_cusp/t_sel, bw_sel, bw_cusp)
end
function sellp_matvec_host(S::SELLP, x)
    y = zeros(eltype(S.val), S.n)
    @inbounds for i in 1:S.n
        s = (i - 1) ÷ S.slice
        r = (i - 1) % S.slice
        base = S.slice_ptr[s+1]; stop = S.slice_ptr[s+2]
        off = base + r; acc = zero(eltype(y))
        while off < stop
            acc += S.val[off] * x[S.col[off]]
            off += S.slice
        end
        y[i] = acc
    end
    return y
end
