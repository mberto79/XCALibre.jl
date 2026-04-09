# ─── Galerkin projection: build P, R = Pᵀ, Ac = R·A·P ────────────────────────
# All construction is performed on CPU (setup is infrequent).
# The numerical Galerkin pass (update!) is also on CPU for simplicity;
# the resulting nzval arrays are transferred to the device on-demand.

# ─── Tentative prolongation P̂ (piecewise-constant aggregates) ─────────────────

"""
    build_tentative_P(n_fine, nagg, agg) → SparseMatrixCSR

P̂[i, agg[i]] = 1.0  (one column per aggregate).
"""
function build_tentative_P(n_fine::Int, nagg::Int, agg::Vector{Int})
    rows = collect(1:n_fine)
    cols = agg
    vals = ones(Float64, n_fine)
    return SparseMatricesCSR.sparsecsr(rows, cols, vals, n_fine, nagg)
end

# ─── Smoothed prolongation P = (I - ω D⁻¹ A) P̂ ──────────────────────────────

"""
    smooth_prolongation(A, P_tent, ω) → SparseMatrixCSR

Applies one step of Jacobi smoothing to the tentative prolongation:
    P = P̂ - ω * D⁻¹ * A * P̂
The product A*P̂ is performed column by column.
"""
function smooth_prolongation(A::SparseMatrixCSR{Bi,Tv,Ti},
                              P_tent::SparseMatrixCSR,
                              ω::Float64) where {Bi,Tv,Ti}
    n, nc = size(P_tent)
    rowptr_A = A.rowptr
    colval_A = A.colval
    nzval_A  = A.nzval

    # Diagonal of A
    diag_A = zeros(Tv, n)
    for i in 1:n
        for nzi in rowptr_A[i]:(rowptr_A[i+1]-1)
            if colval_A[nzi] == i
                diag_A[i] = nzval_A[nzi]
                break
            end
        end
    end

    # P = P̂ - ω * D⁻¹ * (A * P̂)
    # Work directly with CSR pattern of P̂
    rowptr_P = P_tent.rowptr
    colval_P = P_tent.colval
    nzval_P  = P_tent.nzval

    # Row, col, val for smoothed P (sparse accumulator — avoids O(n*nc) scan)
    row_out = Int[]
    col_out = Int[]
    val_out = Tv[]

    tmp  = zeros(Tv, nc)   # dense accumulator, reused across rows
    used = Int[]            # tracks which columns were touched this row
    for i in 1:n
        # Collect contributions: P_i = P̂_i - ω * (1/a_ii) * (A_i · P̂)
        # A_i · P̂ means for each column k: sum_j A[i,j] * P̂[j,k]
        # Since P̂ has one nonzero per row: P̂[j,*] = e_{agg[j]}
        # → (A*P̂)[i,k] = sum over j where agg[j]==k of A[i,j]

        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            j = colval_A[nzi_A]
            # P̂[j, *]: find which column j maps to
            for nzi_P in rowptr_P[j]:(rowptr_P[j+1]-1)
                k = colval_P[nzi_P]
                if tmp[k] == zero(Tv)
                    push!(used, k)
                end
                tmp[k] += nzval_A[nzi_A] * nzval_P[nzi_P]
            end
        end

        # P̂[i, *] — guard against zero diagonal to prevent Inf propagation
        d = diag_A[i]
        coeff = abs(d) > eps(Tv) ? ω / d : zero(Tv)
        for nzi_P in rowptr_P[i]:(rowptr_P[i+1]-1)
            k = colval_P[nzi_P]
            if tmp[k] == zero(Tv)
                push!(used, k)
            end
            tmp[k] = nzval_P[nzi_P] - coeff * tmp[k]
        end

        # Store nonzeros — only scan touched columns
        sort!(used)
        for k in used
            v = tmp[k]
            tmp[k] = zero(Tv)
            if abs(v) > eps(Tv)
                push!(row_out, i)
                push!(col_out, k)
                push!(val_out, v)
            end
        end
        empty!(used)
    end

    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, n, nc)
end

# ─── Restriction R = Pᵀ ───────────────────────────────────────────────────────

"""
    build_restriction(P) → SparseMatrixCSR

Transpose P (n × nc) to get R (nc × n).
"""
function build_restriction(P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    n, nc = size(P)
    rowptr_P = P.rowptr
    colval_P = P.colval
    nzval_P  = P.nzval

    # Collect transposed entries
    row_out = Int[]
    col_out = Int[]
    val_out = Tv[]
    for i in 1:n
        for nzi in rowptr_P[i]:(rowptr_P[i+1]-1)
            k = colval_P[nzi]
            push!(row_out, k)
            push!(col_out, i)
            push!(val_out, nzval_P[nzi])
        end
    end

    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, nc, n)
end

# ─── Galerkin coarse matrix Ac = R · A · P ────────────────────────────────────

"""
    galerkin_product(R, A, P) → SparseMatrixCSR

Computes Ac = R * A * P via two SpGEMM steps (R*(A*P)).
All matrices are SparseMatrixCSR on CPU.
"""
function galerkin_product(R::SparseMatrixCSR{Bi,Tv,Ti},
                           A::SparseMatrixCSR{Bi,Tv,Ti},
                           P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    AP = _spgemm(A, P)
    Ac = _spgemm(R, AP)
    return Ac
end

"""
    galerkin_product_with_AP(R, A, P) → (Ac, AP)

Same as `galerkin_product` but also returns the intermediate A*P matrix.
Used during `amg_setup!` so AP can be pre-allocated for zero-allocation updates.
"""
function galerkin_product_with_AP(R::SparseMatrixCSR{Bi,Tv,Ti},
                                   A::SparseMatrixCSR{Bi,Tv,Ti},
                                   P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    AP = _spgemm(A, P)
    Ac = _spgemm(R, AP)
    return Ac, AP
end

# ── General SpGEMM for CSR matrices on CPU ─────────────────────────────────────

function _spgemm(A::SparseMatrixCSR{Bi,Tv,Ti},
                 B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    m = size(A, 1)
    n = size(B, 2)
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval
    rowptr_B = B.rowptr; colval_B = B.colval; nzval_B = B.nzval

    row_out = Int[]
    col_out = Int[]
    val_out = Tv[]

    tmp = zeros(Tv, n)   # dense accumulator (nc is small at coarse levels)
    used = Int[]

    for i in 1:m
        # Accumulate row i of C = A * B
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            k = colval_A[nzi_A]
            aik = nzval_A[nzi_A]
            for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                j = colval_B[nzi_B]
                if tmp[j] == zero(Tv)
                    push!(used, j)
                end
                tmp[j] += aik * nzval_B[nzi_B]
            end
        end
        # Flush accumulator — keep ALL structurally nonzero entries.
        # Do NOT filter by eps(Tv): the same P/R are reused in update!, but A changes
        # numerically each outer iteration. Filtering by magnitude would produce a
        # different sparsity pattern in update! vs setup, breaking _copy_nzval_to_device!.
        sort!(used)
        for j in used
            push!(row_out, i)
            push!(col_out, j)
            push!(val_out, tmp[j])
            tmp[j] = zero(Tv)
        end
        empty!(used)
    end

    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, m, n)
end

# ─── Zero-allocation numerical Galerkin update ────────────────────────────────
#
# `_spgemm_nzval!` is the allocation-free hot path for `update!()`.
# It requires:
#   - C pre-allocated with the correct sparsity pattern (computed at setup time)
#   - tmp: a zero-initialised dense scratch vector of size >= size(B, 2)
#
# The pattern of A*B is determined solely by the patterns of A and B, both of
# which are fixed (same mesh → same sparsity). Only the numerical values of A
# change between outer PISO/SIMPLE iterations.

"""
    _spgemm_nzval!(C, A, B, tmps)

Compute C = A * B in-place (numerical values only), reusing C's pre-allocated
sparsity pattern. `tmps` is an `(ncols_B × nthreads)` matrix of dense scratch space,
pre-allocated at setup time (column-major: each thread's slice is a contiguous column).
Rows are processed in parallel with `Threads.@threads`; each thread uses column
`tmps[:, threadid()]` so no locking is needed.

`tmps` must be zero on entry and is left zero on return. No heap allocation.
Works correctly with `Threads.nthreads() == 1` (single-threaded path).
"""
function _spgemm_nzval!(C::SparseMatrixCSR{Bi,Tv,Ti},
                         A::SparseMatrixCSR,
                         B::SparseMatrixCSR,
                         tmps::AbstractMatrix{Tv}) where {Bi,Tv,Ti}
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval
    rowptr_B = B.rowptr; colval_B = B.colval; nzval_B = B.nzval
    rowptr_C = C.rowptr; colval_C = C.colval; nzval_C = C.nzval
    m = size(A, 1)
    # Each thread gets an independent column of tmps (contiguous, cache-friendly);
    # writes go to disjoint ranges of nzval_C (row i owns rowptr_C[i]:rowptr_C[i+1]-1).
    Threads.@threads for i in 1:m
        tmp = view(tmps, :, Threads.threadid())
        @inbounds begin
            # Accumulate row i of A*B into thread-local scratch
            for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
                k   = colval_A[nzi_A]
                aik = nzval_A[nzi_A]
                for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                    tmp[colval_B[nzi_B]] += aik * nzval_B[nzi_B]
                end
            end
            # Write into C using its pre-computed pattern; clear scratch in same pass
            for nzi_C in rowptr_C[i]:(rowptr_C[i+1]-1)
                j = colval_C[nzi_C]
                nzval_C[nzi_C] = tmp[j]
                tmp[j] = zero(Tv)
            end
        end
    end
    return C
end

# ─── Power iteration to estimate spectral radius ρ(D⁻¹ A) ──────────────────

"""
    estimate_spectral_radius(A, Dinv; niters=10) → ρ

Rayleigh-quotient power iteration on CPU for the matrix D⁻¹A.
Used to set Chebyshev and smoothed-prolongation damping parameters.
"""
function estimate_spectral_radius(A::SparseMatrixCSR{Bi,Tv,Ti},
                                   Dinv::AbstractVector;
                                   niters::Int=10) where {Bi,Tv,Ti}
    n = size(A, 1)
    v = rand(Tv, n)
    v ./= norm(v)
    w = similar(v)
    rho = one(Tv)

    rowptr = A.rowptr; colval = A.colval; nzval = A.nzval
    for _ in 1:niters
        # w = D⁻¹ * A * v
        for i in 1:n
            acc = zero(Tv)
            for nzi in rowptr[i]:(rowptr[i+1]-1)
                acc += nzval[nzi] * v[colval[nzi]]
            end
            w[i] = Dinv[i] * acc
        end
        rho_new = dot(w, v) / dot(v, v)
        nrm = norm(w)
        nrm > zero(Tv) && (v .= w ./ nrm)
        rho = rho_new
    end
    return abs(rho)
end
