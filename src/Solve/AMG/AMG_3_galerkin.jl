# ─── Galerkin projection: build P, R = Pᵀ, Ac = R·A·P ────────────────────────
# All construction is performed on CPU (setup is infrequent).
# Numerical updates (update!) reuse the pre-allocated T = A·P and Ac scratch on CPU,
# then upload updated nzval to the device.

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
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval

    # Diagonal of A
    diag_A = zeros(Tv, n)
    @inbounds for i in 1:n
        for nzi in rowptr_A[i]:(rowptr_A[i+1]-1)
            if colval_A[nzi] == i; diag_A[i] = nzval_A[nzi]; break; end
        end
    end

    # P = P̂ - ω * D⁻¹ * (A * P̂)
    rowptr_P = P_tent.rowptr; colval_P = P_tent.colval; nzval_P = P_tent.nzval

    # Upper bound on nnz(P): each row of A has degree d, each A[i,j] maps to one
    # aggregate column → nnz(P) ≤ nnz(A) + n.  Pre-sizing avoids all push! resizing.
    nnz_est = length(nzval_A) + n
    row_out = Int[]; sizehint!(row_out, nnz_est)
    col_out = Int[]; sizehint!(col_out, nnz_est)
    val_out = Tv[];  sizehint!(val_out, nnz_est)

    tmp  = zeros(Tv, nc)
    used = Int[];   sizehint!(used, 16)

    @inbounds for i in 1:n
        # Collect contributions: P_i = P̂_i - ω * (1/a_ii) * (A_i · P̂)
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            j = colval_A[nzi_A]
            for nzi_P in rowptr_P[j]:(rowptr_P[j+1]-1)
                k = colval_P[nzi_P]
                if tmp[k] == zero(Tv); push!(used, k); end
                tmp[k] += nzval_A[nzi_A] * nzval_P[nzi_P]
            end
        end

        # P̂[i, *] — guard against zero diagonal to prevent Inf propagation
        d     = diag_A[i]
        coeff = abs(d) > eps(Tv) ? ω / d : zero(Tv)
        for nzi_P in rowptr_P[i]:(rowptr_P[i+1]-1)
            k = colval_P[nzi_P]
            if tmp[k] == zero(Tv); push!(used, k); end
            tmp[k] = nzval_P[nzi_P] - coeff * tmp[k]
        end

        sort!(used)
        for k in used
            v = tmp[k]; tmp[k] = zero(Tv)
            if abs(v) > eps(Tv)
                push!(row_out, i); push!(col_out, k); push!(val_out, v)
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
Pre-allocates COO arrays to exact size (nnz(R) = nnz(P)) — no dynamic resizing.
"""
function build_restriction(P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    n, nc    = size(P)
    nnz_P    = length(P.nzval)
    row_out  = Vector{Int}(undef, nnz_P)
    col_out  = Vector{Int}(undef, nnz_P)
    val_out  = Vector{Tv}(undef, nnz_P)
    pos = 1
    @inbounds for i in 1:n
        for nzi in P.rowptr[i]:(P.rowptr[i+1]-1)
            row_out[pos] = P.colval[nzi]
            col_out[pos] = i
            val_out[pos] = P.nzval[nzi]
            pos += 1
        end
    end
    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, nc, n)
end

# ─── Galerkin coarse matrix Ac = R · A · P ────────────────────────────────────

"""
    galerkin_product(R, A, P) → (AP, Ac)

Computes Ac = R * A * P via two SpGEMM steps (R*(A*P)).
Returns both the intermediate `AP` and the coarse matrix `Ac` so the caller
can reuse `AP` as the pre-allocated scratch buffer for `_spgemm_nzval!` in
later numerical updates, avoiding a redundant SpGEMM at setup time.
All matrices are SparseMatrixCSR on CPU.
"""
function galerkin_product(R::SparseMatrixCSR{Bi,Tv,Ti},
                           A::SparseMatrixCSR{Bi,Tv,Ti},
                           P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    AP = _spgemm(A, P)
    Ac = _spgemm(R, AP)
    return AP, Ac
end

# ── General SpGEMM for CSR matrices on CPU (Gustavson two-pass) ───────────────
# Pass 1 (symbolic): count nnz per row → exact COO pre-allocation.
# Pass 2 (numeric): dense accumulator; flag array reused as first-touch detector.

function _spgemm(A::SparseMatrixCSR{Bi,Tv,Ti},
                 B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    m = size(A, 1); n = size(B, 2)
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval
    rowptr_B = B.rowptr; colval_B = B.colval; nzval_B = B.nzval

    # ── Pass 1: symbolic ─────────────────────────────────────────────────────
    # flag[j] == i means column j already counted for row i (avoids duplicate count).
    flag      = zeros(Int, n)
    total_nnz = 0
    @inbounds for i in 1:m
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            k = colval_A[nzi_A]
            for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                j = colval_B[nzi_B]
                if flag[j] != i
                    flag[j] = i
                    total_nnz += 1
                end
            end
        end
    end

    # ── Pre-allocate COO output (exact size, no resizing) ────────────────────
    row_out = Vector{Int}(undef, total_nnz)
    col_out = Vector{Int}(undef, total_nnz)
    val_out = Vector{Tv}(undef, total_nnz)

    # ── Pass 2: numeric ──────────────────────────────────────────────────────
    fill!(flag, 0)
    tmp  = zeros(Tv, n)
    used = Int[]; sizehint!(used, 64)   # per-row touched-column list
    pos  = 1
    @inbounds for i in 1:m
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            k = colval_A[nzi_A]; aik = nzval_A[nzi_A]
            for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                j = colval_B[nzi_B]
                if flag[j] != i
                    flag[j] = i
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
            row_out[pos] = i; col_out[pos] = j; val_out[pos] = tmp[j]
            tmp[j] = zero(Tv); pos += 1
        end
        empty!(used)
    end

    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, m, n)
end

# ─── Zero-allocation numerical Galerkin update ────────────────────────────────

"""
    _spgemm_nzval!(C, A, B, tmps, col_to_local)

Compute C = A * B in-place (numerical values only), reusing C's pre-allocated
sparsity pattern. Uses a compact accumulator strategy to keep the hot buffer in L1:

- `tmps`         : `(max_nnz_per_row(C) × nthreads)` Float accumulator — fits in L1 cache.
- `col_to_local` : `(ncols_B × nthreads)` Int32 scatter map; for each row i, column j of B
                   is mapped to a compact local slot 1..nnz_i via this array.

Both arrays must be zero on entry and are left zero on return. No heap allocation.
Works correctly with `Threads.nthreads() == 1`.
"""
function _spgemm_nzval!(C::SparseMatrixCSR{Bi,Tv,Ti},
                         A::SparseMatrixCSR,
                         B::SparseMatrixCSR,
                         tmps::AbstractMatrix{Tv},
                         col_to_local::AbstractMatrix{Int32}) where {Bi,Tv,Ti}
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval
    rowptr_B = B.rowptr; colval_B = B.colval; nzval_B = B.nzval
    rowptr_C = C.rowptr; colval_C = C.colval; nzval_C = C.nzval
    m = size(A, 1)
    # Each thread uses independent columns of tmps and col_to_local — no locking.
    Threads.@threads for i in 1:m
        tid  = Threads.threadid()
        tmp  = view(tmps,         :, tid)
        ctol = view(col_to_local, :, tid)
        @inbounds begin
            # Assign compact local slots for all columns of C in row i
            c_start = rowptr_C[i]
            c_end   = rowptr_C[i+1] - 1
            for nzi_C in c_start:c_end
                ctol[colval_C[nzi_C]] = Int32(nzi_C - c_start + 1)
            end
            # Accumulate A[i,:] × B into compact tmp (indexed via ctol)
            for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
                k   = colval_A[nzi_A]
                aik = nzval_A[nzi_A]
                for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                    tmp[ctol[colval_B[nzi_B]]] += aik * nzval_B[nzi_B]
                end
            end
            # Write into C and clear both tmp and ctol in one pass
            for nzi_C in c_start:c_end
                j = colval_C[nzi_C]
                local_slot = nzi_C - c_start + 1
                nzval_C[nzi_C] = tmp[local_slot]
                tmp[local_slot] = zero(Tv)
                ctol[j] = Int32(0)
            end
        end
    end
    return C
end

# ─── Gershgorin upper bound for ρ(D⁻¹ A) ────────────────────────────────────
#
# For each row i: the Gershgorin disk has centre 1 and radius Σ_{j≠i} |a_ij/a_ii|.
# Hence ρ(D⁻¹A) ≤ max_i Σ_j |a_ij/a_ii|  (the full row sum including diagonal).
#
# For FVM M-matrices (pressure Laplacian, diffusion): off-diagonal row sum equals
# the diagonal in magnitude, so the bound is exactly 2 — equal to the true ρ.
# This makes ω_P = 4/(3·ρ) = 2/3, the classical optimal damping for these operators.
#
# Using this instead of power iteration is:
#   • O(nnz) single pass — faster than niters SpMVs
#   • deterministic — no random starting vector → hierarchy is reproducible
#   • tight for M-matrices — not overly conservative

"""
    _gershgorin_rho(A, Dinv) → ρ_upper

Compute the Gershgorin circle upper bound on the spectral radius of D⁻¹A:
    ρ ≤ max_i  Σ_j |a_ij| * |Dinv[i]|
For FVM M-matrices this equals the true spectral radius (= 2 for a uniform Laplacian).
"""
function _gershgorin_rho(A::SparseMatrixCSR{Bi,Tv,Ti},
                          Dinv::AbstractVector{Tv}) where {Bi,Tv,Ti}
    n      = size(A, 1)
    rowptr = A.rowptr; nzval = A.nzval
    rho    = zero(Tv)
    @inbounds for i in 1:n
        row_sum = zero(Tv)
        di = abs(Dinv[i])
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            row_sum += abs(nzval[nzi]) * di
        end
        rho = max(rho, row_sum)
    end
    return rho
end

# ─── Power iteration to estimate spectral radius ρ(D⁻¹ A) ──────────────────
#
# Used for Chebyshev smoother eigenvalue bounds. Starts from a deterministic
# all-ones vector (normalised), which is a good initial guess for smooth dominant
# eigenvectors typical of Laplacian-like operators and avoids run-to-run variation.

"""
    estimate_spectral_radius(A, Dinv; niters=20) → ρ

Rayleigh-quotient power iteration on CPU for the matrix D⁻¹A.
Used to set Chebyshev smoother eigenvalue bounds.
Starting vector is deterministic (normalised all-ones) so the hierarchy is
reproducible across runs.
"""
function estimate_spectral_radius(A::SparseMatrixCSR{Bi,Tv,Ti},
                                   Dinv::AbstractVector;
                                   niters::Int=20) where {Bi,Tv,Ti}
    n = size(A, 1)
    v = fill(one(Tv) / sqrt(Tv(n)), n)   # deterministic: normalised all-ones
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
