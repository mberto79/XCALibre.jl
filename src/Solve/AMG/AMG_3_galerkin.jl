# ─── Galerkin projection: build P, R = Pᵀ, Ac = R·A·P ────────────────────────
# All construction is performed on CPU (setup is infrequent).
# Numerical updates (update!) reuse the pre-allocated T = A·P and Ac scratch on CPU,
# then upload updated nzval to the device.

# ─── Tentative prolongation P̂ (piecewise-constant aggregates) ─────────────────

function build_tentative_P(n_fine::Int, nagg::Int, agg::Vector{Int})
    rows = collect(1:n_fine)
    cols = agg
    vals = ones(Float64, n_fine)
    return SparseMatricesCSR.sparsecsr(rows, cols, vals, n_fine, nagg)
end

# ─── Smoothed prolongation P = (I - ω D⁻¹ A) P̂ ──────────────────────────────

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

    # Upper bound on nnz(P): ≤ nnz(A) + n. Pre-size to avoid push! resizing.
    nnz_est = length(nzval_A) + n
    row_out = Int[]; sizehint!(row_out, nnz_est)
    col_out = Int[]; sizehint!(col_out, nnz_est)
    val_out = Tv[];  sizehint!(val_out, nnz_est)

    tmp  = zeros(Tv, nc)
    used = Int[];   sizehint!(used, 16)

    @inbounds for i in 1:n
        # Step 1: accumulate (A * P̂)[i, k] for all columns k touched by row i of A*P̂
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            j = colval_A[nzi_A]
            for nzi_P in rowptr_P[j]:(rowptr_P[j+1]-1)
                k = colval_P[nzi_P]
                if tmp[k] == zero(Tv); push!(used, k); end
                tmp[k] += nzval_A[nzi_A] * nzval_P[nzi_P]
            end
        end

        # Step 2: apply -(ω/a_ii) to ALL accumulated entries (not just the agg[i] column)
        d     = diag_A[i]
        coeff = abs(d) > eps(Tv) ? ω / d : zero(Tv)
        for k in used
            tmp[k] = -coeff * tmp[k]
        end

        # Step 3: add P̂[i, *] back — gives P_smooth = P̂ - ω D⁻¹ A P̂
        for nzi_P in rowptr_P[i]:(rowptr_P[i+1]-1)
            k = colval_P[nzi_P]
            if tmp[k] == zero(Tv); push!(used, k); end
            tmp[k] += nzval_P[nzi_P]
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
# Returns AP for reuse as pre-allocated scratch in _spgemm_nzval!.

function galerkin_product(R::SparseMatrixCSR{Bi,Tv,Ti},
                           A::SparseMatrixCSR{Bi,Tv,Ti},
                           P::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    AP = _spgemm(A, P)
    Ac = _spgemm(R, AP)
    return AP, Ac
end

# ── General SpGEMM for CSR matrices on CPU (Gustavson two-pass) ───────────────
# Pass 1 (symbolic): count nnz. Pass 2 (numeric): dense accumulator, flag = first-touch detector.

function _spgemm(A::SparseMatrixCSR{Bi,Tv,Ti},
                 B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    m = size(A, 1); n = size(B, 2)
    rowptr_A = A.rowptr; colval_A = A.colval; nzval_A = A.nzval
    rowptr_B = B.rowptr; colval_B = B.colval; nzval_B = B.nzval

    # ── Pass 1: symbolic ─────────────────────────────────────────────────────
    # flag[j] == i avoids duplicate count.
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
        # Do NOT filter by eps: sparsity pattern must match in setup! and update!.
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
# Compact accumulator (tmps: max_nnz_per_row × nthreads) stays in L1; zero on entry and exit.

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
            # Assign compact local slots for all columns of C in row i.
            c_start = rowptr_C[i]
            c_end   = rowptr_C[i+1] - 1
            for nzi_C in c_start:c_end
                ctol[colval_C[nzi_C]] = Int32(nzi_C - c_start + 1)
            end
            # Accumulate A[i,:] × B into compact tmp.
            for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
                k   = colval_A[nzi_A]
                aik = nzval_A[nzi_A]
                for nzi_B in rowptr_B[k]:(rowptr_B[k+1]-1)
                    tmp[ctol[colval_B[nzi_B]]] += aik * nzval_B[nzi_B]
                end
            end
            # Write C and clear both tmp and ctol in one pass.
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

# ─── Per-row threshold truncation of the prolongation operator ───────────────
# Drops entries where |P[i,j]| < τ × max_k|P[i,k]|; τ=0 is no-op.

function _truncate_P(P::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti},
                     τ::Float64) where {Bi,Tv,Ti}
    τ ≤ 0.0 && return P
    n, nc = size(P)
    rowptr = P.rowptr; colval = P.colval; nzval = P.nzval

    nnz_est = length(nzval)
    row_out = Int[]; sizehint!(row_out, nnz_est)
    col_out = Int[]; sizehint!(col_out, nnz_est)
    val_out = Tv[];  sizehint!(val_out, nnz_est)

    @inbounds for i in 1:n
        rs = rowptr[i]; re = rowptr[i+1] - 1
        rs > re && continue
        # Max absolute value in row i
        row_max = zero(Tv)
        for nzi in rs:re
            v = abs(nzval[nzi])
            v > row_max && (row_max = v)
        end
        thresh = Tv(τ) * row_max
        for nzi in rs:re
            abs(nzval[nzi]) >= thresh && (push!(row_out, i); push!(col_out, colval[nzi]); push!(val_out, nzval[nzi]))
        end
    end

    return SparseMatricesCSR.sparsecsr(row_out, col_out, val_out, n, nc)
end

# ─── Gershgorin upper bound for ρ(D⁻¹ A) ────────────────────────────────────
# Tight for FVM M-matrices; ω_P = 2/3 is the classical optimal Chebyshev damping.

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

# ─── Power iteration to estimate ρ(D⁻¹ A) ──────────────────────────────────
# Rayleigh-quotient iteration; deterministic normalised all-ones start.

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
