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

    # Row, col, val for smoothed P (dense pass)
    row_out = Int[]
    col_out = Int[]
    val_out = Tv[]

    tmp = zeros(Tv, nc)   # dense accumulator, reused across rows
    for i in 1:n
        # Collect contributions: P_i = P̂_i - ω * (1/a_ii) * (A_i · P̂)
        # A_i · P̂ means for each column k: sum_j A[i,j] * P̂[j,k]
        # Since P̂ has one nonzero per row: P̂[j,*] = e_{agg[j]}
        # → (A*P̂)[i,k] = sum over j where agg[j]==k of A[i,j]

        fill!(tmp, zero(Tv))
        for nzi_A in rowptr_A[i]:(rowptr_A[i+1]-1)
            j = colval_A[nzi_A]
            # P̂[j, *]: find which column j maps to
            for nzi_P in rowptr_P[j]:(rowptr_P[j+1]-1)
                k = colval_P[nzi_P]
                tmp[k] += nzval_A[nzi_A] * nzval_P[nzi_P]
            end
        end

        # P̂[i, *] — guard against zero diagonal to prevent Inf propagation
        d = diag_A[i]
        coeff = abs(d) > eps(Tv) ? ω / d : zero(Tv)
        for nzi_P in rowptr_P[i]:(rowptr_P[i+1]-1)
            k = colval_P[nzi_P]
            tmp[k] = nzval_P[nzi_P] - coeff * tmp[k]
        end

        # Store nonzeros
        for k in 1:nc
            if abs(tmp[k]) > eps(Tv)
                push!(row_out, i)
                push!(col_out, k)
                push!(val_out, tmp[k])
            end
        end
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
    # Step 1: AP = A * P
    AP = _spgemm(A, P)
    # Step 2: Ac = R * AP
    Ac = _spgemm(R, AP)
    return Ac
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

# ─── Numerical-only Galerkin update (reuse sparsity) ─────────────────────────
# Used in update!() to refresh nzval arrays without recomputing sparsity.

"""
    update_galerkin_nzval!(Ac, R, A, P)

Recompute the numerical values of Ac = R*A*P using the existing sparsity pattern
of Ac. This is cheaper than a full galerkin_product when the pattern is unchanged.
"""
function update_galerkin_nzval!(Ac::SparseMatrixCSR{Bi,Tv,Ti},
                                 R::SparseMatrixCSR,
                                 A::SparseMatrixCSR,
                                 P::SparseMatrixCSR) where {Bi,Tv,Ti}
    # Rebuild from scratch (numerical only, but we recompute the full product)
    # This is correct since the pattern is guaranteed to be identical;
    # we just overwrite the existing nzval buffer.
    Ac_new = galerkin_product(R, A, P)
    # Copy nzval into existing buffer
    length(Ac_new.nzval) == length(Ac.nzval) ||
        error("AMG: Galerkin sparsity pattern changed during update! " *
              "Call full setup instead.")
    copyto!(Ac.nzval, Ac_new.nzval)
    nothing
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
