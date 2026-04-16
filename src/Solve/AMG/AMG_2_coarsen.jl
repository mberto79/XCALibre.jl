# ─── Setup phase: coarsening strategies ───────────────────────────────────────
# Produces aggregate IDs (integer array mapping each fine node to an aggregate).
# Implemented on CPU only (setup is called once; GPU arrays are gathered here).

"""
    amg_coarsen(A_cpu, strength, strategy) → agg_ids

Map each fine row to an aggregate (1-based). Strategies: `:SA` (Smoothed Aggregation) or `:RS` (Ruge–Stüben).
"""
function amg_coarsen(A::SparseMatrixCSR, strength::Float64, strategy::Symbol)
    if strategy === :SA
        return _coarsen_SA(A, strength)
    elseif strategy === :RS
        return _coarsen_RS(A, strength)
    else
        error("Unknown AMG coarsening strategy: $strategy. Use :SA or :RS.")
    end
end

# ─── Shared helper: build flat-CSR strong-connection graph ────────────────────
# Two-pass (count → exact pre-allocation → fill).

function _build_strong_csr(A::SparseMatrixCSR, θ::Float64, max_offd::AbstractVector)
    n      = size(A, 1)
    rowptr = A.rowptr; colval = A.colval; nzval = A.nzval

    # Pass 1: count
    strong_count = zeros(Int, n)
    @inbounds for i in 1:n
        thr = θ * max_offd[i]
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]; j == i && continue
            abs(nzval[nzi]) >= thr && (strong_count[i] += 1)
        end
    end

    # Prefix sum → row pointers
    strong_ptr = zeros(Int, n + 1); strong_ptr[1] = 1
    @inbounds for i in 1:n; strong_ptr[i+1] = strong_ptr[i] + strong_count[i]; end

    # Pass 2: fill
    strong_adj = zeros(Int, strong_ptr[n+1] - 1)
    cur = copy(strong_ptr[1:n])
    @inbounds for i in 1:n
        thr = θ * max_offd[i]
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]; j == i && continue
            if abs(nzval[nzi]) >= thr
                strong_adj[cur[i]] = j; cur[i] += 1
            end
        end
    end

    return strong_ptr, strong_adj
end

# ─── Smoothed Aggregation ─────────────────────────────────────────────────────

function _coarsen_SA(A::SparseMatrixCSR, θ::Float64)
    n      = size(A, 1)
    rowptr = A.rowptr; colval = A.colval; nzval = A.nzval

    # 1. Maximum off-diagonal magnitude per row
    max_offd = zeros(eltype(nzval), n)
    @inbounds for i in 1:n
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]; j == i && continue
            v = abs(nzval[nzi]); v > max_offd[i] && (max_offd[i] = v)
        end
    end

    # 2. Flat-CSR strong connections (avoids n separate vector allocations)
    strong_ptr, strong_adj = _build_strong_csr(A, θ, max_offd)

    # 3. Parallel-style MIS-2 aggregation (sequential approximation)
    agg = fill(-1, n); nagg = 0

    # Pass 1: seed selection — nodes with no strongly connected assigned neighbour
    @inbounds for i in 1:n
        has_assigned = false
        for nzi in strong_ptr[i]:(strong_ptr[i+1]-1)
            if agg[strong_adj[nzi]] >= 0; has_assigned = true; break; end
        end
        if !has_assigned; nagg += 1; agg[i] = nagg; end
    end

    # Pass 2: expand — strongly connected neighbours of seeds join same aggregate
    @inbounds for i in 1:n
        agg[i] >= 0 && continue
        for nzi in strong_ptr[i]:(strong_ptr[i+1]-1)
            j = strong_adj[nzi]
            if agg[j] > 0; agg[i] = agg[j]; break; end
        end
    end

    # Pass 3: remaining isolated nodes form singleton aggregates or join nearest
    @inbounds for i in 1:n
        agg[i] >= 0 && continue
        if strong_ptr[i+1] > strong_ptr[i] && agg[strong_adj[strong_ptr[i]]] > 0
            agg[i] = agg[strong_adj[strong_ptr[i]]]
        else
            nagg += 1; agg[i] = nagg
        end
    end

    return agg, nagg
end

# ─── Ruge–Stüben (Classical AMG) ──────────────────────────────────────────────
# Greedy C/F splitting by decreasing lambda (# strong dependents). One-pass sort replaces O(n²) argmax.

function _coarsen_RS(A::SparseMatrixCSR, θ::Float64)
    n      = size(A, 1)
    rowptr = A.rowptr; colval = A.colval; nzval = A.nzval

    # 1. Maximum off-diagonal magnitude per row
    max_offd = zeros(eltype(nzval), n)
    @inbounds for i in 1:n
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]; j == i && continue
            v = abs(nzval[nzi]); v > max_offd[i] && (max_offd[i] = v)
        end
    end

    # 2. Flat-CSR strong connections (single allocation vs n small vectors)
    strong_ptr, strong_adj = _build_strong_csr(A, θ, max_offd)

    # 3. Lambda[i] = number of nodes that strongly depend on i
    lambda = zeros(Int, n)
    @inbounds for i in 1:n
        for nzi in strong_ptr[i]:(strong_ptr[i+1]-1)
            lambda[strong_adj[nzi]] += 1
        end
    end

    # 4. C/F splitting: sort by decreasing lambda, assign C/F in order.
    order  = sortperm(lambda; rev=true)
    status = fill(0, n)   # 0=undecided, 1=C-point, -1=F-point
    @inbounds for idx in 1:n
        i = order[idx]
        status[i] != 0 && continue   # already assigned
        status[i] = 1                # C-point
        for nzi in strong_ptr[i]:(strong_ptr[i+1]-1)
            j = strong_adj[nzi]
            status[j] == 0 && (status[j] = -1)   # F-point
        end
    end

    # 5. Map C-points to aggregate IDs
    nagg = 0
    cmap = zeros(Int, n)
    @inbounds for i in 1:n
        status[i] == 1 && (nagg += 1; cmap[i] = nagg)
    end

    # 6. Assign F-points to the C-point with the strongest connection
    agg = zeros(Int, n)
    @inbounds for i in 1:n
        if status[i] == 1
            agg[i] = cmap[i]
        else
            best_c = 0; best_v = -Inf
            for nzi in rowptr[i]:(rowptr[i+1]-1)
                j = colval[nzi]
                if status[j] == 1 && abs(nzval[nzi]) > best_v
                    best_v = abs(nzval[nzi]); best_c = cmap[j]
                end
            end
            if best_c > 0
                agg[i] = best_c
            else
                nagg += 1; agg[i] = nagg
            end
        end
    end

    return agg, nagg
end

# ─── Pairwise aggregation (fallback) ─────────────────────────────────────────
# Greedy max-weight matching; guarantees ≥ 2× coarsening.

function _coarsen_pairwise(A::SparseMatrixCSR)
    n      = size(A, 1)
    rowptr = A.rowptr; colval = A.colval; nzval = A.nzval

    matched = fill(false, n)
    agg     = zeros(Int, n)
    nagg    = 0

    @inbounds for i in 1:n
        matched[i] && continue

        # Find the strongest unmatched off-diagonal neighbour
        best_j = 0; best_v = zero(eltype(nzval))
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]
            (j == i || matched[j]) && continue
            v = abs(nzval[nzi])
            if v > best_v; best_v = v; best_j = j; end
        end

        nagg += 1; agg[i] = nagg; matched[i] = true
        if best_j > 0; agg[best_j] = nagg; matched[best_j] = true; end
    end

    return agg, nagg
end
