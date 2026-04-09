# ─── Setup phase: coarsening strategies ───────────────────────────────────────
# Produces aggregate IDs (integer array mapping each fine node to an aggregate).
# Implemented on CPU only (setup is called once; GPU arrays are gathered here).

"""
    amg_coarsen(A_cpu, strength, strategy) → agg_ids

Given a CSR matrix on CPU, return a `Vector{Int}` mapping each row to an aggregate
index (1-based). Aggregates define the coarse degrees of freedom.

Strategies:
- `:SA` — Smoothed Aggregation (parallel-friendly MIS-style)
- `:RS` — Ruge–Stüben classical AMG (strength-of-connection + C/F splitting)
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

# ─── Smoothed Aggregation ─────────────────────────────────────────────────────

function _coarsen_SA(A::SparseMatrixCSR, θ::Float64)
    n = size(A, 1)
    rowptr = A.rowptr
    colval = A.colval
    nzval  = A.nzval

    # 1. Precompute diagonal magnitudes
    diag = zeros(eltype(nzval), n)
    for i in 1:n
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            if colval[nzi] == i
                diag[i] = abs(nzval[nzi])
                break
            end
        end
    end

    # 2. Build strength-of-connection: edge (i,j) is strong if
    #    |a_ij| >= θ * sqrt(|a_ii| * |a_jj|)
    strong = [Int[] for _ in 1:n]
    for i in 1:n
        thr = θ * sqrt(diag[i])
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]
            j == i && continue
            if abs(nzval[nzi]) >= thr * sqrt(diag[j])
                push!(strong[i], j)
            end
        end
    end

    # 3. Parallel-style MIS-2 aggregation (sequential approximation)
    agg = fill(-1, n)        # -1 = unassigned
    nagg = 0

    # Pass 1: seed selection — nodes with no strongly connected assigned neighbour
    for i in 1:n
        has_assigned = any(agg[j] >= 0 for j in strong[i])
        if !has_assigned
            nagg += 1
            agg[i] = nagg
        end
    end

    # Pass 2: expand — strongly connected neighbours of seeds join same aggregate
    for i in 1:n
        agg[i] >= 0 && continue
        for j in strong[i]
            if agg[j] > 0
                agg[i] = agg[j]
                break
            end
        end
    end

    # Pass 3: remaining isolated nodes form singleton aggregates or join nearest
    for i in 1:n
        if agg[i] < 0
            # join nearest strong neighbour's aggregate, else singleton
            if !isempty(strong[i]) && agg[strong[i][1]] > 0
                agg[i] = agg[strong[i][1]]
            else
                nagg += 1
                agg[i] = nagg
            end
        end
    end

    return agg, nagg
end

# ─── Ruge–Stüben (Classical AMG) ──────────────────────────────────────────────

function _coarsen_RS(A::SparseMatrixCSR, θ::Float64)
    n = size(A, 1)
    rowptr = A.rowptr
    colval = A.colval
    nzval  = A.nzval

    # 1. Maximum off-diagonal magnitude per row
    max_offd = zeros(eltype(nzval), n)
    for i in 1:n
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]
            j == i && continue
            v = abs(nzval[nzi])
            if v > max_offd[i]
                max_offd[i] = v
            end
        end
    end

    # 2. Strong connections: |a_ij| >= θ * max_j |a_ij|
    strong = [Int[] for _ in 1:n]
    for i in 1:n
        thr = θ * max_offd[i]
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            j = colval[nzi]
            j == i && continue
            if abs(nzval[nzi]) >= thr
                push!(strong[i], j)
            end
        end
    end

    # 3. C/F splitting via a greedy pass (CLJP-like, sequential approximation)
    # Lambda[i] = number of points that strongly depend on i
    lambda = zeros(Int, n)
    for i in 1:n
        for j in strong[i]
            lambda[j] += 1
        end
    end

    status = fill(0, n)   # 0=undecided, 1=C-point, -1=F-point
    undecided = collect(1:n)

    while !isempty(undecided)
        # Pick node with highest lambda
        best = argmax(lambda[undecided])
        i = undecided[best]
        status[i] = 1   # C-point

        # Make all strongly dependent F-points
        for j in strong[i]
            if status[j] == 0
                status[j] = -1
                # Update lambda for strong neighbours of j
                for k in strong[j]
                    if status[k] == 0
                        lambda[k] += 1
                    end
                end
            end
        end

        filter!(k -> status[k] == 0, undecided)
    end

    # 4. Map C-points to aggregates
    nagg = 0
    cmap = zeros(Int, n)
    for i in 1:n
        if status[i] == 1
            nagg += 1
            cmap[i] = nagg
        end
    end

    # 5. Assign F-points to the C-point with the strongest connection
    agg = zeros(Int, n)
    for i in 1:n
        if status[i] == 1
            agg[i] = cmap[i]
        else
            best_c = 0
            best_v = -Inf
            for nzi in rowptr[i]:(rowptr[i+1]-1)
                j = colval[nzi]
                if status[j] == 1 && abs(nzval[nzi]) > best_v
                    best_v = abs(nzval[nzi])
                    best_c = cmap[j]
                end
            end
            if best_c > 0
                agg[i] = best_c
            else
                # isolated — make singleton
                nagg += 1
                agg[i] = nagg
            end
        end
    end

    return agg, nagg
end
