function _strength_graph(A, threshold)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    strong = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        max_offdiag = zero(eltype(nzval))
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            max_offdiag = max(max_offdiag, abs(nzval[p]))
        end
        limit = threshold * max_offdiag
        count = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            abs(nzval[p]) >= limit && (count += 1)
        end
        strong_i = Vector{Int}(undef, count)
        k = 1
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            if abs(nzval[p]) >= limit
                strong_i[k] = j
                k += 1
            end
        end
        strong[i] = strong_i
    end
    return strong
end

function _rs_strength_graph(A, threshold)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    strong = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        max_neg = zero(eltype(nzval))
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            j == i && continue
            real(aij) < 0 || continue
            max_neg = max(max_neg, abs(aij))
        end
        limit = threshold * max_neg
        count = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            j == i && continue
            real(aij) < 0 || continue
            abs(aij) >= limit && (count += 1)
        end
        strong_i = Vector{Int}(undef, count)
        k = 1
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            j == i && continue
            real(aij) < 0 || continue
            if abs(aij) >= limit
                strong_i[k] = j
                k += 1
            end
        end
        strong[i] = strong_i
    end
    return strong
end

function _coarse_graph(strong)
    n = length(strong)
    counts = zeros(Int, n)
    for i in 1:n
        counts[i] += length(strong[i])
        for j in strong[i]
            counts[j] += 1
        end
    end

    coarse = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        coarse[i] = Vector{Int}(undef, counts[i])
    end

    next = ones(Int, n)
    for i in 1:n
        for j in strong[i]
            coarse[i][next[i]] = j
            next[i] += 1
            coarse[j][next[j]] = i
            next[j] += 1
        end
    end
    for i in 1:n
        sort!(coarse[i])
        unique!(coarse[i])
    end
    return coarse
end

function _level_strength_threshold(coarsening::SmoothAggregation, level_id)
    thresholds = coarsening.level_strength_thresholds
    isnothing(thresholds) && return coarsening.strength_threshold
    isempty(thresholds) && return coarsening.strength_threshold
    return thresholds[min(level_id, length(thresholds))]
end

_level_strength_threshold(coarsening::RugeStuben, level_id) = coarsening.strength_threshold
_initial_candidates(coarsening::SmoothAggregation) = coarsening.near_nullspace
_initial_candidates(::AbstractAMGCoarsening) = nothing

function _coarse_drop_tolerance(coarsening::SmoothAggregation, coarse_level_id)
    tolerances = coarsening.coarse_drop_tolerances
    isempty(tolerances) && return 0.0
    return tolerances[min(coarse_level_id, length(tolerances))]
end

_coarse_drop_tolerance(::AbstractAMGCoarsening, coarse_level_id) = 0.0

function _aggregate_adjacency(agg, strong, nagg)
    adjacency = [Int[] for _ in 1:nagg]
    for i in eachindex(strong)
        ai = agg[i]
        for j in strong[i]
            aj = agg[j]
            ai == aj && continue
            push!(adjacency[ai], aj)
            push!(adjacency[aj], ai)
        end
    end
    for neighbors in adjacency
        sort!(neighbors)
        unique!(neighbors)
    end
    return adjacency
end

function _aggressive_aggregates(agg, strong, nagg, passes)
    passes <= 0 && return agg, nagg
    current_agg = agg
    current_nagg = nagg
    for _ in 1:passes
        current_nagg <= 2 && break
        aggregate_graph = _aggregate_adjacency(current_agg, strong, current_nagg)
        merged_agg, merged_nagg = _standard_aggregates(aggregate_graph)
        merged_nagg >= current_nagg && break
        current_agg = [merged_agg[id] for id in current_agg]
        current_nagg = merged_nagg
    end
    return current_agg, current_nagg
end

function _smooth_prolongation(A, P, weight)
    _, invdiag = _diag_inverse(A)
    lambda_max = _estimate_lambda_max(A, invdiag)
    alpha = weight / max(lambda_max, eps(eltype(invdiag)))
    alpha <= 0 && return P

    Acsc = _csr_to_csc(A)
    AP = Acsc * P
    rowval = rowvals(AP)
    nzval = nonzeros(AP)
    @inbounds for j in 1:size(AP, 2)
        for p in nzrange(AP, j)
            nzval[p] *= alpha * invdiag[rowval[p]]
        end
    end

    Ps = P - AP
    dropzeros!(Ps)
    return Ps
end

function _truncate_prolongation(P, max_entries::Integer)
    max_entries <= 0 && return P
    I, J, V = findnz(P)
    n = size(P, 1)
    nnzP = length(V)
    keep = falses(nnzP)
    nnzP == 0 && return P

    row_counts = zeros(Int, n)
    row_slots = fill(0, n, max_entries)
    row_magnitudes = fill(zero(eltype(V)), n, max_entries)
    @inbounds for k in eachindex(I)
        row = I[k]
        row_counts[row] += 1
        magnitude = abs(V[k])
        insert_at = 0
        for slot in 1:max_entries
            if row_slots[row, slot] == 0 || magnitude > row_magnitudes[row, slot]
                insert_at = slot
                break
            end
        end
        insert_at == 0 && continue
        for slot in max_entries:-1:(insert_at + 1)
            row_slots[row, slot] = row_slots[row, slot - 1]
            row_magnitudes[row, slot] = row_magnitudes[row, slot - 1]
        end
        row_slots[row, insert_at] = k
        row_magnitudes[row, insert_at] = magnitude
    end

    kept_nnz = 0
    @inbounds for row in 1:n
        kept_nnz += min(row_counts[row], max_entries)
        for slot in 1:max_entries
            k = row_slots[row, slot]
            k == 0 && break
            keep[k] = true
        end
    end

    original_row_sum = zeros(eltype(V), n)
    truncated_row_sum = zeros(eltype(V), n)
    @inbounds for k in eachindex(V)
        original_row_sum[I[k]] += V[k]
        keep[k] && (truncated_row_sum[I[k]] += V[k])
    end

    It = Int[]
    Jt = Int[]
    Vt = eltype(V)[]
    sizehint!(It, kept_nnz)
    sizehint!(Jt, kept_nnz)
    sizehint!(Vt, kept_nnz)
    @inbounds for k in eachindex(V)
        keep[k] || continue
        scale = abs(truncated_row_sum[I[k]]) > eps(eltype(V)) ? original_row_sum[I[k]] / truncated_row_sum[I[k]] : one(eltype(V))
        push!(It, I[k])
        push!(Jt, J[k])
        push!(Vt, V[k] * scale)
    end
    Pt = sparse(It, Jt, Vt, size(P, 1), size(P, 2))
    dropzeros!(Pt)
    return Pt
end

function _standard_aggregates(strong)
    n = length(strong)
    coarse = _coarse_graph(strong)
    weights = map(length, coarse)
    seeds = falses(n)
    agg = zeros(Int, n)

    order = sortperm(weights; rev=true)
    for i in order
        seeds[i] && continue
        has_seed_neighbor = false
        for j in coarse[i]
            if seeds[j]
                has_seed_neighbor = true
                break
            end
        end
        has_seed_neighbor && continue
        seeds[i] = true
    end

    next_id = 0
    for seed in eachindex(seeds)
        seeds[seed] || continue
        next_id += 1
        agg[seed] = next_id
        for j in coarse[seed]
            agg[j] == 0 && (agg[j] = next_id)
        end
    end

    marker = zeros(Int, n)
    stamp = 0
    for i in 1:n
        agg[i] != 0 && continue
        best_agg = 0
        best_strength = -1
        stamp += 1
        for k in coarse[i]
            marker[k] = stamp
        end
        for j in coarse[i]
            candidate = agg[j]
            candidate == 0 && continue
            strength = 0
            for k in coarse[j]
                marker[k] == stamp && (strength += 1)
            end
            if strength > best_strength
                best_strength = strength
                best_agg = candidate
            end
        end
        if best_agg != 0
            agg[i] = best_agg
        else
            next_id += 1
            agg[i] = next_id
        end
    end

    return agg, next_id
end

function _strong_row_sets(strong)
    return [Set(row) for row in strong]
end

function _strong_transpose(strong)
    n = length(strong)
    incoming = [Int[] for _ in 1:n]
    for i in 1:n
        for j in strong[i]
            push!(incoming[j], i)
        end
    end
    return incoming
end

function _has_common_coarse_neighbor(strong_sets, splitting, i, j)
    for k in strong_sets[i]
        splitting[k] == 1 || continue
        k in strong_sets[j] && return true
    end
    return false
end

function _rs_coarse_fine_split(strong)
    n = length(strong)
    n == 0 && return Int[]
    incoming = _strong_transpose(strong)
    influence = [length(strong[i]) + length(incoming[i]) for i in 1:n]
    splitting = fill(-1, n) # -1 undecided, 0 F, 1 C

    while true
        seed = 0
        best_weight = -1
        for i in 1:n
            splitting[i] == -1 || continue
            if influence[i] > best_weight
                best_weight = influence[i]
                seed = i
            end
        end
        seed == 0 && break
        splitting[seed] = 1
        for j in strong[seed]
            splitting[j] == -1 && (splitting[j] = 0)
        end
        for j in incoming[seed]
            splitting[j] == -1 && (splitting[j] = 0)
        end
    end

    strong_sets = _strong_row_sets(strong)
    for i in 1:n
        splitting[i] == 0 || continue
        has_c_neighbor = any(splitting[j] == 1 for j in strong[i])
        if !has_c_neighbor
            splitting[i] = 1
            continue
        end
        for j in strong[i]
            splitting[j] == 0 || continue
            _has_common_coarse_neighbor(strong_sets, splitting, i, j) && continue
            promote = influence[i] >= influence[j] ? i : j
            splitting[promote] = 1
        end
    end

    for i in 1:n
        splitting[i] == -1 && (splitting[i] = 1)
    end
    return splitting
end

function _rs_coarse_index(splitting)
    coarse_index = zeros(Int, length(splitting))
    nc = 0
    for i in eachindex(splitting)
        if splitting[i] == 1
            nc += 1
            coarse_index[i] = nc
        end
    end
    return coarse_index, nc
end

function _rs_direct_interpolation(A, strong, splitting)
    coarse_index, nc = _rs_coarse_index(splitting)
    nc < 2 && return nothing

    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)

    I = Int[]
    J = Int[]
    V = T[]

    for i in 1:n
        if splitting[i] == 1
            push!(I, i)
            push!(J, coarse_index[i])
            push!(V, one(T))
            continue
        end

        strong_c = Int[]
        strong_c_pos = T[]
        sum_strong_neg = zero(T)
        sum_strong_pos = zero(T)
        for j in strong[i]
            splitting[j] == 1 || continue
            aij = zero(T)
            for p in rowptr[i]:(rowptr[i + 1] - 1)
                if colval[p] == j
                    aij = nzval[p]
                    break
                end
            end
            push!(strong_c, j)
            push!(strong_c_pos, aij)
            if real(aij) < 0
                sum_strong_neg += aij
            else
                sum_strong_pos += aij
            end
        end

        if isempty(strong_c)
            push!(I, i)
            push!(J, coarse_index[i] > 0 ? coarse_index[i] : 1)
            push!(V, one(T))
            continue
        end

        sum_all_neg = zero(T)
        sum_all_pos = zero(T)
        diag = zero(T)
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            if j == i
                diag += aij
            elseif real(aij) < 0
                sum_all_neg += aij
            else
                sum_all_pos += aij
            end
        end

        if iszero(sum_strong_pos)
            beta = zero(T)
            real(diag) >= 0 && (diag += sum_all_pos)
        else
            beta = sum_all_pos / sum_strong_pos
        end
        if iszero(sum_strong_neg)
            alpha = zero(T)
            real(diag) < 0 && (diag += sum_all_neg)
        else
            alpha = sum_all_neg / sum_strong_neg
        end

        real_diag = real(diag)
        atol = eps(real_diag + one(real_diag))
        neg_coeff = isapprox(real_diag, 0; atol=atol) ? zero(T) : alpha / diag
        pos_coeff = isapprox(real_diag, 0; atol=atol) ? zero(T) : beta / diag
        row_sum = zero(T)
        row_offset = length(V)
        for (idx, j) in pairs(strong_c)
            aij = strong_c_pos[idx]
            wij = real(aij) < 0 ? abs(neg_coeff * aij) : abs(pos_coeff * aij)
            push!(I, i)
            push!(J, coarse_index[j])
            push!(V, wij)
            row_sum += wij
        end

        if row_sum > eps(T)
            scale = inv(row_sum)
            for k in (row_offset + 1):length(V)
                V[k] *= scale
            end
        end
    end

    P = sparse(I, J, V, n, nc)
    dropzeros!(P)
    return P
end

function _near_nullspace_vector(A, candidate)
    T = eltype(_nzval(A))
    n = _m(A)
    if isnothing(candidate)
        return ones(T, n)
    end
    length(candidate) == n || throw(ArgumentError("SmoothAggregation near_nullspace must match the matrix row count"))
    return T.(collect(candidate))
end

function _tentative_prolongation(agg, candidate)
    n = length(agg)
    nc = maximum(agg)
    T = eltype(candidate)
    coarse_candidate = zeros(T, nc)
    for i in 1:n
        coarse_candidate[agg[i]] += candidate[i]^2
    end
    for j in eachindex(coarse_candidate)
        coarse_candidate[j] = sqrt(coarse_candidate[j])
        coarse_candidate[j] <= eps(T) && (coarse_candidate[j] = one(T))
    end

    I = collect(1:n)
    J = copy(agg)
    V = similar(candidate)
    for i in 1:n
        V[i] = candidate[i] / coarse_candidate[agg[i]]
    end
    return sparse(I, J, V, n, nc), coarse_candidate
end

function build_prolongation(A, coarsening::SmoothAggregation, candidate=nothing, level_id=1)
    n = _m(A)
    candidate_vec = _near_nullspace_vector(A, isnothing(candidate) ? coarsening.near_nullspace : candidate)
    n <= 2 && return collect(1:n), nothing, candidate_vec
    strong = _strength_graph(A, _level_strength_threshold(coarsening, level_id))
    agg, nagg = _standard_aggregates(strong)
    if level_id <= coarsening.aggressive_levels
        agg, nagg = _aggressive_aggregates(agg, strong, nagg, coarsening.aggressive_passes)
    end
    nagg < 2 && return agg, nothing, candidate_vec
    P0, coarse_candidate = _tentative_prolongation(agg, candidate_vec)
    P = _smooth_prolongation(A, P0, coarsening.smoother_weight)
    P = _truncate_prolongation(P, coarsening.max_prolongation_entries)
    return agg, P, coarse_candidate
end

function build_prolongation(A, coarsening::RugeStuben, candidate=nothing, level_id=1)
    n = _m(A)
    n <= 2 && return collect(1:n), nothing, nothing
    strong = _rs_strength_graph(A, _level_strength_threshold(coarsening, level_id))
    splitting = _rs_coarse_fine_split(strong)
    P = _rs_direct_interpolation(A, strong, splitting)
    isnothing(P) && return splitting, nothing, nothing
    return splitting, P, nothing
end
