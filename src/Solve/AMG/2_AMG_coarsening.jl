function _classical_strength_graph(A, threshold)
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
        strong_i = Int[]
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            abs(nzval[p]) >= limit && push!(strong_i, j)
        end
        strong[i] = strong_i
    end
    return strong
end

function _symmetric_strength_graph(A, threshold)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    diag = zeros(Float64, n)
    for i in 1:n
        aii = 0.0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[p] == i
                aii = abs(float(real(nzval[p])))
                break
            end
        end
        diag[i] = max(aii, eps(Float64))
    end

    strong = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        strong_i = Int[]
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            limit = threshold * sqrt(diag[i] * diag[j])
            abs(float(real(nzval[p]))) >= limit && push!(strong_i, j)
        end
        strong[i] = strong_i
    end
    return strong
end

function _strength_graph(A, coarsening::SmoothAggregation)
    if coarsening.strength_measure == :symmetric
        return _symmetric_strength_graph(A, coarsening.strength_threshold)
    end
    return _classical_strength_graph(A, coarsening.strength_threshold)
end

function _strength_graph(A, coarsening::RugeStuben)
    if coarsening.strength_measure == :symmetric
        return _symmetric_strength_graph(A, coarsening.strength_threshold)
    end
    return _classical_strength_graph(A, coarsening.strength_threshold)
end

function _coarse_graph(strong)
    n = length(strong)
    coarse = [Set{Int}() for _ in 1:n]
    for i in 1:n
        for j in strong[i]
            push!(coarse[i], j)
            push!(coarse[j], i)
        end
    end
    return coarse
end

function _filtered_smoothing_matrix(A, strong)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    I = Int[]
    J = Int[]
    V = eltype(nzval)[]
    for i in 1:n
        strong_i = Set(strong[i])
        diag_value = zero(eltype(nzval))
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            if j == i
                diag_value += aij
            elseif j in strong_i
                push!(I, i)
                push!(J, j)
                push!(V, aij)
            else
                diag_value += aij
            end
        end
        push!(I, i)
        push!(J, i)
        push!(V, diag_value)
    end
    return sparse(I, J, V, n, n)
end

function _smooth_prolongation_pass(smoothing_A, invdiag, P, α)
    AP = smoothing_A * P
    rowval = rowvals(AP)
    nzval = nonzeros(AP)
    @inbounds for j in 1:size(AP, 2)
        for p in nzrange(AP, j)
            nzval[p] *= α * invdiag[rowval[p]]
        end
    end

    Ps = P - AP
    dropzeros!(Ps)
    return Ps
end

function _smooth_prolongation(A, strong, P, weight; filter_weak_connections, passes)
    smoothing_A = filter_weak_connections ? _filtered_smoothing_matrix(A, strong) : _csr_to_csc(A)
    smoothing_wrap = _wrap_sparse(smoothing_A)
    _, invdiag = _diag_inverse_and_l1(smoothing_wrap)
    λmax = _estimate_lambda_max(smoothing_wrap, invdiag, AMGJacobi())
    α = weight / max(λmax, eps(eltype(invdiag)))
    α <= 0 && return P
    Ps = P
    for _ in 1:passes
        Ps = _smooth_prolongation_pass(smoothing_A, invdiag, Ps, α)
    end
    return Ps
end

function _truncate_smoothed_prolongation(P, candidate, coarse_candidate, truncate_factor, max_interp_entries)
    truncate_factor <= 0 && max_interp_entries == 0 && return P

    I, J, V = findnz(P)
    n = size(P, 1)
    T = eltype(V)
    row_cols = [Int[] for _ in 1:n]
    row_vals = [T[] for _ in 1:n]

    for k in eachindex(I)
        i = I[k]
        push!(row_cols[i], J[k])
        push!(row_vals[i], V[k])
    end

    keep_I = Int[]
    keep_J = Int[]
    keep_V = T[]
    epsT = eps(T)
    for i in 1:n
        cols = row_cols[i]
        vals = row_vals[i]
        isempty(vals) && continue

        rowmax = maximum(abs, vals)
        limit = truncate_factor * rowmax
        keep = Int[]
        for k in eachindex(vals)
            if truncate_factor <= 0 || abs(vals[k]) >= limit
                push!(keep, k)
            end
        end
        if max_interp_entries > 0 && length(keep) > max_interp_entries
            keep = sort(keep; by=k -> abs(vals[k]), rev=true)[1:max_interp_entries]
        end
        if isempty(keep)
            strongest = 1
            strongest_val = abs(vals[1])
            for k in 2:length(vals)
                candidate_val = abs(vals[k])
                if candidate_val > strongest_val
                    strongest = k
                    strongest_val = candidate_val
                end
            end
            push!(keep, strongest)
        end

        denom = zero(T)
        for k in keep
            denom += vals[k] * coarse_candidate[cols[k]]
        end
        scale = abs(denom) > epsT ? candidate[i] / denom : one(T)

        for k in keep
            push!(keep_I, i)
            push!(keep_J, cols[k])
            push!(keep_V, vals[k] * scale)
        end
    end

    Pt = sparse(keep_I, keep_J, keep_V, size(P, 1), size(P, 2))
    dropzeros!(Pt)
    return Pt
end

function _common_neighbour_count(coarse, i, j)
    common = 0
    coarse_i = coarse[i]
    for k in coarse[j]
        k == i && continue
        k in coarse_i && (common += 1)
    end
    return common
end

function _best_unaggregated_match(i, coarse, agg, weights)
    best_j = 0
    best_common = -1
    best_weight = typemax(Int)
    for j in coarse[i]
        agg[j] == 0 || continue
        common = _common_neighbour_count(coarse, i, j)
        weight = weights[j]
        if common > best_common ||
           (common == best_common && weight < best_weight) ||
           (common == best_common && weight == best_weight && (best_j == 0 || j < best_j))
            best_j = j
            best_common = common
            best_weight = weight
        end
    end
    return best_j
end

function _best_neighbouring_aggregate(i, coarse, agg, agg_sizes)
    scores = Dict{Int, Int}()
    for j in coarse[i]
        agg_j = agg[j]
        agg_j == 0 && continue
        scores[agg_j] = get(scores, agg_j, 0) + 1
    end

    best_agg = 0
    best_links = 0
    best_size = typemax(Int)
    for (agg_id, links) in scores
        size = agg_sizes[agg_id]
        if links > best_links ||
           (links == best_links && size < best_size) ||
           (links == best_links && size == best_size && (best_agg == 0 || agg_id < best_agg))
            best_agg = agg_id
            best_links = links
            best_size = size
        end
    end
    return best_agg
end

function _standard_aggregates(strong)
    n = length(strong)
    coarse = _coarse_graph(strong)
    weights = map(length, coarse)
    agg = zeros(Int, n)
    agg_sizes = Int[]
    next_id = 0

    order = sortperm(weights; rev=true)
    for i in order
        agg[i] == 0 || continue
        match = _best_unaggregated_match(i, coarse, agg, weights)
        match == 0 && continue

        next_id += 1
        push!(agg_sizes, 2)
        agg[i] = next_id
        agg[match] = next_id
    end

    progress = true
    while progress
        progress = false
        for i in order
            agg[i] == 0 || continue
            best_agg = _best_neighbouring_aggregate(i, coarse, agg, agg_sizes)
            best_agg == 0 && continue
            agg[i] = best_agg
            agg_sizes[best_agg] += 1
            progress = true
        end
    end

    for i in order
        agg[i] == 0 || continue
        match = _best_unaggregated_match(i, coarse, agg, weights)
        if match != 0
            next_id += 1
            push!(agg_sizes, 2)
            agg[i] = next_id
            agg[match] = next_id
            continue
        end

        best_agg = _best_neighbouring_aggregate(i, coarse, agg, agg_sizes)
        if best_agg != 0
            agg[i] = best_agg
            agg_sizes[best_agg] += 1
            continue
        end

        next_id += 1
        push!(agg_sizes, 1)
        agg[i] = next_id
    end

    return agg, next_id
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

function build_prolongation(A, coarsening::SmoothAggregation, candidate=nothing)
    n = _m(A)
    candidate_vec = _near_nullspace_vector(A, isnothing(candidate) ? coarsening.near_nullspace : candidate)
    n <= 2 && return collect(1:n), nothing, candidate_vec
    strong = _strength_graph(A, coarsening)
    agg, nagg = _standard_aggregates(strong)
    nagg < 2 && return agg, nothing, candidate_vec
    P0, coarse_candidate = _tentative_prolongation(agg, candidate_vec)
    P = _smooth_prolongation(
        A,
        strong,
        P0,
        coarsening.smoother_weight;
        filter_weak_connections=coarsening.filter_weak_connections,
        passes=coarsening.interpolation_passes
    )
    P = _truncate_smoothed_prolongation(P, candidate_vec, coarse_candidate, coarsening.truncate_factor, coarsening.max_interp_entries)
    return agg, P, coarse_candidate
end

function _coarse_fine_split(strong)
    n = length(strong)
    state = fill(:undecided, n)
    coarse_graph = _coarse_graph(strong)
    weights = map(length, coarse_graph)

    while any(==(:undecided), state)
        scores = map(i -> state[i] == :undecided ? weights[i] : -1, eachindex(state))
        seed = argmax(scores)
        state[seed] = :coarse
        for j in coarse_graph[seed]
            state[j] == :undecided && (state[j] = :fine)
        end
    end

    for i in 1:n
        state[i] == :fine || continue
        fine_neigh = [j for j in coarse_graph[i] if state[j] == :fine]
        for j in fine_neigh
            common_coarse = any(k -> state[k] == :coarse && (k in coarse_graph[j]), coarse_graph[i])
            common_coarse && continue
            promote = weights[i] >= weights[j] ? i : j
            state[promote] = :coarse
        end
    end

    return state
end

function _point_interpolation_weight(A, i, j, coarse_neighbors, fine_neighbors)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    aii_idx = spindex(rowptr, colval, i, i)
    aii = aii_idx == 0 ? one(eltype(nzval)) : nzval[aii_idx]
    abs(aii) <= eps(eltype(nzval)) && return zero(eltype(nzval))

    ij = spindex(rowptr, colval, i, j)
    ij == 0 && return zero(eltype(nzval))
    contribution = nzval[ij]

    for m in fine_neighbors
        im = spindex(rowptr, colval, i, m)
        im == 0 && continue

        denom = zero(eltype(nzval))
        mj = 0
        for k in coarse_neighbors
            mk = spindex(rowptr, colval, m, k)
            mk == 0 && continue
            denom += nzval[mk]
            k == j && (mj = mk)
        end

        abs(denom) <= eps(eltype(nzval)) && continue
        mj == 0 && continue
        contribution += nzval[im] * nzval[mj] / denom
    end

    return -contribution / aii
end

function build_prolongation(A, coarsening::RugeStuben, candidate=nothing)
    n = _m(A)
    n <= 2 && return collect(1:n), nothing, nothing
    strong = _strength_graph(A, coarsening)
    split = _coarse_fine_split(strong)
    coarse_graph = _coarse_graph(strong)
    coarse_points = findall(==(:coarse), split)
    isempty(coarse_points) && return collect(1:n), nothing, nothing
    coarse_ids = zeros(Int, n)
    for (k, i) in enumerate(coarse_points)
        coarse_ids[i] = k
    end

    I = Int[]
    J = Int[]
    V = eltype(_nzval(A))[]
    for i in 1:n
        if split[i] == :coarse
            push!(I, i)
            push!(J, coarse_ids[i])
            push!(V, one(eltype(_nzval(A))))
            continue
        end

        coarse_neighbors = [j for j in coarse_graph[i] if split[j] == :coarse]
        fine_neighbors = [j for j in coarse_graph[i] if split[j] == :fine]
        if isempty(coarse_neighbors)
            push!(coarse_points, i)
            coarse_ids[i] = length(coarse_points)
            split[i] = :coarse
            push!(I, i)
            push!(J, coarse_ids[i])
            push!(V, one(eltype(_nzval(A))))
            continue
        end

        weights = [_point_interpolation_weight(A, i, j, coarse_neighbors, fine_neighbors) for j in coarse_neighbors]
        weight_sum = sum(abs, weights)
        if weight_sum <= eps(eltype(_nzval(A)))
            fill!(weights, inv(length(weights)))
        else
            for k in eachindex(weights)
                weights[k] /= weight_sum
            end
        end

        for (j, weight) in zip(coarse_neighbors, weights)
            push!(I, i)
            push!(J, coarse_ids[j])
            push!(V, weight)
        end
    end

    nc = max(maximum(coarse_ids), length(coarse_points))
    nc < 2 && return coarse_ids, nothing, nothing
    return coarse_ids, sparse(I, J, V, n, nc), nothing
end
