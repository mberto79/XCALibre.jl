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

function _smooth_prolongation(A, P, weight)
    invdiag = last(_diag_inverse_and_l1(A)[1:2])
    λmax = _estimate_lambda_max(A, invdiag, AMGJacobi())
    α = weight / max(λmax, eps(eltype(invdiag)))
    α <= 0 && return P

    Acsc = _csr_to_csc(A)
    AP = Acsc * P
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

function _standard_aggregates(strong)
    n = length(strong)
    coarse = _coarse_graph(strong)
    weights = map(length, coarse)
    seeds = falses(n)
    agg = zeros(Int, n)

    order = sortperm(weights; rev=true)
    for i in order
        seeds[i] && continue
        any(seeds[j] for j in coarse[i]) && continue
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

    for i in 1:n
        agg[i] != 0 && continue
        best_agg = 0
        best_strength = -1
        for j in coarse[i]
            candidate = agg[j]
            candidate == 0 && continue
            strength = length(intersect(coarse[i], coarse[j]))
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
    strong = _strength_graph(A, coarsening.strength_threshold)
    agg, nagg = _standard_aggregates(strong)
    nagg < 2 && return agg, nothing, candidate_vec
    P0, coarse_candidate = _tentative_prolongation(agg, candidate_vec)
    P = _smooth_prolongation(A, P0, coarsening.smoother_weight)
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
    strong = _strength_graph(A, coarsening.strength_threshold)
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
