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

function _pairwise_aggregates(strong)
    n = length(strong)
    agg = zeros(Int, n)
    next_id = 0
    for i in 1:n
        agg[i] != 0 && continue
        next_id += 1
        agg[i] = next_id
        partner_index = findfirst(j -> agg[j] == 0, strong[i])
        if !isnothing(partner_index)
            agg[strong[i][partner_index]] = next_id
        end
    end
    return agg, next_id
end

function _smooth_prolongation(A, P, weight)
    invdiag = last(_diag_and_inverse(A))
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

function build_prolongation(A, coarsening::SmoothAggregation)
    n = _m(A)
    n <= 2 && return collect(1:n), nothing
    strong = _strength_graph(A, coarsening.strength_threshold)
    agg, nagg = _pairwise_aggregates(strong)
    nagg < 2 && return agg, nothing
    P0 = sparse(collect(1:n), agg, ones(eltype(_nzval(A)), n), n, nagg)
    P = _smooth_prolongation(A, P0, coarsening.smoother_weight)
    return agg, P
end

function _coarse_fine_split(strong)
    n = length(strong)
    state = fill(:undecided, n)
    weights = map(length, strong)
    while any(==(:undecided), state)
        scores = map(i -> state[i] == :undecided ? weights[i] : -1, eachindex(state))
        seed = argmax(scores)
        state[seed] = :coarse
        for j in strong[seed]
            state[j] == :undecided && (state[j] = :fine)
        end
    end
    state
end

function build_prolongation(A, coarsening::RugeStuben)
    n = _m(A)
    n <= 2 && return collect(1:n), nothing
    strong = _strength_graph(A, coarsening.strength_threshold)
    split = _coarse_fine_split(strong)
    coarse_points = findall(==(:coarse), split)
    isempty(coarse_points) && return collect(1:n), nothing
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

        neigh = [j for j in strong[i] if split[j] == :coarse]
        if isempty(neigh)
            push!(coarse_points, i)
            coarse_ids[i] = length(coarse_points)
            push!(I, i)
            push!(J, coarse_ids[i])
            push!(V, one(eltype(_nzval(A))))
        else
            weight = inv(length(neigh))
            for j in neigh
                push!(I, i)
                push!(J, coarse_ids[j])
                push!(V, weight)
            end
        end
    end

    nc = max(maximum(coarse_ids), length(coarse_points))
    nc < 2 && return coarse_ids, nothing
    return coarse_ids, sparse(I, J, V, n, nc)
end
