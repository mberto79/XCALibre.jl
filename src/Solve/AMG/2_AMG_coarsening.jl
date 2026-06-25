function _sa_strength_graph(A, threshold)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    diag = zeros(T, n)
    @inbounds for i in 1:n
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            colval[p] == i && (diag[i] += nzval[p])
        end
    end

    strong = Vector{Vector{Int}}(undef, n)
    θ2 = threshold * threshold
    @inbounds for i in 1:n
        count = 0
        limit_i = θ2 * abs(diag[i])
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            abs2(nzval[p]) >= limit_i * abs(diag[j]) && (count += 1)
        end
        strong_i = Vector{Int}(undef, count)
        k = 1
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            if abs2(nzval[p]) >= limit_i * abs(diag[j])
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

_initial_candidates(coarsening::SmoothAggregation) = coarsening.near_nullspace
_initial_candidates(::AbstractAMGCoarsening) = nothing

function _sa_filtered_matrix(A, strong)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    marker = zeros(Int, n)
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, length(nzval))
    sizehint!(J, length(nzval))
    sizehint!(V, length(nzval))
    @inbounds for i in 1:n
        for j in strong[i]
            marker[j] = i
        end
        dii = zero(T)
        lump = zero(T)
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            aij = nzval[p]
            if j == i
                dii += aij
            elseif marker[j] == i
                push!(I, i); push!(J, j); push!(V, aij)
            else
                lump += aij
            end
        end
        push!(I, i); push!(J, i); push!(V, dii + lump)
    end
    return sparse(I, J, V, n, n)
end

function _spectral_radius_DinvA(DinvA, ::Type{T}) where {T}
    n = size(DinvA, 1)
    v = T[isodd(i) ? one(T) : -one(T) for i in 1:n]
    rho = one(T)
    for _ in 1:15
        w = DinvA * v
        nw = norm(w)
        nv = norm(v)
        rho = nw / max(nv, eps(T))
        nw <= eps(T) && break
        v = w ./ nw
    end
    return rho
end

function _smooth_prolongation(A, P, strong, weight)
    weight <= 0 && return P
    Af = _sa_filtered_matrix(A, strong)
    T = eltype(nonzeros(Af))
    d = diag(Af)
    Dinv = T[abs(d[i]) > eps(T) ? one(T) / d[i] : zero(T) for i in eachindex(d)]
    DinvA = Diagonal(Dinv) * Af
    rho = T(11//10) * _spectral_radius_DinvA(DinvA, T) # margin: power iteration approaches rho from below
    omega = T(weight) / max(rho, eps(T))
    Ps = P - omega * (DinvA * P)
    dropzeros!(Ps)
    return Ps
end

function _standard_aggregates(strong)
    n = length(strong)
    x = zeros(Int, n)
    next_aggregate = 1

    for i in 1:n
        x[i] != 0 && continue
        has_neighbors = false
        has_agg_neighbors = false
        for row in strong[i]
            row == i && continue
            has_neighbors = true
            if x[row] != 0
                has_agg_neighbors = true
                break
            end
        end
        if !has_neighbors
            x[i] = -n
        elseif !has_agg_neighbors
            x[i] = next_aggregate
            for row in strong[i]
                x[row] = next_aggregate
            end
            next_aggregate += 1
        end
    end

    for i in 1:n
        x[i] != 0 && continue
        for row in strong[i]
            if x[row] > 0
                x[i] = -x[row]
                break
            end
        end
    end

    next_aggregate -= 1
    agg = zeros(Int, n)
    for i in 1:n
        xi = x[i]
        if xi > 0
            agg[i] = xi
        elseif xi < 0 && xi != -n
            agg[i] = -xi
        else
            next_aggregate += 1
            agg[i] = next_aggregate
            for row in strong[i]
                x[row] == 0 && (x[row] = next_aggregate)
            end
        end
    end

    return agg, next_aggregate
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
    splitting = fill(-1, n)
    @inbounds for seed in sortperm(influence; rev=true)
        splitting[seed] == -1 || continue
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
    strong = _sa_strength_graph(A, coarsening.strength_threshold)
    agg, nagg = _standard_aggregates(strong)
    nagg < 1 && return agg, nothing, candidate_vec
    P0, coarse_candidate = _tentative_prolongation(agg, candidate_vec)
    P = _smooth_prolongation(A, P0, strong, coarsening.smoother_weight)
    return agg, P, coarse_candidate
end

function _geometric_aggregates(A, merge_levels::Int)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    agg = collect(1:n)
    ncl = n
    for _ in 1:merge_levels
        cnt = zeros(Int, ncl)
        @inbounds for i in 1:n
            cnt[agg[i]] += 1
        end
        off = cumsum(vcat(1, cnt)) # cluster->cells offsets
        cells = Vector{Int}(undef, n)
        pos = copy(off)
        @inbounds for i in 1:n
            c = agg[i]
            cells[pos[c]] = i
            pos[c] += 1
        end
        merged = fill(false, ncl)
        newlab = zeros(Int, ncl)
        nnew = 0
        acc = zeros(T, ncl)
        stamp = zeros(Int, ncl)
        order = sortperm(cnt)
        @inbounds for c in order
            merged[c] && continue
            bestc = 0
            bestw = zero(T)
            for t in off[c]:(off[c + 1] - 1)
                i = cells[t]
                for p in rowptr[i]:(rowptr[i + 1] - 1)
                    j = colval[p]
                    cj = agg[j]
                    cj == c && continue
                    if stamp[cj] != c
                        stamp[cj] = c
                        acc[cj] = zero(T)
                    end
                    acc[cj] += abs(nzval[p])
                    if !merged[cj] && acc[cj] > bestw
                        bestw = acc[cj]
                        bestc = cj
                    end
                end
            end
            nnew += 1
            newlab[c] = nnew
            merged[c] = true
            if bestc != 0
                newlab[bestc] = nnew
                merged[bestc] = true
            end
        end
        @inbounds for i in 1:n
            agg[i] = newlab[agg[i]]
        end
        ncl = nnew
    end
    return agg, ncl
end

function build_prolongation(A, coarsening::Geometric, _candidate=nothing, _level_id=1)
    n = _m(A)
    candidate_vec = _near_nullspace_vector(A, nothing)
    n <= 2 && return collect(1:n), nothing, candidate_vec
    agg, nagg = _geometric_aggregates(A, coarsening.merge_levels)
    nagg < 1 && return agg, nothing, candidate_vec
    P0, coarse_candidate = _tentative_prolongation(agg, candidate_vec)
    return agg, P0, coarse_candidate
end

function build_prolongation(A, coarsening::RugeStuben, _candidate=nothing, _level_id=1)
    n = _m(A)
    n <= 2 && return collect(1:n), nothing, nothing
    strong = _rs_strength_graph(A, coarsening.strength_threshold)
    splitting = _rs_coarse_fine_split(strong)
    P = _rs_direct_interpolation(A, strong, splitting)
    isnothing(P) && return splitting, nothing, nothing
    return splitting, P, nothing
end
