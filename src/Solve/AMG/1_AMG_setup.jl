function _cpu_vector(x::AbstractVector)
    Array(x)
end

function _amg_setup_backend(backend)
    backend
end

function _amg_setup_matrix(A, backend)
    A
end

function _amg_needs_cpu_apply(A)
    false
end

function _csr_triplets(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    m = length(rowptr) - 1
    nnz = length(nzval)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{eltype(nzval)}(undef, nnz)
    k = 1
    for i in 1:m
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            I[k] = i
            J[k] = colval[p]
            V[k] = nzval[p]
            k += 1
        end
    end
    return I, J, V
end

function _csr_to_csc(A)
    I, J, V = _csr_triplets(A)
    sparse(I, J, V, _m(A), _n(A))
end

function _wrap_sparse(A::SparseMatrixCSC)
    I, J, V = findnz(A)
    SparseXCSR(sparsecsr(I, J, V, size(A, 1), size(A, 2)))
end

function _diag_and_inverse(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    d = zeros(T, n)
    dinv = zeros(T, n)
    for i in 1:n
        idx = spindex(rowptr, colval, i, i)
        aii = idx == 0 ? one(T) : nzval[idx]
        d[i] = aii
        dinv[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
    return d, dinv
end

function _diag_and_inverse!(diag, invdiag, A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(nzval)
    for i in eachindex(diag)
        idx = spindex(rowptr, colval, i, i)
        aii = idx == 0 ? one(T) : nzval[idx]
        diag[i] = aii
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
    return diag, invdiag
end

function _estimate_lambda_max(A, invdiag, smoother::AbstractAMGSmoother)
    T = eltype(invdiag)
    v = ones(T, _m(A))
    w = similar(v)
    λ = one(T)
    for _ in 1:(smoother isa AMGChebyshev ? max(1, smoother.power_iterations) : 3)
        mul!(w, A, v)
        @inbounds for i in eachindex(w)
            w[i] *= invdiag[i]
        end
        λ = max(norm(w), eps(T))
        @inbounds for i in eachindex(v)
            v[i] = w[i] / λ
        end
    end
    return max(λ, one(T))
end

function _allocate_level(A, level_id, aggregate_ids, backend, smoother)
    n = _m(A)
    T = eltype(_nzval(A))
    diag, invdiag = _diag_and_inverse(A)
    rhs = KernelAbstractions.zeros(backend, T, n)
    x = KernelAbstractions.zeros(backend, T, n)
    tmp = KernelAbstractions.zeros(backend, T, n)
    coarse_work = KernelAbstractions.zeros(backend, T, max(1, n))
    λ = _estimate_lambda_max(A, invdiag, smoother)
    return AMGLevel(A, nothing, nothing, diag, invdiag, rhs, x, tmp, aggregate_ids, coarse_work, λ, level_id, nothing)
end

function refresh_hierarchy!(hierarchy::AMGHierarchy, solver::AMG)
    level = hierarchy.levels[1]
    _diag_and_inverse!(level.diagonal, level.inv_diagonal, level.A)
    level.lambda_max = _estimate_lambda_max(level.A, level.inv_diagonal, solver.smoother)
    hierarchy.nnz = length(_nzval(level.A))
    hierarchy.nrows = _m(level.A)
    return hierarchy
end

function _should_stop_coarsening(n, coarse_rows, solver::AMG, level_id)
    coarse_rows < 2 || coarse_rows >= n || coarse_rows <= solver.min_coarse_rows ||
        coarse_rows <= solver.max_coarse_rows || level_id >= solver.max_levels
end

function setup_hierarchy(A, solver::AMG, backend)
    levels = AMGLevel[]
    current_A = A
    level_id = 1
    while true
        n = _m(current_A)
        aggregate_ids, P = build_prolongation(current_A, solver.coarsening)
        level = _allocate_level(current_A, level_id, aggregate_ids, backend, solver.smoother)

        if isnothing(P)
            push!(levels, level)
            break
        end

        R = transpose(P)
        coarse_A = _wrap_sparse(R * _csr_to_csc(current_A) * P)
        level = AMGLevel(current_A, P, R, level.diagonal, level.inv_diagonal, level.rhs, level.x, level.tmp,
            aggregate_ids, level.coarse_work, level.lambda_max, level.level_id, level.coarse_solver)
        push!(levels, level)

        coarse_rows = _m(coarse_A)
        if _should_stop_coarsening(n, coarse_rows, solver, level_id)
            push!(levels, _allocate_level(coarse_A, level_id + 1, collect(1:coarse_rows), backend, solver.smoother))
            break
        end

        current_A = coarse_A
        level_id += 1
    end
    last_level = levels[end]
    last_level.coarse_solver = factorize(_csr_to_csc(last_level.A))
    hierarchy = AMGHierarchy(levels, backend, _m(A), length(_nzval(A)))
    rows_summary = join(map(level -> string(_m(level.A)), levels), " -> ")
    @info "AMG hierarchy built" mode=solver.mode levels=length(levels) rows=rows_summary coarsening=typeof(solver.coarsening) smoother=typeof(solver.smoother) backend=typeof(backend)
    hierarchy
end
