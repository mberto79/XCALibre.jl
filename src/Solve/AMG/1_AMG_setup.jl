function _cpu_vector(x::AbstractVector)
    Array(x)
end

function _cpu_vector(x::Vector)
    x
end

function _pattern_signature(A)
    return Int.(_cpu_vector(_rowptr(A))), Int.(_cpu_vector(_colval(A)))
end

function _numeric_refresh_ratio(snapshot, A)
    current = _nzval(A)
    length(snapshot) == length(current) || return Inf
    max_delta = 0.0
    max_scale = 0.0
    @inbounds for i in eachindex(current)
        ai = current[i]
        scale = max(abs(snapshot[i]), abs(ai))
        delta = abs(ai - snapshot[i])
        max_delta = max(max_delta, delta)
        max_scale = max(max_scale, scale)
    end
    max_scale <= eps(Float64) && return max_delta > 0 ? Inf : 0.0
    return max_delta / max_scale
end

function _update_finest_snapshot!(hierarchy::AMGHierarchy)
    hierarchy.finest_snapshot = copy(_cpu_vector(_nzval(hierarchy.levels[1].A)))
    hierarchy.reuse_steps = 0
    return hierarchy
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

function _diag_inverse_and_l1(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    d = zeros(T, n)
    dinv = zeros(T, n)
    l1inv = zeros(T, n)
    for i in 1:n
        aii = one(T)
        l1row = zero(T)
        idx = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            if j == i
                idx = p
                aii = nzval[p]
            else
                l1row += abs(nzval[p])
            end
        end
        idx == 0 && (aii = one(T))
        d[i] = aii
        dinv[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
        l1diag = max(abs(aii), l1row)
        l1inv[i] = l1diag > eps(T) ? inv(l1diag) : one(T)
    end
    return d, dinv, l1inv
end

function _diag_inverse_and_l1!(diag, invdiag, l1inv, A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(nzval)
    for i in eachindex(diag)
        aii = one(T)
        l1row = zero(T)
        idx = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            if j == i
                idx = p
                aii = nzval[p]
            else
                l1row += abs(nzval[p])
            end
        end
        idx == 0 && (aii = one(T))
        diag[i] = aii
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
        l1diag = max(abs(aii), l1row)
        l1inv[i] = l1diag > eps(T) ? inv(l1diag) : one(T)
    end
    return diag, invdiag, l1inv
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
    diag, invdiag, l1inv = _diag_inverse_and_l1(A)
    rhs = KernelAbstractions.zeros(backend, T, n)
    x = KernelAbstractions.zeros(backend, T, n)
    tmp = KernelAbstractions.zeros(backend, T, n)
    coarse_work = KernelAbstractions.zeros(backend, T, max(1, n))
    λ = _estimate_lambda_max(A, invdiag, smoother)
    return AMGLevel(A, nothing, nothing, diag, invdiag, l1inv, rhs, x, tmp, aggregate_ids, coarse_work, λ, level_id, nothing)
end

function _refresh_level!(level::AMGLevel, solver::AMG)
    _diag_inverse_and_l1!(level.diagonal, level.inv_diagonal, level.l1_inv_diagonal, level.A)
    level.lambda_max = _estimate_lambda_max(level.A, level.inv_diagonal, solver.smoother)
    return level
end

function _regalerkin!(fine_level::AMGLevel)
    coarse_A = _wrap_sparse(fine_level.R * _csr_to_csc(fine_level.A) * fine_level.P)
    return coarse_A
end

function _build_coarse_solver(A)
    Acsc = _csr_to_csc(A)
    try
        return factorize(Acsc)
    catch err
        if err isa LinearAlgebra.SingularException
            return qr(Acsc)
        end
        rethrow(err)
    end
end

function refresh_hierarchy!(hierarchy::AMGHierarchy, solver::AMG)
    levels = hierarchy.levels
    for level_index in 1:(length(levels) - 1)
        level = levels[level_index]
        _refresh_level!(level, solver)
        coarse_level = levels[level_index + 1]
        coarse_level.A = _regalerkin!(level)
    end
    _refresh_level!(levels[end], solver)
    levels[end].coarse_solver = _build_coarse_solver(levels[end].A)
    hierarchy.nnz = length(_nzval(levels[1].A))
    hierarchy.nrows = _m(levels[1].A)
    hierarchy.operator_complexity, hierarchy.grid_complexity = _hierarchy_complexities(levels)
    hierarchy.force_rebuild = false
    _update_finest_snapshot!(hierarchy)
    return hierarchy
end

function refresh_finest_level!(hierarchy::AMGHierarchy, solver::AMG)
    _refresh_level!(hierarchy.levels[1], solver)
    hierarchy.force_rebuild = false
    hierarchy.reuse_steps += 1
    return hierarchy
end

function _needs_numeric_refresh(hierarchy::AMGHierarchy, A, solver::AMG)
    length(hierarchy.levels) <= 1 && return false
    hierarchy.reuse_steps >= solver.coarse_refresh_interval && return true
    isnothing(hierarchy.finest_snapshot) && return true
    return _numeric_refresh_ratio(hierarchy.finest_snapshot, A) > solver.numeric_refresh_rtol
end

function _should_stop_coarsening(n, coarse_rows, solver::AMG, level_id)
    coarse_rows < 2 || coarse_rows >= n || coarse_rows <= solver.min_coarse_rows || level_id >= solver.max_levels
end

function _hierarchy_complexities(levels)
    fine_rows = max(1, _m(levels[1].A))
    fine_nnz = max(1, length(_nzval(levels[1].A)))
    grid_complexity = sum(_m(level.A) for level in levels) / fine_rows
    operator_complexity = sum(length(_nzval(level.A)) for level in levels) / fine_nnz
    return float(operator_complexity), float(grid_complexity)
end

function setup_hierarchy(A, solver::AMG, backend; log_diagnostics=true)
    levels = AMGLevel[]
    current_A = A
    current_candidates = solver.coarsening isa SmoothAggregation ? solver.coarsening.near_nullspace : nothing
    level_id = 1
    while true
        n = _m(current_A)
        aggregate_ids, P, current_candidates = build_prolongation(current_A, solver.coarsening, current_candidates)
        level = _allocate_level(current_A, level_id, aggregate_ids, backend, solver.smoother)

        if isnothing(P)
            push!(levels, level)
            break
        end

        R = transpose(P)
        coarse_A = _wrap_sparse(R * _csr_to_csc(current_A) * P)
        level = AMGLevel(current_A, P, R, level.diagonal, level.inv_diagonal, level.l1_inv_diagonal, level.rhs, level.x, level.tmp,
            aggregate_ids, level.coarse_work, level.lambda_max, level.level_id, level.coarse_solver)
        push!(levels, level)

        coarse_rows = _m(coarse_A)
        if coarse_rows <= solver.max_coarse_rows
            push!(levels, _allocate_level(coarse_A, level_id + 1, collect(1:coarse_rows), backend, solver.smoother))
            break
        end
        if _should_stop_coarsening(n, coarse_rows, solver, level_id)
            push!(levels, _allocate_level(coarse_A, level_id + 1, collect(1:coarse_rows), backend, solver.smoother))
            break
        end

        current_A = coarse_A
        level_id += 1
    end
    last_level = levels[end]
    last_level.coarse_solver = _build_coarse_solver(last_level.A)
    rowptr_pattern, colval_pattern = solver.assume_fixed_pattern ? (Int[], Int[]) : _pattern_signature(A)
    is_symmetric = solver.mode == :cg ? _is_symmetric(A) : true
    operator_complexity, grid_complexity = _hierarchy_complexities(levels)
    finest_snapshot = copy(_cpu_vector(_nzval(levels[1].A)))
    hierarchy = AMGHierarchy(levels, backend, _m(A), length(_nzval(A)), rowptr_pattern, colval_pattern, is_symmetric,
        operator_complexity, grid_complexity, 0.0, false, 0, finest_snapshot, nothing, nothing, nothing)
    rows_summary = join(map(level -> string(_m(level.A)), levels), " -> ")
    if log_diagnostics
        @info "AMG hierarchy built" mode=solver.mode levels=length(levels) rows=rows_summary
        @info "AMG hierarchy diagnostics" cycle=solver.cycle coarsening=typeof(solver.coarsening) smoother=typeof(solver.smoother) backend=typeof(backend) operator_complexity=operator_complexity grid_complexity=grid_complexity
    end
    hierarchy
end
