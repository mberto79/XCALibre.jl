function _cpu_vector(x::AbstractVector)
    return Array(x)
end

function _cpu_vector(x::Vector)
    return x
end

function _cpu_copyto!(dest, src)
    copyto!(dest, src)
    return dest
end

function _device_copyto!(backend::CPU, dest, src)
    copyto!(dest, src)
    return dest
end

function _device_copyto!(backend, dest, src)
    KernelAbstractions.copyto!(backend, dest, src)
    return dest
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
    hierarchy.finest_snapshot = copy(_cpu_vector(_nzval(hierarchy.host_levels[1].A)))
    hierarchy.reuse_steps = 0
    return hierarchy
end

function _amg_setup_backend(backend)
    return backend
end

function _amg_setup_matrix(A, backend)
    return A
end

function _amg_needs_cpu_apply(A)
    return false
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
    return sparse(I, J, V, _m(A), _n(A))
end

SparseArrays.findnz(A::AMGMatrixCSR) = _csr_triplets(A)
Base.Matrix(A::AMGMatrixCSR) = Matrix(_csr_to_csc(A))

function _wrap_sparse(A::SparseMatrixCSC)
    I, J, V = findnz(A)
    return SparseXCSR(sparsecsr(I, J, V, size(A, 1), size(A, 2)))
end

function _amg_matrix(A)
    rowptr = Int.(_cpu_vector(_rowptr(A)))
    colval = Int.(_cpu_vector(_colval(A)))
    nzval = copy(_cpu_vector(_nzval(A)))
    return AMGMatrixCSR(rowptr, colval, nzval, _m(A), _n(A))
end

function _amg_matrix(A::SparseMatrixCSC)
    return _amg_matrix(_wrap_sparse(A))
end

function _amg_matrix(A::Transpose{T,SparseMatrixCSC{T,Int}}) where {T}
    return _amg_matrix(sparse(A))
end

function _diag_inverse(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    diag = zeros(T, n)
    invdiag = zeros(T, n)
    for i in 1:n
        aii = one(T)
        idx = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[p] == i
                idx = p
                aii = nzval[p]
                break
            end
        end
        idx == 0 && (aii = one(T))
        diag[i] = aii
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
    return diag, invdiag
end

function _diag_inverse!(diag, invdiag, A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    @inbounds for i in 1:n
        aii = one(T)
        idx = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[p] == i
                idx = p
                aii = nzval[p]
                break
            end
        end
        idx == 0 && (aii = one(T))
        diag[i] = aii
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
    return diag, invdiag
end

function _estimate_lambda_max!(v, w, A, invdiag; iters::Int=5)
    T = eltype(invdiag)
    n = _m(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    @inbounds for i in 1:n
        v[i] = isodd(i) ? one(T) : -one(T)
    end
    vnorm = sqrt(T(n))
    @inbounds for i in 1:n
        v[i] /= vnorm
    end
    lambda = one(T)
    @inbounds for _ in 1:iters
        for i in 1:n
            wi = zero(T)
            for p in rowptr[i]:(rowptr[i + 1] - 1)
                wi += nzval[p] * v[colval[p]]
            end
            w[i] = wi * invdiag[i]
        end
        lambda = max(norm(w), eps(T))
        for i in 1:n
            v[i] = w[i] / lambda
        end
    end
    return max(lambda, one(T))
end

function _estimate_lambda_max(A, invdiag)
    T = eltype(invdiag)
    v = Vector{T}(undef, _m(A))
    w = Vector{T}(undef, _m(A))
    return _estimate_lambda_max!(v, w, A, invdiag)
end

function _empty_transfer_matrix(T)
    return AMGMatrixCSR([1], Int[], T[], 0, 0)
end

function _allocate_level(A, P, R, level_id, aggregate_ids, backend, smoother)
    n = _m(A)
    T = eltype(_nzval(A))
    diag, invdiag = _diag_inverse(A)
    rhs = KernelAbstractions.zeros(backend, T, n)
    x = KernelAbstractions.zeros(backend, T, n)
    tmp = KernelAbstractions.zeros(backend, T, n)
    lambda = _estimate_lambda_max(A, invdiag)
    has_transfer = _m(P) > 0 && _n(P) > 0 && length(_nzval(P)) > 0
    aggregate = _amg_backend_array(backend, aggregate_ids)
    is_cpu = backend isa CPU
    return AMGLevel(
        is_cpu ? A : adapt(backend, A),
        is_cpu ? P : adapt(backend, P),
        is_cpu ? R : adapt(backend, R),
        _amg_backend_array(backend, diag),
        _amg_backend_array(backend, invdiag),
        rhs,
        x,
        tmp,
        aggregate,
        lambda,
        level_id,
        has_transfer
    )
end

function _refresh_level!(level::AMGLevel, solver::AMG)
    _diag_inverse!(level.diagonal, level.inv_diagonal, level.A)
    # reuse level.rhs / level.tmp as scratch (host-side refresh only; kernels overwrite before read)
    level.lambda_max = _estimate_lambda_max!(level.rhs, level.tmp, level.A, level.inv_diagonal)
    return level
end

function _drop_coarse_matrix(A, tolerance)
    tolerance <= 0 && return A
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    row_max = zeros(T, n)
    for i in 1:n
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            row_max[i] = max(row_max[i], abs(nzval[p]))
        end
    end

    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, length(nzval))
    sizehint!(J, length(nzval))
    sizehint!(V, length(nzval))

    for i in 1:n
        diag_index = 0
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[p] == i
                diag_index = p
            end
        end
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            symmetric_scale = j <= n ? max(row_max[i], row_max[j]) : row_max[i]
            keep = j == i || abs(nzval[p]) >= tolerance * symmetric_scale
            if keep
                push!(I, i)
                push!(J, j)
                push!(V, nzval[p])
            end
        end
        if diag_index == 0
            push!(I, i)
            push!(J, i)
            push!(V, one(T))
        end
    end

    dropped = sparse(I, J, V, n, _n(A))
    dropzeros!(dropped)
    return _amg_matrix(dropped)
end

function _regalerkin_cached!(fine_level::AMGLevel, P_csc, R_csc, solver::AMG)
    coarse_A = _amg_matrix(R_csc * _csr_to_csc(fine_level.A) * P_csc)
    tolerance = _coarse_drop_tolerance(solver.coarsening, fine_level.level_id + 1)
    return _drop_coarse_matrix(coarse_A, tolerance)
end

function _regalerkin_numeric!(coarse_A, fine_level::AMGLevel, P_csc, R_csc, solver::AMG)
    updated_A = _regalerkin_cached!(fine_level, P_csc, R_csc, solver)
    if _m(coarse_A) == _m(updated_A) &&
       _n(coarse_A) == _n(updated_A) &&
       _rowptr(coarse_A) == _rowptr(updated_A) &&
       _colval(coarse_A) == _colval(updated_A)
        copyto!(_nzval(coarse_A), _nzval(updated_A))
        return coarse_A
    end
    return updated_A
end

function _refresh_coarse_cpu!(coarse_cpu::AMGCPUCoarseLevel, A)
    coarse_cpu.A = A
    coarse_cpu.Acsc = _csr_to_csc(A)
    if length(coarse_cpu.rhs) != _m(A)
        coarse_cpu.rhs = zeros(eltype(_nzval(A)), _m(A))
        coarse_cpu.x = similar(coarse_cpu.rhs)
    end
    try
        F = coarse_cpu.lu_factor
        Acsc = coarse_cpu.Acsc
        # reuse symbolic factor when pattern unchanged; fall back to full lu on first call or rebuild
        if !coarse_cpu.use_qr && F.m == size(Acsc, 1) && length(F.colptr) == length(Acsc.colptr) && length(F.rowval) == length(Acsc.rowval)
            lu!(F, Acsc)
        else
            coarse_cpu.lu_factor = lu(Acsc)
        end
        coarse_cpu.use_qr = false
    catch err
        if err isa LinearAlgebra.SingularException
            coarse_cpu.qr_factor = qr(coarse_cpu.Acsc)
            coarse_cpu.use_qr = true
        else
            rethrow(err)
        end
    end
    return coarse_cpu
end

function refresh_hierarchy!(hierarchy::AMGHierarchy, solver::AMG)
    levels = hierarchy.host_levels
    transfer = hierarchy.transfer_csc
    for level_index in 1:(length(levels) - 1)
        level = levels[level_index]
        _refresh_level!(level, solver)
        coarse_level = levels[level_index + 1]
        P_csc, R_csc = transfer[level_index]
        coarse_level.A = _regalerkin_numeric!(coarse_level.A, level, P_csc, R_csc, solver)
    end
    _refresh_level!(levels[end], solver)
    _refresh_coarse_cpu!(hierarchy.coarse_cpu, levels[end].A)
    hierarchy.nnz = length(_nzval(levels[1].A))
    hierarchy.nrows = _m(levels[1].A)
    hierarchy.operator_complexity, hierarchy.grid_complexity = _hierarchy_complexities(levels)
    hierarchy.force_rebuild = false
    _update_finest_snapshot!(hierarchy)
    return hierarchy
end

function refresh_finest_level!(hierarchy::AMGHierarchy, solver::AMG)
    _refresh_level!(hierarchy.host_levels[1], solver)
    hierarchy.force_rebuild = false
    hierarchy.reuse_steps += 1
    return hierarchy
end

function _needs_numeric_refresh(hierarchy::AMGHierarchy, A, solver::AMG)
    length(hierarchy.host_levels) <= 1 && return false
    hierarchy.reuse_steps >= solver.coarse_refresh_interval && return true
    isempty(hierarchy.finest_snapshot) && return true
    return _numeric_refresh_ratio(hierarchy.finest_snapshot, A) > solver.numeric_refresh_rtol
end

function _should_stop_coarsening(n, coarse_rows, solver::AMG, level_id)
    return coarse_rows < 2 || coarse_rows >= n || coarse_rows <= solver.min_coarse_rows || level_id >= solver.max_levels
end

function _hierarchy_complexities(levels)
    fine_rows = max(1, _m(levels[1].A))
    fine_nnz = max(1, length(_nzval(levels[1].A)))
    grid_complexity = sum(_m(level.A) for level in levels) / fine_rows
    operator_complexity = sum(length(_nzval(level.A)) for level in levels) / fine_nnz
    return float(operator_complexity), float(grid_complexity)
end

function _amg_workgroup(backend, workgroup, ndrange)
    if workgroup isa AutoTune
        return backend isa CPU ? cld(max(ndrange, 1), Threads.nthreads()) : min(max(ndrange, 1), 256)
    end
    return Int(workgroup)
end

function _sync_device_levels!(hierarchy::AMGHierarchy)
    if hierarchy.backend isa CPU
        hierarchy.levels = hierarchy.host_levels
    else
        hierarchy.levels = [adapt(hierarchy.backend, level) for level in hierarchy.host_levels]
    end
    return hierarchy
end

function _sync_device_finest_level!(hierarchy::AMGHierarchy)
    hierarchy.backend isa CPU && return hierarchy
    backend = hierarchy.backend
    dev = hierarchy.levels[1]
    host = hierarchy.host_levels[1]
    _device_copyto!(backend, dev.A.nzval, host.A.nzval)
    _device_copyto!(backend, dev.diagonal, host.diagonal)
    _device_copyto!(backend, dev.inv_diagonal, host.inv_diagonal)
    dev.lambda_max = host.lambda_max
    return hierarchy
end

function _sync_device_levels_numeric!(hierarchy::AMGHierarchy)
    hierarchy.backend isa CPU && return hierarchy
    backend = hierarchy.backend
    for k in eachindex(hierarchy.levels)
        dev = hierarchy.levels[k]
        host = hierarchy.host_levels[k]
        if length(dev.A.nzval) != length(host.A.nzval)
            hierarchy.levels[k] = adapt(backend, host)
            continue
        end
        _device_copyto!(backend, dev.A.nzval, host.A.nzval)
        _device_copyto!(backend, dev.diagonal, host.diagonal)
        _device_copyto!(backend, dev.inv_diagonal, host.inv_diagonal)
        dev.lambda_max = host.lambda_max
    end
    return hierarchy
end

function setup_hierarchy(A, solver::AMG, backend; log_diagnostics=true)
    return setup_hierarchy(A, solver, backend, AutoTune(); log_diagnostics=log_diagnostics)
end

function setup_hierarchy(A, solver::AMG, backend, workgroup; log_diagnostics=true)
    host_levels = nothing
    transfer_csc = Any[]
    current_A = _amg_matrix(A)
    current_candidates = _initial_candidates(solver.coarsening)
    level_id = 1
    T = eltype(_nzval(current_A))
    while true
        aggregate_ids, P_csc, current_candidates = build_prolongation(current_A, solver.coarsening, current_candidates, level_id)
        if isnothing(P_csc)
            P = _empty_transfer_matrix(T)
            R = _empty_transfer_matrix(T)
            level = _allocate_level(current_A, P, R, level_id, aggregate_ids, CPU(), solver.smoother)
            if isnothing(host_levels)
                host_levels = typeof(level)[]
            end
            push!(host_levels, level)
            break
        end

        R_csc_lazy = transpose(P_csc)
        R_csc = sparse(R_csc_lazy)
        coarse_A = _amg_matrix(R_csc * _csr_to_csc(current_A) * P_csc)
        coarse_A = _drop_coarse_matrix(coarse_A, _coarse_drop_tolerance(solver.coarsening, level_id + 1))
        P = _amg_matrix(P_csc)
        R = _amg_matrix(R_csc)
        level = _allocate_level(current_A, P, R, level_id, aggregate_ids, CPU(), solver.smoother)
        if isnothing(host_levels)
            host_levels = typeof(level)[]
        end
        push!(host_levels, level)
        push!(transfer_csc, (P_csc, R_csc))

        coarse_rows = _m(coarse_A)
        if coarse_rows <= solver.max_coarse_rows || _should_stop_coarsening(_m(current_A), coarse_rows, solver, level_id)
            P_last = _empty_transfer_matrix(T)
            R_last = _empty_transfer_matrix(T)
            push!(host_levels, _allocate_level(coarse_A, P_last, R_last, level_id + 1, collect(1:coarse_rows), CPU(), solver.smoother))
            break
        end

        current_A = coarse_A
        level_id += 1
    end

    coarse_cpu = _empty_cpu_coarse_level(T)
    _refresh_coarse_cpu!(coarse_cpu, host_levels[end].A)
    rowptr_pattern, colval_pattern = _pattern_signature(A)
    pattern_hash = hash(colval_pattern, hash(rowptr_pattern))
    is_symmetric = solver.mode == :cg ? _is_symmetric(host_levels[1].A) : true
    operator_complexity, grid_complexity = _hierarchy_complexities(host_levels)
    device_levels = backend isa CPU ? host_levels : [adapt(backend, level) for level in host_levels]
    hierarchy = AMGHierarchy(
        device_levels,
        host_levels,
        coarse_cpu,
        backend,
        _amg_workgroup(backend, workgroup, _m(host_levels[1].A)),
        _m(host_levels[1].A),
        length(_nzval(host_levels[1].A)),
        rowptr_pattern,
        colval_pattern,
        Ref{Any}(_rowptr(A)),
        Ref{Any}(_colval(A)),
        pattern_hash,
        transfer_csc,
        is_symmetric,
        operator_complexity,
        grid_complexity,
        0.0,
        false,
        0,
        copy(_cpu_vector(_nzval(host_levels[1].A)))
    )
    rows_summary = join(map(level -> string(_m(level.A)), host_levels), " -> ")
    if log_diagnostics
        @info "AMG hierarchy built" mode=solver.mode levels=length(host_levels) rows=rows_summary
        @info "AMG hierarchy diagnostics" cycle=solver.cycle coarsening=typeof(solver.coarsening) smoother=typeof(solver.smoother) backend=typeof(backend) operator_complexity=operator_complexity grid_complexity=grid_complexity
    end
    return hierarchy
end
