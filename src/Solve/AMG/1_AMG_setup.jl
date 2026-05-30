# Galerkin cache entries scale as K^2 * nnz (K = max P entries per row).
# Beyond this nnz, the cache exceeds ~100 MB and causes OOM on large meshes.
const _GALERKIN_CACHE_MAX_NNZ = 400_000

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

function _diag_index(A)
    rowptr = _rowptr(A)
    colval = _colval(A)
    n = _m(A)
    diag_index = zeros(Int, n)
    @inbounds for i in 1:n
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[p] == i
                diag_index[i] = p
                break
            end
        end
    end
    return diag_index
end

function _diag_inverse!(diag, invdiag, A, diag_index)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(nzval)
    @inbounds for i in 1:n
        idx = diag_index[i]
        aii = idx == 0 ? one(T) : nzval[idx]
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
    return max(lambda, _scaled_gershgorin_bound(A, invdiag), one(T))
end

function _scaled_gershgorin_bound(A, invdiag)
    rowptr = _rowptr(A)
    nzval = _nzval(A)
    n = _m(A)
    T = eltype(invdiag)
    bound = one(T)
    @inbounds for i in 1:n
        row_sum = zero(T)
        dinv = abs(invdiag[i])
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            row_sum += abs(nzval[p]) * dinv
        end
        bound = max(bound, row_sum)
    end
    return bound
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
    diag_index = _diag_index(A)
    rhs = KernelAbstractions.zeros(backend, T, n)
    x = KernelAbstractions.zeros(backend, T, n)
    tmp = KernelAbstractions.zeros(backend, T, n)
    direction = KernelAbstractions.zeros(backend, T, n)
    coarse_tmp = KernelAbstractions.zeros(backend, T, n)
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
        _amg_backend_array(backend, diag_index),
        rhs,
        x,
        tmp,
        direction,
        coarse_tmp,
        aggregate,
        lambda,
        level_id,
        has_transfer
    )
end

function _refresh_level!(level::AMGLevel, solver::AMG)
    _diag_inverse!(level.diagonal, level.inv_diagonal, level.A, level.diagonal_index)
    # reuse level.rhs / level.tmp as scratch (host-side refresh only; kernels overwrite before read)
    level.lambda_max = _estimate_lambda_max!(level.rhs, level.tmp, level.A, level.inv_diagonal)
    return level
end

function _regalerkin_cached!(fine_level::AMGLevel, P_csc, R_csc)
    return _amg_matrix(R_csc * _csr_to_csc(fine_level.A) * P_csc)
end

@inline function _csr_find_index(rowptr, colval, row::Integer, col::Integer)
    lo = rowptr[row]
    hi = rowptr[row + 1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        value = colval[mid]
        if value == col
            return mid
        elseif value < col
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

function _galerkin_index_type(max_index::Integer)
    return max_index <= typemax(Int32) ? Int32 : Int
end

function _build_galerkin_cache(fine_A, P, coarse_A)
    index_type = _galerkin_index_type(max(length(_nzval(fine_A)), length(_nzval(coarse_A))))
    return _build_galerkin_cache(index_type, fine_A, P, coarse_A)
end

function _build_galerkin_cache(::Type{I}, fine_A, P, coarse_A) where {I<:Integer}
    Arowptr = _rowptr(fine_A)
    Acolval = _colval(fine_A)
    Prowptr = _rowptr(P)
    Pcolval = _colval(P)
    Pnzval = _nzval(P)
    Crowptr = _rowptr(coarse_A)
    Ccolval = _colval(coarse_A)
    n = _m(fine_A)
    T = promote_type(eltype(_nzval(fine_A)), eltype(Pnzval))

    targets = I[]
    fine_indices = I[]
    weights = T[]
    n_hint = min(20 * length(_nzval(fine_A)), 10_000_000)
    sizehint!(targets, n_hint)
    sizehint!(fine_indices, n_hint)
    sizehint!(weights, n_hint)

    @inbounds for i in 1:n
        pi_start = Prowptr[i]
        pi_stop = Prowptr[i + 1] - 1
        pi_start <= pi_stop || continue
        for ap in Arowptr[i]:(Arowptr[i + 1] - 1)
            j = Acolval[ap]
            1 <= j <= n || continue
            pj_start = Prowptr[j]
            pj_stop = Prowptr[j + 1] - 1
            pj_start <= pj_stop || continue
            for rp in pi_start:pi_stop
                ci = Pcolval[rp]
                wi = Pnzval[rp]
                for pp in pj_start:pj_stop
                    cj = Pcolval[pp]
                    target = _csr_find_index(Crowptr, Ccolval, ci, cj)
                    target == 0 && continue
                    push!(targets, I(target))
                    push!(fine_indices, I(ap))
                    push!(weights, T(wi * Pnzval[pp]))
                end
            end
        end
    end
    return AMGGalerkinCache(targets, fine_indices, weights)
end

function _regalerkin_numeric!(coarse_A, fine_level::AMGLevel, cache::AMGGalerkinCache)
    coarse_nzval = _nzval(coarse_A)
    fine_nzval = _nzval(fine_level.A)
    fill!(coarse_nzval, zero(eltype(coarse_nzval)))
    targets = cache.targets
    fine_indices = cache.fine_indices
    weights = cache.weights
    @inbounds for k in eachindex(targets)
        coarse_nzval[Int(targets[k])] += weights[k] * fine_nzval[Int(fine_indices[k])]
    end
    return coarse_A
end

function _regalerkin_numeric!(coarse_A, fine_level::AMGLevel, P_csc, R_csc)
    updated_A = _regalerkin_cached!(fine_level, P_csc, R_csc)
    if _m(coarse_A) == _m(updated_A) &&
       _n(coarse_A) == _n(updated_A) &&
       _rowptr(coarse_A) == _rowptr(updated_A) &&
       _colval(coarse_A) == _colval(updated_A)
        copyto!(_nzval(coarse_A), _nzval(updated_A))
        return coarse_A
    end
    return updated_A
end

# SYMBOLIC/NUMERIC SPLIT RAP (PETSc/Ginkgo-style, no allocation on refresh)

function _build_ra_pattern(R, A)
    I = Int32
    R_rowptr = _rowptr(R); R_colval = _colval(R)
    A_rowptr = _rowptr(A); A_colval = _colval(A)
    n_coarse = _m(R); n_fine = _m(A)
    marker = zeros(Int, n_fine)

    nnz_per_row = zeros(Int, n_coarse)
    @inbounds for r in 1:n_coarse
        for rp in R_rowptr[r]:(R_rowptr[r+1]-1)
            i = Int(R_colval[rp])
            for ap in A_rowptr[i]:(A_rowptr[i+1]-1)
                j = Int(A_colval[ap])
                if marker[j] != r
                    marker[j] = r
                    nnz_per_row[r] += 1
                end
            end
        end
    end

    ra_rowptr = Vector{Int}(undef, n_coarse + 1)
    ra_rowptr[1] = 1
    for r in 1:n_coarse
        ra_rowptr[r+1] = ra_rowptr[r] + nnz_per_row[r]
    end
    total_nnz = ra_rowptr[n_coarse+1] - 1
    ra_colval = Vector{I}(undef, total_nnz)

    fill!(marker, 0)
    pos = copy(ra_rowptr[1:n_coarse])
    @inbounds for r in 1:n_coarse
        for rp in R_rowptr[r]:(R_rowptr[r+1]-1)
            i = Int(R_colval[rp])
            for ap in A_rowptr[i]:(A_rowptr[i+1]-1)
                j = Int(A_colval[ap])
                if marker[j] != r
                    marker[j] = r
                    ra_colval[pos[r]] = I(j)
                    pos[r] += 1
                end
            end
        end
        lo = ra_rowptr[r]; hi = ra_rowptr[r+1] - 1
        lo < hi && sort!(@view ra_colval[lo:hi])
    end
    return ra_rowptr, ra_colval
end

function _build_rap_plan_cpu(R, A, P)
    T = eltype(_nzval(A))
    n_fine = _m(A)
    n_coarse_out = _n(P)
    ra_rowptr, ra_colval = _build_ra_pattern(R, A)
    ra_nzval      = zeros(T, length(ra_colval))
    workspace_ra  = zeros(T, n_fine)
    workspace_rap = zeros(T, n_coarse_out)
    flag_ra  = zeros(Int, n_fine)
    flag_rap = zeros(Int, n_coarse_out)
    return AMGRAPPlanCPU(ra_rowptr, ra_colval, ra_nzval,
                         workspace_ra, workspace_rap, flag_ra, flag_rap)
end

# SPA-based refresh: O(nnz(A)×K) with O(1) scatter — no binary search.
function _refresh_rap_numeric!(coarse_A, fine_level::AMGLevel, plan::AMGRAPPlanCPU)
    R = fine_level.R; A = fine_level.A; P = fine_level.P
    R_rowptr = _rowptr(R); R_colval = _colval(R); R_nzval = _nzval(R)
    A_rowptr = _rowptr(A); A_colval = _colval(A); A_nzval = _nzval(A)
    P_rowptr = _rowptr(P); P_colval = _colval(P); P_nzval = _nzval(P)
    C_rowptr = _rowptr(coarse_A); C_colval = _colval(coarse_A); C_nzval = _nzval(coarse_A)
    ra_rowptr = plan.ra_rowptr; ra_colval = plan.ra_colval; ra_nzval = plan.ra_nzval
    wra = plan.workspace_ra; wrap = plan.workspace_rap
    fra = plan.flag_ra;      frap = plan.flag_rap
    n_coarse = length(ra_rowptr) - 1

    # Pass 1: fill ra_nzval using SPA over fine columns
    @inbounds for r in 1:n_coarse
        for rp in R_rowptr[r]:(R_rowptr[r+1]-1)
            i = Int(R_colval[rp]); Rri = R_nzval[rp]
            for ap in A_rowptr[i]:(A_rowptr[i+1]-1)
                j = Int(A_colval[ap])
                fra[j] != r && (fra[j] = r; wra[j] = zero(eltype(wra)))
                wra[j] += Rri * A_nzval[ap]
            end
        end
        for p in ra_rowptr[r]:(ra_rowptr[r+1]-1)
            j = Int(ra_colval[p])
            ra_nzval[p] = fra[j] == r ? wra[j] : zero(eltype(ra_nzval))
        end
    end

    # Pass 2: fill coarse_A using ra_nzval and SPA over coarse columns
    @inbounds for r in 1:n_coarse
        for rp in ra_rowptr[r]:(ra_rowptr[r+1]-1)
            j = Int(ra_colval[rp]); RAij = ra_nzval[rp]
            iszero(RAij) && continue
            for pp in P_rowptr[j]:(P_rowptr[j+1]-1)
                c = Int(P_colval[pp])
                frap[c] != r && (frap[c] = r; wrap[c] = zero(eltype(wrap)))
                wrap[c] += RAij * P_nzval[pp]
            end
        end
        for p in C_rowptr[r]:(C_rowptr[r+1]-1)
            c = Int(C_colval[p])
            C_nzval[p] = frap[c] == r ? wrap[c] : zero(eltype(C_nzval))
        end
    end
    return coarse_A
end

# Hook: backend extensions override to build device-resident plans (e.g. cuSPARSE SpGEMMreuse)
_amg_finalize_transfer_plans(_backend, transfer_csc, _host_levels, _device_levels) = transfer_csc

function _csr_pattern_matches(A, B)
    _m(A) == _m(B) || return false
    _n(A) == _n(B) || return false
    length(_nzval(A)) == length(_nzval(B)) || return false
    _rowptr(A) == _rowptr(B) || return false
    _colval(A) == _colval(B) || return false
    return true
end

@inline function _csc_find_index(colptr, rowval, row::Integer, col::Integer)
    lo = colptr[col]
    hi = colptr[col + 1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        value = rowval[mid]
        if value == row
            return mid
        elseif value < row
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

function _build_csr_to_csc_nzval_index(A, Acsc::SparseMatrixCSC)
    rowptr = _rowptr(A)
    colval = _colval(A)
    index = Vector{Int}(undef, length(_nzval(A)))
    @inbounds for row in 1:_m(A)
        for p in rowptr[row]:(rowptr[row + 1] - 1)
            q = _csc_find_index(Acsc.colptr, Acsc.rowval, row, colval[p])
            q == 0 && return Int[]
            index[p] = q
        end
    end
    return index
end

function _update_csc_nzval_from_csr!(Acsc::SparseMatrixCSC, A, csc_nzval_index)
    nzval = _nzval(A)
    length(csc_nzval_index) == length(nzval) || return false
    length(Acsc.nzval) >= length(nzval) || return false
    @inbounds for p in eachindex(nzval)
        Acsc.nzval[csc_nzval_index[p]] = nzval[p]
    end
    return true
end

function _refresh_coarse_csc!(coarse_cpu::AMGCPUCoarseLevel, A)
    pattern_matches =
        _csr_pattern_matches(coarse_cpu.A, A) &&
        size(coarse_cpu.Acsc, 1) == _m(A) &&
        size(coarse_cpu.Acsc, 2) == _n(A) &&
        length(coarse_cpu.csc_nzval_index) == length(_nzval(A))

    if pattern_matches && _update_csc_nzval_from_csr!(coarse_cpu.Acsc, A, coarse_cpu.csc_nzval_index)
        coarse_cpu.A = A
        return coarse_cpu.Acsc
    end

    coarse_cpu.A = A
    coarse_cpu.Acsc = _csr_to_csc(A)
    coarse_cpu.csc_nzval_index = _build_csr_to_csc_nzval_index(A, coarse_cpu.Acsc)
    return coarse_cpu.Acsc
end

function _refresh_coarse_cpu!(coarse_cpu::AMGCPUCoarseLevel, A)
    Acsc = _refresh_coarse_csc!(coarse_cpu, A)
    if length(coarse_cpu.rhs) != _m(A)
        coarse_cpu.rhs = zeros(eltype(_nzval(A)), _m(A))
        coarse_cpu.x = similar(coarse_cpu.rhs)
    end
    try
        F = coarse_cpu.lu_factor
        # reuse symbolic factor when pattern unchanged; fall back to full lu if SuiteSparse rejects reuse
        if !coarse_cpu.use_qr && F.m == size(Acsc, 1) && length(F.colptr) == length(Acsc.colptr) && length(F.rowval) == length(Acsc.rowval)
            try
                lu!(F, Acsc)
            catch err
                err isa ArgumentError || rethrow(err)
                coarse_cpu.lu_factor = lu(Acsc)
            end
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
        xfer = level_index <= length(transfer) ? transfer[level_index] : nothing
        isnothing(xfer) || _refresh_coarse_level!(coarse_level.A, level, xfer)
    end
    _refresh_level!(levels[end], solver)
    _refresh_coarse_cpu!(hierarchy.coarse_cpu, levels[end].A)
    hierarchy.nnz = length(_nzval(levels[1].A))
    hierarchy.nrows = _m(levels[1].A)
    hierarchy.operator_complexity, hierarchy.grid_complexity = _hierarchy_complexities(levels)
    return hierarchy
end

# CPU/default path: symbolic/numeric split, no allocation
function _refresh_coarse_level!(coarse_A, fine_level::AMGLevel, plan::AMGRAPPlanCPU)
    _refresh_rap_numeric!(coarse_A, fine_level, plan)
end

# Legacy fallback (kept for any non-plan entries; should not occur after setup)
function _refresh_coarse_level!(coarse_A, fine_level::AMGLevel, xfer::Tuple)
    P_csc, R_csc = xfer
    result = _regalerkin_numeric!(coarse_A, fine_level, P_csc, R_csc)
    coarse_A === result || copyto!(_nzval(coarse_A), _nzval(result))
end

function refresh_finest_level!(hierarchy::AMGHierarchy, solver::AMG)
    _refresh_level!(hierarchy.host_levels[1], solver)
    return hierarchy
end

function _should_stop_coarsening(n, coarse_rows, solver::AMG, level_id)
    poor_reduction = coarse_rows > solver.max_coarse_rows && coarse_rows >= 0.85 * n
    return coarse_rows < 2 || coarse_rows >= n || coarse_rows <= solver.min_coarse_rows || poor_reduction || level_id >= solver.max_levels
end

function _hierarchy_complexities(levels)
    fine_rows = max(1, _m(levels[1].A))
    fine_nnz = max(1, length(_nzval(levels[1].A)))
    grid_complexity = sum(_m(level.A) for level in levels) / fine_rows
    operator_complexity = sum(length(_nzval(level.A)) for level in levels) / fine_nnz
    return float(operator_complexity), float(grid_complexity)
end

function _hierarchy_level_summary(levels)
    lines = ["level rows nnz"]
    for (level_index, level) in pairs(levels)
        push!(lines, string(level_index, " ", _m(level.A), " ", length(_nzval(level.A))))
    end
    return join(lines, "\n")
end

function _coarse_solve_name(backend, solver::AMG, coarse_cpu::AMGCPUCoarseLevel, A, is_symmetric::Bool)
    !(backend isa CPU) && _is_diagonal_matrix(A) && return "device_diagonal"
    if !(backend isa CPU) &&
       lowercase(get(ENV, "XCALIBRE_AMG_DEVICE_COARSE_SOLVE", "")) == "cg" &&
       solver.mode isa Cg &&
       is_symmetric &&
       _m(A) == _n(A)
        return "device_cg_experimental"
    end
    return coarse_cpu.use_qr ? "sparse_qr" : "sparse_lu"
end

function _amg_workgroup(backend, workgroup, ndrange)
    if workgroup isa AutoTune
        return backend isa CPU ? cld(max(ndrange, 1), Threads.nthreads()) : min(max(ndrange, 1), 256)
    end
    return Int(workgroup)
end

# Hook to specialise device level storage (e.g. wrap finest operators as CuSparseMatrixCSR for cuSPARSE SpMV).
# Default is a no-op; backend extensions may return a new levels container.
_amg_finalize_device_levels(backend, levels) = levels

function _sync_device_levels!(hierarchy::AMGHierarchy)
    if hierarchy.backend isa CPU
        hierarchy.levels = hierarchy.host_levels
    else
        device_levels = [adapt(hierarchy.backend, level) for level in hierarchy.host_levels]
        hierarchy.levels = _amg_finalize_device_levels(hierarchy.backend, device_levels)
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
        # Pattern is guaranteed fixed by AMGRAPPlanCPU; sync only values, not structure.
        # For k==1 (finest): A may be a CuSparseMatrixCSR wrapping the same nzval — check size.
        if length(_nzval(dev.A)) != length(host.A.nzval)
            hierarchy.levels[k] = adapt(backend, host)
            continue
        end
        _device_copyto!(backend, _nzval(dev.A), host.A.nzval)
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
    _validate_amg_smoother_backend(backend, solver.smoother)
    host_levels = nothing
    transfer_csc = Any[]
    galerkin_caches = Any[]
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
        log_diagnostics && @info "AMG build: level=$level_id fine rows=$(_m(current_A)) nnz=$(length(_nzval(current_A)))"
        coarse_A = _amg_matrix(R_csc * _csr_to_csc(current_A) * P_csc)
        log_diagnostics && @info "AMG build: level=$level_id coarse rows=$(_m(coarse_A)) nnz=$(length(_nzval(coarse_A)))"
        P = _amg_matrix(P_csc)
        R = _amg_matrix(R_csc)
        plan = _build_rap_plan_cpu(R, current_A, P)
        log_diagnostics && @info "AMG build: level=$level_id RA nnz=$(length(plan.ra_colval))"
        push!(galerkin_caches, nothing)
        P_csc = nothing; R_csc = nothing  # CSC matrices no longer needed
        push!(transfer_csc, plan)
        GC.gc(false)
        level = _allocate_level(current_A, P, R, level_id, aggregate_ids, CPU(), solver.smoother)
        if isnothing(host_levels)
            host_levels = typeof(level)[]
        end
        push!(host_levels, level)

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
    is_symmetric = solver.mode isa Cg ? _is_symmetric(host_levels[1].A) : true
    operator_complexity, grid_complexity = _hierarchy_complexities(host_levels)
    device_levels = backend isa CPU ? host_levels : Any[adapt(backend, level) for level in host_levels]
    device_levels = backend isa CPU ? device_levels : _amg_finalize_device_levels(backend, device_levels)
    # Allow backend extensions to replace CPU plans with device-resident plans (e.g. cuSPARSE)
    transfer_csc = _amg_finalize_transfer_plans(backend, transfer_csc, host_levels, device_levels)
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
        galerkin_caches,
        is_symmetric,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0,
        operator_complexity,
        grid_complexity,
        0.0
    )
    rows_summary = join(map(level -> string(_m(level.A)), host_levels), " -> ")
    if log_diagnostics
        coarse_solver = _coarse_solve_name(backend, solver, coarse_cpu, host_levels[end].A, is_symmetric)
        coarse_host_transfer = !(backend isa CPU) && coarse_solver in ("sparse_lu", "sparse_qr")
        @info "AMG hierarchy built" mode=_amg_mode_name(solver.mode) levels=length(host_levels) rows=rows_summary
        @info "AMG hierarchy levels\n$(_hierarchy_level_summary(host_levels))"
        @info "AMG hierarchy diagnostics" cycle=solver.cycle coarsening=typeof(solver.coarsening) smoother=typeof(solver.smoother) backend=typeof(backend) operator_complexity=operator_complexity grid_complexity=grid_complexity
        @info "AMG coarse solve" rows=_m(host_levels[end].A) nnz=length(_nzval(host_levels[end].A)) solver=coarse_solver device_transfer=coarse_host_transfer
    end
    return hierarchy
end
