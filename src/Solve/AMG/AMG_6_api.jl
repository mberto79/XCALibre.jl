export update!

# ─── Workspace constructor ────────────────────────────────────────────────────

"""
    _workspace(amg::AMG, A, b) → AMGWorkspace

Build a fully-typed AMGWorkspace from the matrix `A` and RHS vector `b`.
The concrete `MultigridLevel` element type is determined here so that
`ws.levels` is a `Vector{LType}` — no `Any`, no dynamic dispatch in the cycle.
`A` is not stored or copied; it is only used to infer the matrix and element types.
"""
function _workspace(amg::AMG, ::AT, b::AbstractVector{Tv}) where {AT, Tv}
    LType = MultigridLevel{Tv, AT, Union{Nothing, AT}, typeof(b),
                            LevelExtras{Tv, SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}}}
    x = similar(b)
    fill!(x, zero(Tv))
    return AMGWorkspace(LType[], x, amg, false, 0)
end

# ─── Full hierarchy setup ─────────────────────────────────────────────────────

"""
    amg_setup!(ws, A_device, backend, workgroup)

Build the complete multigrid hierarchy from `A_device`. Coarsening and all
Galerkin products are performed on the CPU; matrices and work vectors are
then transferred to the target device. Calling this again replaces the entire
hierarchy.
"""
function amg_setup!(ws::AMGWorkspace{LType}, A_device, backend, workgroup) where {LType}
    opts = ws.opts
    Tv   = eltype(ws.x)
    Ti   = eltype(_rowptr(A_device))

    mk_vec(n) = KernelAbstractions.zeros(backend, Tv, n)

    # ── Phase 1: build hierarchy on CPU ────────────────────────────────────────
    A_cpu     = _to_cpu_csr(A_device)
    A_cpus    = [A_cpu]       # level matrices (CPU)
    P_cpus    = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]  # inter-level P
    R_cpus    = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]  # inter-level R
    rhos      = Tv[]           # spectral radius per level

    # Fine level spectral radius
    D_fine = _extract_dinv_cpu(A_cpu)
    push!(rhos, Tv(estimate_spectral_radius(A_cpu, D_fine)))
    _fine_diag_max = maximum(inv, D_fine; init=zero(Tv))

    A_cur = A_cpu
    for _ in 2:opts.max_levels
        n_cur = size(A_cur, 1)
        n_cur <= opts.coarsest_size && break

        agg, nagg = amg_coarsen(A_cur, opts.strength, opts.coarsening)
        nagg >= n_cur  && break
        nagg > 0.8 * n_cur && break

        D_cur = _extract_dinv_cpu(A_cur)
        ρ_cur = estimate_spectral_radius(A_cur, D_cur)

        ω_P   = min(4.0 / (3.0 * max(ρ_cur, eps(Float64))), 4.0/3.0)

        P_tent = build_tentative_P(n_cur, nagg, agg)
        P_cpu  = smooth_prolongation(A_cur, P_tent, ω_P)

        any(!isfinite, P_cpu.nzval) && break

        R_cpu  = build_restriction(P_cpu)
        Ac_cpu = galerkin_product(R_cpu, A_cur, P_cpu)

        nc = size(Ac_cpu, 1)
        (nc == 0 || any(!isfinite, Ac_cpu.nzval)) && break

        # Stop if coarse diagonal decays to near machine epsilon
        Ac_diag_min = minimum(
            v for i in 1:nc
              for (j, v) in zip(Ac_cpu.colval[Ac_cpu.rowptr[i]:Ac_cpu.rowptr[i+1]-1],
                                Ac_cpu.nzval[Ac_cpu.rowptr[i]:Ac_cpu.rowptr[i+1]-1])
              if j == i;
            init = Tv(Inf))
        Ac_diag_min < sqrt(eps(Float64)) * _fine_diag_max && break

        D_c = _extract_dinv_cpu(Ac_cpu)
        ρ_c = estimate_spectral_radius(Ac_cpu, D_c)

        push!(P_cpus,  P_cpu)
        push!(R_cpus,  R_cpu)
        push!(A_cpus,  Ac_cpu)
        push!(rhos,    Tv(ρ_c))

        A_cur = Ac_cpu
    end

    n_levels = length(A_cpus)

    # ── Phase 2: build device matrices and MultigridLevel objects ─────────────
    # Coarsest-level LU (dense, on CPU) — only if small enough
    n_coarsest = size(A_cpus[end], 1)
    lu_f, lu_rhs = n_coarsest <= opts.coarsest_size ?
                       _build_lu(A_cpus[end], Tv) :
                       (nothing, Tv[])

    # Build all levels (same LType for the whole vector → typed storage)
    levels = LType[]
    sizehint!(levels, n_levels)

    for i in 1:n_levels
        n  = size(A_cpus[i], 1)
        A_dev = (i == 1) ? A_device : _csr_to_device(A_cpus[i], backend, Tv, Ti)

        # Transfer operators (only for non-coarsest levels)
        P_dev = (i <= length(P_cpus)) ? _csr_to_device(P_cpus[i], backend, Tv, Ti) : nothing
        R_dev = (i <= length(R_cpus)) ? _csr_to_device(R_cpus[i], backend, Tv, Ti) : nothing

        Dinv = mk_vec(n)
        amg_build_Dinv!(Dinv, A_dev, backend, workgroup)

        extras = LevelExtras{Tv, SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}}()
        extras.P_cpu = (i <= length(P_cpus)) ? P_cpus[i] : nothing
        extras.R_cpu = (i <= length(R_cpus)) ? R_cpus[i] : nothing
        extras.rho   = rhos[i]
        if i == n_levels
            extras.lu_factor = lu_f
            extras.lu_rhs    = lu_rhs
        end

        # Construct with LType directly so PType = Union{Nothing, AType} is enforced
        # regardless of whether P_dev is nothing or a concrete sparse matrix.
        level = LType(A_dev, P_dev, R_dev,
                      Dinv, mk_vec(n), mk_vec(n), mk_vec(n), mk_vec(n),
                      extras)
        push!(levels, level)
    end

    # ── Commit to workspace ────────────────────────────────────────────────────
    ws.levels       = levels
    ws.x            = levels[1].x
    ws.setup_valid  = true
    ws.setup_count += 1
    nothing
end

# ─── Numerical update (reuse hierarchy structure) ─────────────────────────────

"""
    update!(ws, A_device, backend, workgroup)

Refresh numerical values in the AMG hierarchy after the fine-level matrix
`A_device` has changed (same sparsity pattern, new coefficients). Coarse
matrices are recomputed via Galerkin products; work vectors are reused. If
the hierarchy has not yet been built, calls `amg_setup!` instead.
"""
function update!(ws::AMGWorkspace, A_device, backend, workgroup)
    if !ws.setup_valid || isempty(ws.levels)
        amg_setup!(ws, A_device, backend, workgroup)
        return
    end

    # Level 1 matrix is A_device directly (mutated in-place by the outer solver)
    amg_build_Dinv!(ws.levels[1].Dinv, ws.levels[1].A, backend, workgroup)

    A_cur = _to_cpu_csr(A_device)
    for lvl in 1:(length(ws.levels) - 1)
        L  = ws.levels[lvl]
        Lc = ws.levels[lvl + 1]

        P_cpu = L.extras.P_cpu
        R_cpu = L.extras.R_cpu
        Ac_new = galerkin_product(R_cpu, A_cur, P_cpu)

        _copy_nzval_to_device!(Lc.A, Ac_new.nzval, backend)
        amg_build_Dinv!(Lc.Dinv, Lc.A, backend, workgroup)

        A_cur = Ac_new
    end

    # Rebuild coarsest-level LU (only if it was built during setup)
    if !isnothing(ws.levels[end].extras.lu_factor)
        lu_f, lu_rhs = _build_lu(A_cur, eltype(ws.x))
        ws.levels[end].extras.lu_factor = lu_f
        ws.levels[end].extras.lu_rhs    = lu_rhs
    end
    nothing
end

# ─── solve_system! dispatch ────────────────────────────────────────────────────

"""
    solve_system!(phiEqn, setup, result, component, config)

AMG-specific override dispatched when `phiEqn.solver isa AMGWorkspace`.
`ws.levels` is a fully-typed `Vector{LType}`, so all cycle kernel launches are
type-stable with no boxing.
"""
function solve_system!(
    phiEqn::ModelEquation{T,M,E,S,P}, setup, result, component, config
) where {T, M, E, S<:AMGWorkspace, P}
    (; itmax, atol, rtol) = setup
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; values) = result

    ws = phiEqn.solver
    A  = _A(phiEqn)
    b  = _b(phiEqn, component)

    # Build or refresh hierarchy
    update!(ws, A, backend, workgroup)

    L1 = ws.levels[1]   # concrete LType — fully type-stable

    # Initialise finest level from current field values
    amg_copy!(L1.x, values, backend, workgroup)
    amg_copy!(L1.b, b,      backend, workgroup)

    # Initial residual norm
    amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
    r0 = amg_norm(L1.r)
    r0 = ifelse(r0 > eps(r0), r0, one(r0))

    # Multigrid cycle iterations — levels::Vector{LType} is typed, no dynamic dispatch
    for _ in 1:itmax
        run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)

        amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
        res_norm = amg_norm(L1.r)
        (res_norm < atol || res_norm / r0 < rtol) && break
    end

    # Copy solution back to the field values array
    amg_copy!(values, L1.x, backend, workgroup)

    return residual(phiEqn, component, config)
end

# ─── CSR ↔ device helpers ─────────────────────────────────────────────────────

_to_cpu_csr(A::SparseXCSR) = parent(A)
_to_cpu_csr(A::SparseMatricesCSR.SparseMatrixCSR) = A

function _to_cpu_csr(A)
    nzval  = Vector(Array(_nzval(A)))
    colval = Vector(Array(_colval(A)))
    rowptr = Vector{Int}(Array(_rowptr(A)))
    m, n   = size(A)
    nnz    = length(nzval)
    rows   = zeros(Int, nnz)
    for i in 1:m
        for nzi in rowptr[i]:(rowptr[i+1]-1)
            rows[nzi] = i
        end
    end
    return sparsecsr(rows, Vector{Int}(colval), nzval, m, n)
end

function _csr_to_device(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti},
                         backend, ::Type{Tv2}, ::Type{Ti2}) where {Bi,Tv,Ti,Tv2,Ti2}
    if backend isa CPU
        return SparseXCSR(A)
    else
        m, n = size(A)
        rowptr = Vector{Int32}(A.rowptr)
        colval = Vector{Int32}(A.colval)
        nzval  = Vector{Tv2}(A.nzval)
        return _build_sparse_device(backend, rowptr, colval, nzval, m, n)
    end
end

function _copy_nzval_to_device!(A::SparseXCSR, nzval_cpu::AbstractVector, ::CPU)
    copyto!(parent(A).nzval, nzval_cpu)
end

function _copy_nzval_to_device!(A, nzval_cpu::AbstractVector, backend)
    KernelAbstractions.copyto!(backend, _nzval(A), nzval_cpu)
end

# ─── Scalar helpers ───────────────────────────────────────────────────────────

function _extract_dinv_cpu(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    n    = size(A, 1)
    Dinv = ones(Tv, n)
    for i in 1:n
        for nzi in A.rowptr[i]:(A.rowptr[i+1]-1)
            if A.colval[nzi] == i
                d = A.nzval[nzi]
                Dinv[i] = d != zero(Tv) ? one(Tv)/d : one(Tv)
                break
            end
        end
    end
    return Dinv
end

# Build dense LU for the coarsest level; returns (lu_factor, lu_rhs).
function _build_lu(A_cpu::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti}, ::Type{Tv2}) where {Bi,Tv,Ti,Tv2}
    n = size(A_cpu, 1)
    Adense = zeros(Tv2, n, n)
    for i in 1:n
        for nzi in A_cpu.rowptr[i]:(A_cpu.rowptr[i+1]-1)
            Adense[i, A_cpu.colval[nzi]] = Tv2(A_cpu.nzval[nzi])
        end
    end
    return lu!(Adense), zeros(Tv2, n)
end

# Coarse solve: dense LU if available, else extra Jacobi sweeps.
function amg_coarse_solve!(level::MultigridLevel, backend)
    ex = level.extras
    if !isnothing(ex.lu_factor)
        copyto!(ex.lu_rhs, level.b)
        ldiv!(ex.lu_factor, ex.lu_rhs)
        KernelAbstractions.copyto!(backend, level.x, ex.lu_rhs)
    else
        # Fall back to many Jacobi sweeps when LU is too expensive
        workgroup = 1024
        amg_smooth!(level, 50, 2/3, backend, workgroup)
    end
end
