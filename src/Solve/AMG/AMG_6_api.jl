export update!

# ─── Dinv build dispatch: standard (diagonal) vs l1 (row-norm) ───────────────
# L1Jacobi requires the l1 kernel (full row sum); all other smoothers use the
# fast diagonal-pointer path. Called at setup and every update! iteration.

_amg_build_smoother_dinv!(::AbstractSmoother, Dinv, A, diag_ptr, backend, workgroup) =
    amg_build_Dinv!(Dinv, A, diag_ptr, backend, workgroup)

_amg_build_smoother_dinv!(::L1Jacobi, Dinv, A, diag_ptr, backend, workgroup) =
    amg_build_l1_Dinv!(Dinv, A, backend, workgroup)

# ─── Workspace constructor ────────────────────────────────────────────────────

"""
    _workspace(amg::AMG, A, b) → AMGWorkspace

Build a fully-typed AMGWorkspace from the matrix `A` and RHS vector `b`.

The concrete `MultigridLevel` element type is fixed here so that `ws.levels` is
a `Vector{LType}` — no `Any`, no dynamic dispatch in the cycle hot path.

The multigrid hierarchy is **not** built here. `A` may have all-zero values at
initialisation time (before the first equation assembly), which would cause the
Galerkin diagonal check to break the coarsening loop after a single level.
The hierarchy is instead built lazily on the first `update!` call, when `A`
carries the actual assembled coefficients.
"""
function _workspace(amg::AMG, A::AT, b::AbstractVector{Tv}) where {AT, Tv}
    LType = MultigridLevel{Tv, AT, Union{Nothing, AT}, typeof(b),
                            LevelExtras{Tv, SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}}}
    x     = similar(b); fill!(x,     zero(Tv))
    x_pcg = similar(b); fill!(x_pcg, zero(Tv))
    p_cg  = similar(b); fill!(p_cg,  zero(Tv))
    return AMGWorkspace(LType[], x, amg, false, 0, 0, x_pcg, p_cg)
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
    AP_cpus   = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]  # A*P intermediates (reused in update!)
    rhos      = Tv[]           # spectral radius per level

    # ω_P = 4/(3ρ) where ρ = ρ(D⁻¹A); for FVM Laplacian ρ ≈ 2 → ω_P ≈ 2/3.
    use_jacobi = opts.smoother isa JacobiSmoother || opts.smoother isa L1Jacobi

    D_fine    = _extract_dinv_cpu(A_cpu)
    # Gershgorin bound: deterministic, tight for FVM M-matrices.
    ρ_fine    = Tv(_gershgorin_rho(A_cpu, D_fine))
    _rho_fine = use_jacobi ? one(Tv) : ρ_fine
    push!(rhos, _rho_fine)
    _fine_diag_max = maximum(inv, D_fine; init=zero(Tv))

    A_cur = A_cpu
    for _ in 2:opts.max_levels
        n_cur = size(A_cur, 1)
        n_cur <= opts.coarsest_size && break

        agg, nagg = amg_coarsen(A_cur, opts.strength, opts.coarsening)
        nagg >= n_cur && break
        # If primary coarsening stagnated (< 30% reduction), fall back to
        # pairwise aggregation which guarantees ≥ 2× reduction.
        if nagg > (n_cur * 7) ÷ 10
            agg_p, nagg_p = _coarsen_pairwise(A_cur)
            if nagg_p < nagg
                agg, nagg = agg_p, nagg_p
            end
        end
        # Stop only if even the pairwise fallback couldn't reduce size ≥ 10%.
        nagg > 0.9 * n_cur && break

        # Unsmoothed aggregation: use P̂ directly (1 nnz per row).
        # Prolongation smoothing inflates nnz(P) from 1 to ~stencil_width per row,
        # multiplying op_complexity by ~3-4× with no benefit for near-isotropic FVM
        # M-matrices (all connections equally strong, smooth error already piecewise-const).
        P_tent = build_tentative_P(n_cur, nagg, agg)
        P_cpu  = P_tent

        any(!isfinite, P_cpu.nzval) && break

        R_cpu  = build_restriction(P_cpu)
        AP_cpu, Ac_cpu = galerkin_product(R_cpu, A_cur, P_cpu)

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
        ρ_c = use_jacobi ? one(Tv) : Tv(_gershgorin_rho(Ac_cpu, D_c))

        push!(P_cpus,  P_cpu)
        push!(R_cpus,  R_cpu)
        push!(AP_cpus, AP_cpu)
        push!(A_cpus,  Ac_cpu)
        push!(rhos,    ρ_c)

        A_cur = Ac_cpu
    end

    n_levels = length(A_cpus)

    # Diagnostic: show hierarchy sizes and operator complexity.
    # Operator complexity = Σ nnz(Aₗ) / nnz(A₀); target < 2.5 for 3-D meshes.
    # Values > 2.5 indicate fat coarse levels — each V-cycle costs proportionally more.
    level_sizes = [size(A_cpus[i], 1) for i in 1:n_levels]
    nnz_fine    = length(A_cpus[1].nzval)
    nnz_total   = sum(length(A_cpus[i].nzval) for i in 1:n_levels)
    op_cmplx    = round(nnz_total / nnz_fine; digits=2)
    @info "AMG hierarchy ($(opts.coarsening), strength=$(opts.strength)): $(level_sizes) — op_complexity=$(op_cmplx) — direct solve at coarsest: $(level_sizes[end] <= opts.coarsest_size)"

    # ── Phase 2: build device matrices, Galerkin plans, and MultigridLevel objects ─
    n_coarsest  = size(A_cpus[end], 1)
    lu_dense_c  = nothing
    lu_f        = nothing
    lu_rhs_init = Tv[]
    if n_coarsest <= opts.coarsest_size
        lu_dense_c = zeros(Tv, n_coarsest, n_coarsest)
        _fill_dense_from_sparse!(lu_dense_c, A_cpus[end])
        lu_f        = lu!(lu_dense_c)
        lu_rhs_init = zeros(Tv, n_coarsest)
    end

    levels = LType[]
    sizehint!(levels, n_levels)

    for i in 1:n_levels
        n  = size(A_cpus[i], 1)
        A_dev = (i == 1) ? A_device : _csr_to_device(A_cpus[i], backend, Tv, Ti)

        P_dev = (i <= length(P_cpus)) ? _csr_to_device(P_cpus[i], backend, Tv, Ti) : nothing
        R_dev = (i <= length(R_cpus)) ? _csr_to_device(R_cpus[i], backend, Tv, Ti) : nothing

        Dinv = mk_vec(n)

        diag_ptr_cpu = _build_diag_ptr_cpu(A_cpus[i])
        diag_ptr_dev = KernelAbstractions.zeros(backend, Int32, n)
        KernelAbstractions.copyto!(backend, diag_ptr_dev, diag_ptr_cpu)

        _amg_build_smoother_dinv!(opts.smoother, Dinv, A_dev, diag_ptr_dev, backend, workgroup)

        extras = LevelExtras{Tv, SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}}()
        extras.P_cpu  = (i <= length(P_cpus)) ? P_cpus[i] : nothing
        extras.R_cpu  = (i <= length(R_cpus)) ? R_cpus[i] : nothing
        extras.rho    = rhos[i]
        extras.diag_ptr = diag_ptr_dev

        # Pre-allocate two-step SpGEMM scratch for each non-coarsest level:
        #   Step 1: T = A·P  (intermediate, same sparsity each update)
        #   Step 2: Ac = R·T (result copied to device Lc.A.nzval)
        # AP_cpus[i] was already computed by galerkin_product in Phase 1 — reuse it
        # to avoid a redundant symbolic+numeric SpGEMM here.
        if i < n_levels
            nc     = size(P_cpus[i], 2)
            extras.AP_cpu = AP_cpus[i]               # reuse from Phase 1 (no redundant SpGEMM)
            extras.Ac_cpu = deepcopy(A_cpus[i+1])   # writable Ac scratch
            extras.A_cpu  = deepcopy(A_cpus[i])      # writable fine-A mirror for nzval download
            # Compact accumulator: tmps is max_nnz_per_row × nthreads (L1-resident).
            # col_to_local is nc × nthreads Int32 scatter map (filled/cleared per row at runtime).
            nrows_AP = size(AP_cpus[i], 1)
            nrows_Ac = size(A_cpus[i+1], 1)
            max_nnz  = max(
                maximum(AP_cpus[i].rowptr[r+1] - AP_cpus[i].rowptr[r] for r in 1:nrows_AP; init=0),
                maximum(A_cpus[i+1].rowptr[r+1] - A_cpus[i+1].rowptr[r] for r in 1:nrows_Ac; init=0),
            )
            extras.cpu_tmps     = zeros(Tv,     max_nnz, Threads.nthreads())
            extras.col_to_local = zeros(Int32,  nc,      Threads.nthreads())
        end

        # Coarsest level: store CPU copy of A for LU rebuild in update!()
        if i == n_levels
            extras.A_cpu     = A_cpus[end]
            extras.lu_dense  = lu_dense_c
            extras.lu_factor = lu_f
            extras.lu_rhs    = lu_rhs_init
        end

        level = LType(A_dev, P_dev, R_dev,
                      Dinv, mk_vec(n), mk_vec(n), mk_vec(n), mk_vec(n),
                      extras)
        push!(levels, level)
    end

    # ── Commit to workspace ────────────────────────────────────────────────────
    ws.levels        = levels
    ws.x             = levels[1].x
    ws.setup_valid   = true
    ws.setup_count  += 1
    # Set to 1 so the first update! after setup skips the Galerkin refresh:
    # the hierarchy was just built from the current matrix, so recomputing immediately
    # is redundant. For update_freq=1 the check (count-1)%1==0 still runs every call.
    ws.update_count  = 1
    nothing
end

# ─── Numerical update (reuse hierarchy structure) ─────────────────────────────

"""
    update!(ws, A_device, backend, workgroup)

Refresh numerical values in the AMG hierarchy after the fine-level matrix
`A_device` has changed (same sparsity pattern, new coefficients). Work vectors
are reused; no allocations. Falls back to `amg_setup!` if the hierarchy has
not yet been built.

## What is always updated (every call)

- **Fine-level D⁻¹**: extracted from the current `A_device` nzval using a
  pre-computed diagonal-position pointer (`extras.diag_ptr`). Single kernel
  launch, no row scan, no branch divergence.

## What is lazily updated (every `update_freq` calls)

Controlled by `ws.opts.update_freq` (set via `AMG(; update_freq=N)`).

- **Galerkin products** `Ac = R·A·P` for each coarse level: device-resident
  kernel reads the current `A.nzval` and overwrites `Ac.nzval` using the
  pre-built index plan. No CPU↔device transfers.
- **Coarse D⁻¹** for each level: rebuilt from the updated `Ac.nzval`.
- **Coarsest-level LU**: the tiny coarsest `nzval` (~50 floats) is downloaded
  to CPU, the pre-allocated dense buffer is filled, and `lu!` is called
  in-place.

### Lazy refresh schedule

`update_count` is incremented on every call and reset to 1 after `amg_setup!`
(not 0), so the first `update!` after setup skips the Galerkin refresh —
the hierarchy was just computed from the same matrix, so recomputing it
immediately is redundant.
The lazy part runs when `(update_count - 1) % update_freq == 0`, i.e. on
calls 1+update_freq, 1+2·update_freq, … after setup.
Exception: when `update_freq == 1` the condition fires every call.

### Why lazy refresh is safe

In SIMPLE/PISO the outer loop is itself an iterative correction: a slightly
stale coarse correction causes the AMG to need a few more V-cycles per outer
iteration, but the outer iteration still converges. As the solution approaches
steady state the matrix changes very slowly, so the approximation cost becomes
negligible. The fine-level D⁻¹ (which directly controls smoother accuracy) is
never skipped.
"""
function update!(ws::AMGWorkspace, A_device, backend, workgroup)
    # Build the hierarchy on the first call (A has real coefficients by then).
    # Also handles manually-constructed workspaces without levels (e.g. in tests).
    if !ws.setup_valid || isempty(ws.levels)
        amg_setup!(ws, A_device, backend, workgroup)
        return
    end

    ws.update_count += 1

    # Fine-level D⁻¹ always updated; diagonal changes every outer iteration.
    L1 = ws.levels[1]
    _amg_build_smoother_dinv!(ws.opts.smoother, L1.Dinv, L1.A, L1.extras.diag_ptr, backend, workgroup)

    # Lazy Galerkin refresh: coarse levels skipped when (update_count-1) % update_freq ≠ 0.
    update_freq = ws.opts.update_freq
    (ws.update_count - 1) % update_freq != 0 && return

    for lvl in 1:(length(ws.levels) - 1)
        L  = ws.levels[lvl]
        Lc = ws.levels[lvl + 1]
        _galerkin_update!(L, Lc, backend, workgroup)
        _amg_build_smoother_dinv!(ws.opts.smoother, Lc.Dinv, Lc.A, Lc.extras.diag_ptr, backend, workgroup)
    end

    # Refresh coarsest-level LU from updated nzval. check=false avoids exception alloc.
    if !isnothing(ws.levels[end].extras.lu_factor)
        ex_c     = ws.levels[end].extras
        nzval_c, _, _ = get_sparse_fields(ws.levels[end].A)
        copyto!(ex_c.A_cpu.nzval, nzval_c)   # device → CPU (tiny)
        fill!(ex_c.lu_dense, zero(eltype(ws.x)))
        _fill_dense_from_sparse!(ex_c.lu_dense, ex_c.A_cpu)
        ex_c.lu_factor = lu!(ex_c.lu_dense; check=false)
    end
    nothing
end

# ─── Preconditioned Conjugate Gradient (PCG) solve ────────────────────────────
# One V-cycle per iteration as preconditioner M ≈ A⁻¹; O(√κ) vs O(κ) for plain Richardson.

function _amg_pcg_solve!(ws::AMGWorkspace, b, values, itmax, atol, rtol,
                           backend, workgroup)
    L1  = ws.levels[1]
    Tv  = eltype(ws.x_pcg)
    x   = ws.x_pcg
    p   = ws.p_cg

    # ── Initialise x from current field ──────────────────────────────────────
    amg_copy!(x, values, backend, workgroup)

    # ── r ← b - A x ──────────────────────────────────────────────────────────
    amg_residual!(L1.r, L1.A, x, b, backend, workgroup)
    r0 = amg_norm(L1.r)
    r0 = ifelse(r0 > eps(r0), r0, one(r0))

    # Early exit: initial guess already satisfies tolerance (common with warm start)
    if r0 < atol
        amg_copy!(values, x, backend, workgroup)
        return
    end

    # ── z ← M⁻¹ r  (one V-cycle: L1.b = r, L1.x = 0 → L1.x = z) ───────────
    amg_copy!(L1.b, L1.r, backend, workgroup)
    amg_zero!(L1.x, backend, workgroup)
    run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)
    # CRITICAL: vcycle! calls amg_residual!(L.r, ...) at the fine level, which
    # overwrites L1.r with the V-cycle's internal residual (≈ 0 at cycle end).
    # Restore the CG residual from L1.b, which the V-cycle never modifies.
    amg_copy!(L1.r, L1.b, backend, workgroup)

    # ── p ← z ;  rz = rᵀ z ───────────────────────────────────────────────────
    amg_copy!(p, L1.x, backend, workgroup)
    rz = dot(L1.r, p)

    # ── PCG iterations ────────────────────────────────────────────────────────
    for k in 1:itmax

        # Ap ← A p  (stored in L1.tmp; consumed before the next V-cycle)
        amg_spmv!(L1.tmp, L1.A, p, backend, workgroup)

        # α = (rᵀ z) / (pᵀ A p)
        pAp   = dot(p, L1.tmp)
        alpha = rz / pAp

        # x ← x + α p  ;  r ← r − α Ap
        amg_axpy!(x, p,      alpha,  backend, workgroup)
        amg_axpy!(L1.r, L1.tmp, -alpha, backend, workgroup)

        # Convergence check — exit before spending another V-cycle
        res_norm = amg_norm(L1.r)
        (res_norm < atol || res_norm / r0 < rtol) && break
        k == itmax && break

        # z ← M⁻¹ r  (V-cycle overwrites L1.r — restore CG residual from L1.b)
        amg_copy!(L1.b, L1.r, backend, workgroup)
        amg_zero!(L1.x, backend, workgroup)
        run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)
        amg_copy!(L1.r, L1.b, backend, workgroup)   # restore r after V-cycle

        # β = (rᵀ z_new) / (rᵀ z_old) ;  z_new is in L1.x
        rz_new = dot(L1.r, L1.x)
        beta   = rz_new / rz
        rz     = rz_new

        # p ← z + β p
        amg_axpby!(p, L1.x, one(Tv), beta, backend, workgroup)
    end

    # ── Copy PCG solution → field values ──────────────────────────────────────
    amg_copy!(values, x, backend, workgroup)
end

# ─── solve_system! dispatch ────────────────────────────────────────────────────

"""
    solve_system!(phiEqn, setup, result, component, config)

AMG-specific override dispatched when `phiEqn.solver isa AMGWorkspace`.
`ws.levels` is a fully-typed `Vector{LType}`, so all cycle kernel launches are
type-stable with no boxing.

Dispatches to PCG (`opts.krylov === :cg`, default) or plain Richardson
(`opts.krylov === :none`) based on the `krylov` field of the `AMG` options.
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

    if ws.opts.krylov === :cg
        _amg_pcg_solve!(ws, b, values, itmax, atol, rtol, backend, workgroup)
    else
        # ── Standalone Richardson: plain V-cycle loop ─────────────────────────
        L1 = ws.levels[1]

        amg_copy!(L1.x, values, backend, workgroup)
        amg_copy!(L1.b, b,      backend, workgroup)

        amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
        r0 = amg_norm(L1.r)
        r0 = ifelse(r0 > eps(r0), r0, one(r0))

        # Early exit: initial guess already satisfies tolerance.
        if r0 >= atol
            check_freq = 5   # check at k=1 then every check_freq cycles
            for k in 1:itmax
                run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)
                if k == 1 || k % check_freq == 0 || k == itmax
                    amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
                    res_norm = amg_norm(L1.r)
                    (res_norm < atol || res_norm / r0 < rtol) && break
                end
            end
        end

        amg_copy!(values, L1.x, backend, workgroup)
    end

    return residual(phiEqn, component, config)
end

# ─── Galerkin update: Ac = R·A·P ─────────────────────────────────────────────
# CPU path: download A.nzval, run two in-place CPU SpGEMMs, upload Ac.nzval.
# All scratch buffers are pre-allocated in amg_setup!; no allocation here.

function _galerkin_update!(L::MultigridLevel, Lc::MultigridLevel,
                            backend::KernelAbstractions.CPU, workgroup)
    ex = L.extras
    nzval_dev, _, _ = get_sparse_fields(L.A)
    copyto!(ex.A_cpu.nzval, nzval_dev)
    _spgemm_nzval!(ex.AP_cpu, ex.A_cpu, ex.P_cpu, ex.cpu_tmps, ex.col_to_local)
    _spgemm_nzval!(ex.Ac_cpu, ex.R_cpu, ex.AP_cpu, ex.cpu_tmps, ex.col_to_local)
    nzval_dev_c, _, _ = get_sparse_fields(Lc.A)
    KernelAbstractions.copyto!(backend, nzval_dev_c, ex.Ac_cpu.nzval)
end

# GPU path: KA kernel scatters contributions entirely on-device — no PCIe transfer.
function _galerkin_update!(L::MultigridLevel, Lc::MultigridLevel, backend, workgroup)
    amg_rap_update!(Lc, L, backend, workgroup)
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

function _fill_dense_from_sparse!(Adense::Matrix{Tv},
                                   A::SparseMatricesCSR.SparseMatrixCSR) where {Tv}
    n = size(A, 1)
    @inbounds for i in 1:n
        for nzi in A.rowptr[i]:(A.rowptr[i+1]-1)
            Adense[i, A.colval[nzi]] = Tv(A.nzval[nzi])
        end
    end
    return Adense
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
