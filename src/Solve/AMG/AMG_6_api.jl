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

Two-tier mixed-precision layout:
- Fine level type `LFType`: Float64 for A/x/b/r/tmp/Dinv; Float32 for P/R (boundary SpMVs).
- Coarse level type `LCType`: Float32 for all fields.

The multigrid hierarchy is **not** built here — hierarchy construction is deferred to the
first `update!` call when `A` carries assembled coefficients.
"""
function _workspace(amg::AMG, A::AT, b::AbstractVector{Tv}) where {AT, Tv}
    Tc        = amg.coarse_float        # user-configured coarse float type (default Float32)
    AT_c      = _tc_sparse_type(AT)     # coarse device sparse type (Float32)
    Vec_f     = typeof(b)
    Vec_c     = _tc_vec_type(Vec_f)     # coarse device vector type (Float32)
    # Verify the coarse_float setting is consistent with the GPU type mapping.
    # _tc_sparse_type/_tc_vec_type currently only support Float32; update those methods
    # if a different coarse_float is needed.
    @assert eltype(Vec_c) === Tc "coarse_float=$(Tc) does not match _tc_vec_type result $(eltype(Vec_c)); add GPU type-mapping methods for the desired coarse type"
    CpuSpT_f  = SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}
    CpuSpT_c  = SparseMatricesCSR.SparseMatrixCSR{1, Tc, Int}

    # Fine level: Float64 A; Float32 P/R; boundary buffers typed as Vec_c
    LFType = MultigridLevel{Tv,  AT,   Union{Nothing, AT_c}, Vec_f, LevelExtras{Tv, Vec_c,  CpuSpT_f}}
    # Coarse levels: all Float32; TcVec=Nothing (no boundary buffers needed)
    LCType = MultigridLevel{Tc, AT_c,  Union{Nothing, AT_c}, Vec_c, LevelExtras{Tc, Nothing, CpuSpT_c}}

    x     = similar(b); fill!(x,     zero(Tv))
    x_pcg = similar(b); fill!(x_pcg, zero(Tv))
    p_cg  = similar(b); fill!(p_cg,  zero(Tv))
    return AMGWorkspace{LFType, LCType, Vec_f, typeof(amg)}(
        nothing, LCType[], x, amg, false, 0, 0, x_pcg, p_cg)
end

# ─── Full hierarchy setup ─────────────────────────────────────────────────────

"""
    amg_setup!(ws, A_device, backend, workgroup)

Build the complete mixed-precision multigrid hierarchy from `A_device`.

Phase 1: CPU coarsening in Float64 (precision-safe for aggregation decisions).
Phase 2: Upload to device — fine level in Float64 with Float32 P/R, coarse
         levels entirely in Float32. Calling this again replaces the entire
         hierarchy.
"""
function amg_setup!(ws::AMGWorkspace{LFType, LCType}, A_device, backend, workgroup) where {LFType, LCType}
    opts = ws.opts
    Tv   = eltype(ws.x)          # fine float type (Float64)
    Tc   = opts.coarse_float     # coarse float type (Float32 by default)
    Ti   = eltype(_rowptr(A_device))

    mk_vec_f(n) = KernelAbstractions.zeros(backend, Tv, n)
    mk_vec_c(n) = KernelAbstractions.zeros(backend, Tc, n)
    CpuSpT_f = SparseMatricesCSR.SparseMatrixCSR{1, Tv, Int}
    CpuSpT_c = SparseMatricesCSR.SparseMatrixCSR{1, Tc, Int}

    # ── Phase 1: build hierarchy on CPU (all Float64) ─────────────────────────
    A_cpu     = _to_cpu_csr(A_device)
    A_cpus    = [A_cpu]
    P_cpus    = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]
    R_cpus    = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]
    AP_cpus   = SparseMatricesCSR.SparseMatrixCSR{1,Tv,Int}[]
    rhos      = Tv[]

    use_jacobi = opts.smoother isa JacobiSmoother || opts.smoother isa L1Jacobi

    D_fine    = _extract_dinv_cpu(A_cpu)
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
        if nagg > (n_cur * 7) ÷ 10
            agg_p, nagg_p = _coarsen_pairwise(A_cur)
            if nagg_p < nagg
                agg, nagg = agg_p, nagg_p
            end
        end
        nagg > 0.9 * n_cur && break

        P_tent = build_tentative_P(n_cur, nagg, agg)
        P_cpu  = P_tent
        any(!isfinite, P_cpu.nzval) && break

        R_cpu  = build_restriction(P_cpu)
        AP_cpu, Ac_cpu = galerkin_product(R_cpu, A_cur, P_cpu)

        nc = size(Ac_cpu, 1)
        (nc == 0 || any(!isfinite, Ac_cpu.nzval)) && break

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

    level_sizes = [size(A_cpus[i], 1) for i in 1:n_levels]
    nnz_fine    = length(A_cpus[1].nzval)
    nnz_total   = sum(length(A_cpus[i].nzval) for i in 1:n_levels)
    op_cmplx    = round(nnz_total / nnz_fine; digits=2)
    @info "AMG hierarchy ($(opts.coarsening), strength=$(opts.strength)): $(level_sizes) — op_complexity=$(op_cmplx) — direct solve at coarsest: $(level_sizes[end] <= opts.coarsest_size)"

    # ── Phase 2a: coarsest-level LU (Float32 unless n_levels==1) ─────────────
    n_coarsest  = size(A_cpus[end], 1)
    lu_dense_c  = nothing
    lu_f        = nothing
    lu_rhs_init_c = Tc[]   # Float32 rhs for coarse LU
    lu_dense_f  = nothing  # Float64 LU only for single-level case
    lu_f_fine   = nothing
    lu_rhs_init_f = Tv[]

    if n_coarsest <= opts.coarsest_size
        if n_levels == 1
            # Fine IS the coarsest: use Float64 LU
            lu_dense_f    = zeros(Tv, n_coarsest, n_coarsest)
            _fill_dense_from_sparse!(lu_dense_f, A_cpus[end])
            lu_f_fine     = lu!(lu_dense_f)
            lu_rhs_init_f = zeros(Tv, n_coarsest)
        else
            # Coarsest is a Float32 coarse level
            lu_dense_c    = zeros(Tc, n_coarsest, n_coarsest)
            _fill_dense_from_sparse!(lu_dense_c, A_cpus[end])
            lu_f          = lu!(lu_dense_c)
            lu_rhs_init_c = zeros(Tc, n_coarsest)
        end
    end

    # ── Phase 2b: build fine level (Float64 A, Float32 P/R) ──────────────────
    n1 = size(A_cpus[1], 1)

    # Fine P/R as Float32 device matrices (bandwidth-efficient boundary SpMVs)
    P_fine_dev = (n_levels > 1) ? _csr_to_device(P_cpus[1], backend, Tc, Ti) : nothing
    R_fine_dev = (n_levels > 1) ? _csr_to_device(R_cpus[1], backend, Tc, Ti) : nothing

    Dinv_f = mk_vec_f(n1)
    diag_ptr_f_cpu = _build_diag_ptr_cpu(A_cpus[1])
    diag_ptr_f_dev = KernelAbstractions.zeros(backend, Int32, n1)
    KernelAbstractions.copyto!(backend, diag_ptr_f_dev, diag_ptr_f_cpu)
    _amg_build_smoother_dinv!(opts.smoother, Dinv_f, A_device, diag_ptr_f_dev, backend, workgroup)

    Vec_c    = _tc_vec_type(typeof(ws.x))   # same computation as in _workspace
    extras_f = LevelExtras{Tv, Vec_c, CpuSpT_f}()
    extras_f.rho      = rhos[1]
    extras_f.diag_ptr = diag_ptr_f_dev

    if n_levels > 1
        # Galerkin scratch for fine level (Float64 CPU intermediates)
        nc1 = size(P_cpus[1], 2)
        extras_f.P_cpu  = P_cpus[1]
        extras_f.R_cpu  = R_cpus[1]
        extras_f.AP_cpu = AP_cpus[1]
        extras_f.Ac_cpu = deepcopy(A_cpus[2])
        extras_f.A_cpu  = deepcopy(A_cpus[1])
        nrows_AP1 = size(AP_cpus[1], 1)
        nrows_Ac1 = size(A_cpus[2], 1)
        max_nnz1  = max(
            maximum(AP_cpus[1].rowptr[r+1] - AP_cpus[1].rowptr[r] for r in 1:nrows_AP1; init=0),
            maximum(A_cpus[2].rowptr[r+1]  - A_cpus[2].rowptr[r]  for r in 1:nrows_Ac1; init=0),
        )
        extras_f.cpu_tmps     = zeros(Tv,    max_nnz1, Threads.nthreads())
        extras_f.col_to_local = zeros(Int32, nc1,      Threads.nthreads())
        # Float32 boundary buffers for the fine↔coarse precision cast
        extras_f.r_Tc   = mk_vec_c(n1)
        extras_f.tmp_Tc = mk_vec_c(n1)
    else
        # Single-level: fine = coarsest; store LU in fine extras
        extras_f.A_cpu     = A_cpus[1]
        extras_f.lu_dense  = lu_dense_f
        extras_f.lu_factor = lu_f_fine
        extras_f.lu_rhs    = lu_rhs_init_f
    end

    fine_level = LFType(A_device, P_fine_dev, R_fine_dev,
                        Dinv_f, mk_vec_f(n1), mk_vec_f(n1), mk_vec_f(n1), mk_vec_f(n1),
                        extras_f)

    # ── Phase 2c: build coarse levels (Float32) ───────────────────────────────
    coarse_levels = LCType[]
    sizehint!(coarse_levels, n_levels - 1)

    for i in 2:n_levels
        n_c = size(A_cpus[i], 1)
        A_dev_c = _csr_to_device(A_cpus[i], backend, Tc, Ti)

        # P/R: Float32 device; nothing at coarsest
        P_dev_c = (i <= length(P_cpus)) ? _csr_to_device(P_cpus[i], backend, Tc, Ti) : nothing
        R_dev_c = (i <= length(R_cpus)) ? _csr_to_device(R_cpus[i], backend, Tc, Ti) : nothing

        Dinv_c = mk_vec_c(n_c)
        diag_ptr_c_cpu = _build_diag_ptr_cpu(A_cpus[i])
        diag_ptr_c_dev = KernelAbstractions.zeros(backend, Int32, n_c)
        KernelAbstractions.copyto!(backend, diag_ptr_c_dev, diag_ptr_c_cpu)
        _amg_build_smoother_dinv!(opts.smoother, Dinv_c, A_dev_c, diag_ptr_c_dev, backend, workgroup)

        extras_c = LevelExtras{Tc, Nothing, CpuSpT_c}()
        extras_c.rho      = Tc(rhos[i])
        extras_c.diag_ptr = diag_ptr_c_dev

        if i < n_levels
            # Non-coarsest coarse level: SpGEMM scratch in Float32
            nc_c = size(P_cpus[i], 2)
            extras_c.P_cpu  = _to_tc_csr(P_cpus[i],         Tc)
            extras_c.R_cpu  = _to_tc_csr(R_cpus[i],         Tc)
            extras_c.AP_cpu = _to_tc_csr(AP_cpus[i],        Tc)
            extras_c.Ac_cpu = _to_tc_csr(deepcopy(A_cpus[i+1]), Tc)
            extras_c.A_cpu  = _to_tc_csr(deepcopy(A_cpus[i]),   Tc)
            nrows_AP_c = size(AP_cpus[i], 1)
            nrows_Ac_c = size(A_cpus[i+1], 1)
            max_nnz_c  = max(
                maximum(AP_cpus[i].rowptr[r+1] - AP_cpus[i].rowptr[r] for r in 1:nrows_AP_c; init=0),
                maximum(A_cpus[i+1].rowptr[r+1] - A_cpus[i+1].rowptr[r] for r in 1:nrows_Ac_c; init=0),
            )
            extras_c.cpu_tmps     = zeros(Tc,    max_nnz_c, Threads.nthreads())
            extras_c.col_to_local = zeros(Int32, nc_c,      Threads.nthreads())
        else
            # Coarsest coarse level: Float32 LU
            extras_c.A_cpu     = _to_tc_csr(A_cpus[end], Tc)
            extras_c.lu_dense  = lu_dense_c
            extras_c.lu_factor = lu_f
            extras_c.lu_rhs    = lu_rhs_init_c
        end

        level_c = LCType(A_dev_c, P_dev_c, R_dev_c,
                         Dinv_c, mk_vec_c(n_c), mk_vec_c(n_c), mk_vec_c(n_c), mk_vec_c(n_c),
                         extras_c)
        push!(coarse_levels, level_c)
    end

    # ── Commit to workspace ────────────────────────────────────────────────────
    ws.fine_level    = fine_level
    ws.coarse_levels = coarse_levels
    ws.x             = fine_level.x
    ws.setup_valid   = true
    ws.setup_count  += 1
    ws.update_count  = 1
    nothing
end

# ─── CPU CSR type-cast helper ─────────────────────────────────────────────────
# Convert a CPU CSR matrix to a different float type (Float64→Float32).
# Index arrays are shared (never mutated); only nzval is a new allocation.

function _to_tc_csr(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti},
                    ::Type{Tc}) where {Bi,Tv,Ti,Tc}
    m, n = size(A)
    # SparseMatrixCSR only accepts {Bi}; Tv and Ti are inferred from the vectors.
    SparseMatricesCSR.SparseMatrixCSR{Bi}(m, n, A.rowptr, A.colval, Tc.(A.nzval))
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
function update!(ws::AMGWorkspace{LFType, LCType}, A_device, backend, workgroup) where {LFType, LCType}
    if !ws.setup_valid || isnothing(ws.fine_level)
        amg_setup!(ws, A_device, backend, workgroup)
        return
    end

    ws.update_count += 1

    # Fine-level D⁻¹ always updated; diagonal changes every outer iteration.
    fine = ws.fine_level::LFType
    _amg_build_smoother_dinv!(ws.opts.smoother, fine.Dinv, fine.A, fine.extras.diag_ptr, backend, workgroup)

    # Lazy Galerkin refresh.
    update_freq = ws.opts.update_freq
    (ws.update_count - 1) % update_freq != 0 && return

    coarse = ws.coarse_levels
    if !isempty(coarse)
        # Fine→coarse1 Galerkin
        _galerkin_update!(fine, coarse[1], backend, workgroup)
        _amg_build_smoother_dinv!(ws.opts.smoother, coarse[1].Dinv, coarse[1].A,
                                   coarse[1].extras.diag_ptr, backend, workgroup)
        # Coarse-to-coarse Galerkin
        for lvl in 1:(length(coarse) - 1)
            _galerkin_update!(coarse[lvl], coarse[lvl + 1], backend, workgroup)
            _amg_build_smoother_dinv!(ws.opts.smoother, coarse[lvl+1].Dinv, coarse[lvl+1].A,
                                       coarse[lvl+1].extras.diag_ptr, backend, workgroup)
        end
        # Refresh coarsest-level (Float32) LU
        if !isnothing(coarse[end].extras.lu_factor)
            ex_c = coarse[end].extras
            nzval_c, _, _ = get_sparse_fields(coarse[end].A)
            copyto!(ex_c.A_cpu.nzval, nzval_c)   # Float32 device → Float32 CPU
            fill!(ex_c.lu_dense, zero(eltype(ex_c.lu_dense)))
            _fill_dense_from_sparse!(ex_c.lu_dense, ex_c.A_cpu)
            ex_c.lu_factor = lu!(ex_c.lu_dense; check=false)
        end
    else
        # Single-level: fine = coarsest — refresh Float64 LU
        if !isnothing(fine.extras.lu_factor)
            ex_f = fine.extras
            nzval_f, _, _ = get_sparse_fields(fine.A)
            copyto!(ex_f.A_cpu.nzval, nzval_f)
            fill!(ex_f.lu_dense, zero(eltype(ex_f.lu_dense)))
            _fill_dense_from_sparse!(ex_f.lu_dense, ex_f.A_cpu)
            ex_f.lu_factor = lu!(ex_f.lu_dense; check=false)
        end
    end
    nothing
end

# ─── Preconditioned Conjugate Gradient (PCG) solve ────────────────────────────
# One V-cycle per iteration as preconditioner M ≈ A⁻¹; O(√κ) vs O(κ) for plain Richardson.

function _amg_pcg_solve!(ws::AMGWorkspace{LFType}, b, values, itmax, atol, rtol,
                           backend, workgroup) where {LFType}
    L1  = ws.fine_level::LFType
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
    run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)
    # CRITICAL: vcycle_fine! calls amg_residual! at the fine level, overwriting
    # L1.r with the V-cycle's internal residual (≈ 0 at cycle end).
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
        run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)
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
Two-tier mixed-precision hierarchy: fine level Float64, coarse levels Float32.
All cycle kernel launches are type-stable with no boxing.

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

    update!(ws, A, backend, workgroup)

    if ws.opts.krylov === :cg
        _amg_pcg_solve!(ws, b, values, itmax, atol, rtol, backend, workgroup)
    else
        # ── Standalone Richardson: plain V-cycle loop ─────────────────────────
        L1 = ws.fine_level

        amg_copy!(L1.x, values, backend, workgroup)
        amg_copy!(L1.b, b,      backend, workgroup)

        amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
        r0 = amg_norm(L1.r)
        r0 = ifelse(r0 > eps(r0), r0, one(r0))

        if r0 >= atol
            check_freq = 5
            for k in 1:itmax
                run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)
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
    copyto!(ex.A_cpu.nzval, nzval_dev)   # device → CPU (may be Float32 or Float64)
    _spgemm_nzval!(ex.AP_cpu, ex.A_cpu, ex.P_cpu, ex.cpu_tmps, ex.col_to_local)
    _spgemm_nzval!(ex.Ac_cpu, ex.R_cpu, ex.AP_cpu, ex.cpu_tmps, ex.col_to_local)
    nzval_dev_c, _, _ = get_sparse_fields(Lc.A)
    # copyto! handles Float64→Float32 truncation at the fine→coarse1 boundary
    # (element-wise setindex! with implicit convert) and is a direct copy otherwise.
    copyto!(nzval_dev_c, ex.Ac_cpu.nzval)
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
        if Tv2 === Tv
            return SparseXCSR(A)
        else
            # Float type conversion (e.g. Float64 → Float32 for coarse levels).
            # SparseMatrixCSR only accepts {Bi} — Tv and Ti are inferred from the vectors.
            m, n = size(A)
            nzval_c = Tv2.(A.nzval)
            A_c = SparseMatricesCSR.SparseMatrixCSR{Bi}(m, n, A.rowptr, A.colval, nzval_c)
            return SparseXCSR(A_c)
        end
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
        amg_smooth!(level, 50, eltype(level.x)(2/3), backend, workgroup)
    end
end
