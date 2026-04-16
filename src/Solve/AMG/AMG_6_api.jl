export update!, _build_coarse_lu!, _refresh_coarse_lu!, _fill_dense_from_sparse!
export _build_smooth_AP_device!, _MAX_DENSE_LU_N, amg_clear_cache!
export _RANS_T_K, _RANS_T_OMEGA   # accessed from RANS turbulence models for sub-timing
export _AMG_T_RAP_CAST, _AMG_T_RAP_AP, _AMG_T_RAP_RAP  # RAP sub-timers (CUDA ext)

# Dense LU guard: above this coarsest-level size, fall back to 50 Jacobi sweeps.
# Prevents O(n²) memory allocation when coarsest_size is set to a large value to
# control hierarchy depth independently of the direct-solve threshold.
const _MAX_DENSE_LU_N = 5000

# Profiling accumulators; reset by amg_reset_stats!.
const _AMG_T_UPDATE       = Ref(0.0)
const _AMG_T_SOLVE        = Ref(0.0)
const _AMG_T_COARSE_SOLVE = Ref(0.0)
const _AMG_N_COARSE_SOLVE = Ref(0)
const _AMG_N_EARLY_EXIT   = Ref(0)
const _AMG_N_PCG_ITERS    = Ref(0)
const _AMG_N_PCG_SOLVES   = Ref(0)

# RANS and RAP sub-timers (GPU profiling).
const _RANS_T_K     = Ref(0.0)
const _RANS_T_OMEGA = Ref(0.0)
const _AMG_T_RAP_CAST = Ref(0.0)
const _AMG_T_RAP_AP   = Ref(0.0)
const _AMG_T_RAP_RAP  = Ref(0.0)

# update! phase timers — identify where the unexplained 150ms/iter overhead lives.
# All amortised over total solve_system! calls (not just lazy-update calls).
const _AMG_T_UPD_FINE_DINV  = Ref(0.0)  # fine-level Dinv rebuild (every call)
const _AMG_T_UPD_GALERKIN   = Ref(0.0)  # all _galerkin_update! calls combined (lazy)
const _AMG_T_UPD_COARSE_DINV= Ref(0.0)  # all coarse Dinv rebuilds (lazy)
const _AMG_T_UPD_LU         = Ref(0.0)  # coarsest LU refresh (lazy)

# Global workspace cache — allows reuse of pre-built hierarchy across run!() calls.
# Avoids repeated amg_setup! (~7s) when re-running the same problem (e.g. benchmark warmup).
const _GLOBAL_AMG_CACHE = Ref{Any}(nothing)
amg_clear_cache!() = (_GLOBAL_AMG_CACHE[] = nothing)

# ─── Dinv build dispatch: standard (diagonal) vs l1 (row-norm) ───────────────
# L1Jacobi uses l1 kernel; all others use fast diagonal-pointer path.

_amg_build_smoother_dinv!(::AbstractSmoother, Dinv, A, diag_ptr, backend, workgroup) =
    amg_build_Dinv!(Dinv, A, diag_ptr, backend, workgroup)

_amg_build_smoother_dinv!(::L1Jacobi, Dinv, A, diag_ptr, backend, workgroup) =
    amg_build_l1_Dinv!(Dinv, A, backend, workgroup)

# ─── Coarse LU build/refresh (CPU default; GPU extensions override) ──────────
# Build at setup; refresh every update_freq iterations. GPU override keeps LU on device.

"""
    _build_coarse_lu!(extras, A_cpu, backend, workgroup)

Build coarse LU in extras; CPU default on host, GPU override keeps on device.
"""
function _build_coarse_lu!(extras::LevelExtras, A_cpu, backend, workgroup)
    Tv = typeof(extras.rho)
    n  = size(A_cpu, 1)
    lu_dense     = zeros(Tv, n, n)
    _fill_dense_from_sparse!(lu_dense, A_cpu)
    extras.lu_dense  = lu_dense
    extras.lu_factor = lu!(lu_dense; check=false)
    extras.lu_rhs    = zeros(Tv, n)
    nothing
end

"""
    _refresh_coarse_lu!(extras, A_device, backend)

Refill dense matrix and recompute LU; CPU default on host, GPU override on device.
"""
function _refresh_coarse_lu!(extras::LevelExtras, A_device, backend)
    nzval_c, _, _ = get_sparse_fields(A_device)
    copyto!(extras.A_cpu.nzval, nzval_c)          # device sparse nzval → CPU
    n = size(extras.A_cpu, 1)
    Tv = typeof(extras.rho)
    lu_dense_new = zeros(Tv, n, n)
    _fill_dense_from_sparse!(lu_dense_new, extras.A_cpu)
    copyto!(extras.lu_dense, lu_dense_new)         # update dense (CPU or GPU)
    extras.lu_factor = lu!(extras.lu_dense; check=false)
    nothing
end

# ─── Device AP pre-allocation hook (smooth_P only) ────────────────────────────
# CPU default: no-op. GPU override uploads AP_cpu to device (zero-allocation kernel path).

"""
    _build_smooth_AP_device!(extras, AP_cpu, backend)

Upload pre-computed A*P to device; CPU default no-op, GPU override allocates device matrix.
"""
function _build_smooth_AP_device!(extras::LevelExtras, AP_cpu, backend)
    nothing   # CPU: AP_cpu is already in extras.AP_cpu; KA kernels use it directly
end

# ─── Workspace constructor ────────────────────────────────────────────────────

"""
    _workspace(amg::AMG, A, b) → AMGWorkspace

Build fully-typed workspace: fine (Float64 A, Float32 P/R), coarse (all Float32).
Hierarchy construction deferred to first `update!` call (A not yet assembled).
"""
function _workspace(amg::AMG, A::AT, b::AbstractVector{Tv}) where {AT, Tv}
    # Return cached workspace if one exists with matching problem size and is fully built.
    cached = _GLOBAL_AMG_CACHE[]
    if cached isa AMGWorkspace && cached.setup_valid &&
       !isnothing(cached.fine_level) && length(cached.x) == length(b)
        return cached
    end

    Tc        = amg.coarse_float
    AT_c      = _tc_sparse_type(AT)
    Vec_f     = typeof(b)
    Vec_c     = _tc_vec_type(Vec_f)
    # Verify coarse_float consistency with type-mapping methods.
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
    r_pcg = similar(b); fill!(r_pcg, zero(Tv))
    ws = AMGWorkspace{LFType, LCType, Vec_f, typeof(amg)}(
        nothing, LCType[], x, amg, false, 0, 0, x_pcg, p_cg, r_pcg, 0, 0)
    _GLOBAL_AMG_CACHE[] = ws
    return ws
end

# ─── Full hierarchy setup ─────────────────────────────────────────────────────

"""
    amg_setup!(ws, A_device, backend, workgroup)

Build complete mixed-precision hierarchy: Phase 1 (CPU coarsening Float64), Phase 2 (device upload).
Calling again replaces entire hierarchy.
"""
function amg_setup!(ws::AMGWorkspace{LFType, LCType}, A_device, backend, workgroup) where {LFType, LCType}
    # Reset all diagnostics on hierarchy rebuild so timing and iteration counts stay in sync.
    _AMG_T_UPDATE[]       = 0.0
    _AMG_T_SOLVE[]        = 0.0
    _AMG_T_COARSE_SOLVE[] = 0.0
    _AMG_N_COARSE_SOLVE[] = 0
    _AMG_N_EARLY_EXIT[]   = 0
    _AMG_N_PCG_ITERS[]    = 0
    _AMG_N_PCG_SOLVES[]   = 0
    _AMG_T_RAP_CAST[]        = 0.0
    _AMG_T_RAP_AP[]          = 0.0
    _AMG_T_RAP_RAP[]         = 0.0
    _AMG_T_UPD_FINE_DINV[]   = 0.0
    _AMG_T_UPD_GALERKIN[]    = 0.0
    _AMG_T_UPD_COARSE_DINV[] = 0.0
    _AMG_T_UPD_LU[]          = 0.0
    ws._pcg_iters            = 0
    ws._solve_count          = 0
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
    for level in 2:opts.max_levels
        n_cur = size(A_cur, 1)
        n_cur <= opts.coarsest_size && break

        # Apply user θ at fine→coarse-1 only; use θ=0 at deeper levels (Galerkin no longer reflects mesh).
        θ_level = level == 2 ? opts.strength : 0.0
        agg, nagg = amg_coarsen(A_cur, θ_level, opts.coarsening)
        # Fallback to pairwise when primary coarsening is degenerate; also used when coarsest_size
        # forces levels deeper than the natural SA/RS quality threshold.
        used_pairwise = false
        if nagg >= n_cur || nagg > (n_cur * 55) ÷ 100 || nagg < max(2, n_cur ÷ 50)
            agg_p, nagg_p = _coarsen_pairwise(A_cur)
            if nagg_p < n_cur
                agg, nagg = agg_p, nagg_p
                used_pairwise = true
            end
        end
        used_pairwise && @debug "AMG level $(level): pairwise fallback (n=$(n_cur) → nagg=$(nagg))"
        nagg >= n_cur && break   # pairwise also failed — truly disconnected
        nagg < 2 && break        # absolute minimum

        P_tent = build_tentative_P(n_cur, nagg, agg)
        P_cpu  = opts.smooth_P ? smooth_prolongation(A_cur, P_tent, 2/3) : P_tent
        P_cpu  = opts.trunc_P > 0.0 ? _truncate_P(P_cpu, opts.trunc_P) : P_cpu
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
        Ac_diag_min < eps(Float64) * _fine_diag_max && break

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
    _coarsest_solve = level_sizes[end] <= opts.coarsest_size && level_sizes[end] <= _MAX_DENSE_LU_N ? "dense LU" : "Jacobi"
    @info "AMG hierarchy ($(opts.coarsening), strength=$(opts.strength)): $(level_sizes) — op_complexity=$(op_cmplx) — coarsest solve: $(_coarsest_solve)"

    # W-cycle on GPU: 2^(n_levels-2) coarsest-level transfers per cycle.
    if opts.cycle isa WCycle && !(backend isa CPU) && n_levels > 4
        @warn "WCycle on GPU with $(n_levels) coarse levels: each cycle requires 2^$(n_levels-2) = $(2^(n_levels-2)) synchronous CPU↔GPU coarse-solve transfers. Use VCycle() instead — WCycle is only efficient on CPU or with ≤4 levels."
    end

    # ── Phase 2a: coarsest-level LU (single-level case only) ─────────────────
    n_coarsest  = size(A_cpus[end], 1)
    lu_dense_f  = nothing   # Float64 LU only for the single-level (n_levels==1) case
    lu_f_fine   = nothing
    lu_rhs_init_f = Tv[]

    if n_coarsest <= opts.coarsest_size && n_levels == 1
        lu_dense_f    = zeros(Tv, n_coarsest, n_coarsest)
        _fill_dense_from_sparse!(lu_dense_f, A_cpus[end])
        lu_f_fine     = lu!(lu_dense_f; check=false)
        lu_rhs_init_f = zeros(Tv, n_coarsest)
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
        extras_f.P_cpu    = P_cpus[1]
        extras_f.R_cpu    = R_cpus[1]
        extras_f.AP_cpu   = AP_cpus[1]
        extras_f.Ac_cpu   = deepcopy(A_cpus[2])
        extras_f.A_cpu    = deepcopy(A_cpus[1])
        extras_f.smooth_P = opts.smooth_P
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
            extras_c.P_cpu    = _to_tc_csr(P_cpus[i],         Tc)
            extras_c.R_cpu    = _to_tc_csr(R_cpus[i],         Tc)
            extras_c.AP_cpu   = _to_tc_csr(AP_cpus[i],        Tc)
            extras_c.Ac_cpu   = _to_tc_csr(deepcopy(A_cpus[i+1]), Tc)
            extras_c.A_cpu    = _to_tc_csr(deepcopy(A_cpus[i]),   Tc)
            extras_c.smooth_P = opts.smooth_P
            nrows_AP_c = size(AP_cpus[i], 1)
            nrows_Ac_c = size(A_cpus[i+1], 1)
            max_nnz_c  = max(
                maximum(AP_cpus[i].rowptr[r+1] - AP_cpus[i].rowptr[r] for r in 1:nrows_AP_c; init=0),
                maximum(A_cpus[i+1].rowptr[r+1] - A_cpus[i+1].rowptr[r] for r in 1:nrows_Ac_c; init=0),
            )
            extras_c.cpu_tmps     = zeros(Tc,    max_nnz_c, Threads.nthreads())
            extras_c.col_to_local = zeros(Int32, nc_c,      Threads.nthreads())
        else
            # Coarsest coarse level: build dense LU if small enough.
            extras_c.A_cpu = _to_tc_csr(A_cpus[end], Tc)
            if n_coarsest <= opts.coarsest_size && n_coarsest <= _MAX_DENSE_LU_N
                _build_coarse_lu!(extras_c, extras_c.A_cpu, backend, workgroup)
            end
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
    _GLOBAL_AMG_CACHE[] = ws
    nothing
end

# ─── CPU CSR type-cast helper ─────────────────────────────────────────────────
# Convert CPU CSR matrix float type (index arrays shared, nzval reallocated).

function _to_tc_csr(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti},
                    ::Type{Tc}) where {Bi,Tv,Ti,Tc}
    m, n = size(A)
    SparseMatricesCSR.SparseMatrixCSR{Bi}(m, n, A.rowptr, A.colval, Tc.(A.nzval))
end

# ─── Numerical update (reuse hierarchy structure) ─────────────────────────────

"""
    update!(ws, A_device, backend, workgroup)

Refresh numerical values in AMG hierarchy (same sparsity, new coefficients).
Falls back to `amg_setup!` if not yet built.

**Always updated**: fine D⁻¹ (single kernel, no row scan).
**Lazily updated** (every `update_freq` calls): Galerkin products, coarse D⁻¹, coarsest LU.

Lazy refresh is safe: outer SIMPLE/PISO loop is itself iterative; stale coarse correction
adds a few V-cycles per outer iteration. Fine D⁻¹ never skipped.
"""
function update!(ws::AMGWorkspace{LFType, LCType}, A_device, backend, workgroup) where {LFType, LCType}
    if !ws.setup_valid || isnothing(ws.fine_level)
        amg_setup!(ws, A_device, backend, workgroup)
        return
    end

    fine = ws.fine_level::LFType

    # Sync nzvals when workspace was built for a different A object (cross-run cache reuse).
    # Both matrices have identical sparsity; nzval copy is GPU-to-GPU and cheap.
    if fine.A !== A_device
        copyto!(_nzval(fine.A), _nzval(A_device))
        KernelAbstractions.synchronize(backend)
    end

    ws.update_count += 1

    # Fine-level D⁻¹ always updated; diagonal changes every outer iteration.
    _AMG_T_UPD_FINE_DINV[] += @elapsed begin
        _amg_build_smoother_dinv!(ws.opts.smoother, fine.Dinv, fine.A, fine.extras.diag_ptr, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end

    # Lazy Galerkin refresh.
    update_freq = ws.opts.update_freq
    (ws.update_count - 1) % update_freq != 0 && return

    coarse = ws.coarse_levels
    if !isempty(coarse)
        # Fine→coarse1 Galerkin
        _AMG_T_UPD_GALERKIN[] += @elapsed begin
            _galerkin_update!(fine, coarse[1], backend, workgroup)
            KernelAbstractions.synchronize(backend)
        end
        _AMG_T_UPD_COARSE_DINV[] += @elapsed begin
            _amg_build_smoother_dinv!(ws.opts.smoother, coarse[1].Dinv, coarse[1].A,
                                       coarse[1].extras.diag_ptr, backend, workgroup)
            KernelAbstractions.synchronize(backend)
        end
        # Coarse-to-coarse Galerkin
        for lvl in 1:(length(coarse) - 1)
            _AMG_T_UPD_GALERKIN[] += @elapsed begin
                _galerkin_update!(coarse[lvl], coarse[lvl + 1], backend, workgroup)
                KernelAbstractions.synchronize(backend)
            end
            _AMG_T_UPD_COARSE_DINV[] += @elapsed begin
                _amg_build_smoother_dinv!(ws.opts.smoother, coarse[lvl+1].Dinv, coarse[lvl+1].A,
                                           coarse[lvl+1].extras.diag_ptr, backend, workgroup)
                KernelAbstractions.synchronize(backend)
            end
        end
        # Refresh coarsest-level LU
        let ex = coarse[end].extras
            if !isnothing(ex.lu_factor)
                _AMG_T_UPD_LU[] += @elapsed _refresh_coarse_lu!(ex, coarse[end].A, backend)
            end
        end
    else
        # Single-level: fine = coarsest — refresh LU (always CPU path)
        ex_f = fine.extras
        if !isnothing(ex_f.lu_factor)
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
# One V-cycle per iteration as preconditioner; O(√κ) convergence vs O(κ) for Richardson.

function _amg_pcg_solve!(ws::AMGWorkspace{LFType}, b, values, itmax, atol, rtol,
                           backend, workgroup) where {LFType}
    L1  = ws.fine_level::LFType
    Tv  = eltype(ws.x_pcg)
    x   = ws.x_pcg
    p   = ws.p_cg
    r   = ws.r_pcg   # CG residual — kept here so the V-cycle cannot clobber it

    # Initialize and compute initial residual.
    amg_copy!(x, values, backend, workgroup)
    amg_residual!(r, L1.A, x, b, backend, workgroup)
    r0 = amg_norm(r)
    r0 = ifelse(r0 > eps(r0), r0, one(r0))

    # Early exit if initial guess already satisfies tolerance.
    if r0 < atol
        _AMG_N_EARLY_EXIT[] += 1
        return 0
    end

    # First V-cycle: z ← M⁻¹ r.
    amg_copy!(L1.b, r, backend, workgroup)
    amg_zero!(L1.x, backend, workgroup)
    run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)

    # Initialize CG direction: p ← z, compute rz = rᵀ z.
    amg_copy!(p, L1.x, backend, workgroup)
    rz = dot(r, p)

    # PCG iterations.
    niters = 1
    for k in 1:itmax
        niters = k
        amg_spmv!(L1.tmp, L1.A, p, backend, workgroup)
        pAp   = dot(p, L1.tmp)
        alpha = rz / pAp
        amg_axpy!(x, p,    alpha,  backend, workgroup)
        amg_axpy!(r, L1.tmp, -alpha, backend, workgroup)

        # Convergence check before next V-cycle.
        res_norm = amg_norm(r)
        (res_norm < atol || res_norm / r0 < rtol) && break

        # Next V-cycle: z ← M⁻¹ r.
        amg_copy!(L1.b, r, backend, workgroup)
        amg_zero!(L1.x, backend, workgroup)
        run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)

        # CG update: β = (rᵀ z_new) / (rᵀ z_old).
        rz_new = dot(r, L1.x)
        beta   = rz_new / rz
        rz     = rz_new
        amg_axpby!(p, L1.x, one(Tv), beta, backend, workgroup)
    end

    # Copy PCG solution back to field values.
    amg_copy!(values, x, backend, workgroup)
    return niters
end

# ─── solve_system! dispatch ────────────────────────────────────────────────────

"""
    solve_system!(phiEqn, setup, result, component, config)

AMG override: two-tier mixed-precision hierarchy (fine Float64, coarse Float32).
Type-stable cycles. Dispatches to PCG (default) or Richardson based on `krylov` option.
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

    KernelAbstractions.synchronize(backend)
    t_upd = @elapsed begin
        update!(ws, A, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end
    _AMG_T_UPDATE[] += t_upd

    t_slv = @elapsed begin
        niters = if ws.opts.krylov === :cg
            _amg_pcg_solve!(ws, b, values, itmax, atol, rtol, backend, workgroup)
        else
            # Richardson: plain V-cycle loop (check convergence every 5 iterations).
            L1 = ws.fine_level
            amg_copy!(L1.x, values, backend, workgroup)
            amg_copy!(L1.b, b,      backend, workgroup)
            amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
            r0 = amg_norm(L1.r)
            r0 = ifelse(r0 > eps(r0), r0, one(r0))

            niters_rich = 0
            if r0 >= atol
                check_freq = 5
                for k in 1:itmax
                    run_cycle!(ws, ws.opts, ws.opts.cycle, backend, workgroup)
                    niters_rich = k
                    if k == 1 || k % check_freq == 0 || k == itmax
                        amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
                        res_norm = amg_norm(L1.r)
                        (res_norm < atol || res_norm / r0 < rtol) && break
                    end
                end
            end
            amg_copy!(values, L1.x, backend, workgroup)
            niters_rich
        end
        KernelAbstractions.synchronize(backend)
    end
    _AMG_T_SOLVE[] += t_slv

    # Accumulate diagnostics; print every 50 solves.
    ws._pcg_iters   += niters
    ws._solve_count += 1
    if niters > 0
        _AMG_N_PCG_ITERS[]  += niters
        _AMG_N_PCG_SOLVES[] += 1
    end
    if ws._solve_count % 50 == 0
        n_early  = _AMG_N_EARLY_EXIT[]
        n_active = ws._solve_count - n_early
        avg_iter = n_active > 0 ? round(ws._pcg_iters / n_active; digits=1) : 0.0
        n        = ws._solve_count
        upd_ms   = round(_AMG_T_UPDATE[]          / n * 1e3; digits=2)
        slv_ms   = round(_AMG_T_SOLVE[]            / n * 1e3; digits=2)
        coarse_μs = _AMG_N_COARSE_SOLVE[] > 0 ?
                    round(_AMG_T_COARSE_SOLVE[] / _AMG_N_COARSE_SOLVE[] * 1e6; digits=1) : 0.0
        fdinv_ms = round(_AMG_T_UPD_FINE_DINV[]   / n * 1e3; digits=2)
        gal_ms   = round(_AMG_T_UPD_GALERKIN[]     / n * 1e3; digits=2)
        cdinv_ms = round(_AMG_T_UPD_COARSE_DINV[]  / n * 1e3; digits=2)
        lu_ms    = round(_AMG_T_UPD_LU[]           / n * 1e3; digits=2)
        @info "AMG pressure: $(ws._solve_count) solves ($(n_early) early-exit, $(n_active) active), " *
              "avg $(avg_iter) iter/active — update $(upd_ms) ms, solve $(slv_ms) ms, " *
              "coarse $(coarse_μs) μs/call ($(ws.opts.krylov))\n" *
              "  update breakdown: fine_dinv=$(fdinv_ms) ms, galerkin=$(gal_ms) ms, " *
              "coarse_dinv=$(cdinv_ms) ms, lu=$(lu_ms) ms"
    end

    return residual(phiEqn, component, config)
end

"""
    amg_reset_stats!(ws::AMGWorkspace)

Reset accumulated iteration and solve-count diagnostics (call after warm-up phase).
"""
function amg_reset_stats!(ws::AMGWorkspace)
    ws._pcg_iters           = 0
    ws._solve_count         = 0
    _AMG_T_UPDATE[]         = 0.0
    _AMG_T_SOLVE[]          = 0.0
    _AMG_T_COARSE_SOLVE[]   = 0.0
    _AMG_N_COARSE_SOLVE[]   = 0
    _AMG_N_EARLY_EXIT[]     = 0
    _AMG_N_PCG_ITERS[]      = 0
    _AMG_N_PCG_SOLVES[]     = 0
    _RANS_T_K[]              = 0.0
    _RANS_T_OMEGA[]          = 0.0
    _AMG_T_RAP_CAST[]       = 0.0
    _AMG_T_RAP_AP[]         = 0.0
    _AMG_T_RAP_RAP[]        = 0.0
    _AMG_T_UPD_FINE_DINV[]  = 0.0
    _AMG_T_UPD_GALERKIN[]   = 0.0
    _AMG_T_UPD_COARSE_DINV[]= 0.0
    _AMG_T_UPD_LU[]         = 0.0
    nothing
end

# Zero-arg overload: reset global accumulators (no workspace available).
function amg_reset_stats!()
    _AMG_T_UPDATE[]          = 0.0
    _AMG_T_SOLVE[]           = 0.0
    _AMG_T_COARSE_SOLVE[]    = 0.0
    _AMG_N_COARSE_SOLVE[]    = 0
    _AMG_N_EARLY_EXIT[]      = 0
    _AMG_N_PCG_ITERS[]       = 0
    _AMG_N_PCG_SOLVES[]      = 0
    _RANS_T_K[]               = 0.0
    _RANS_T_OMEGA[]           = 0.0
    _AMG_T_RAP_CAST[]        = 0.0
    _AMG_T_RAP_AP[]          = 0.0
    _AMG_T_RAP_RAP[]         = 0.0
    _AMG_T_UPD_FINE_DINV[]   = 0.0
    _AMG_T_UPD_GALERKIN[]    = 0.0
    _AMG_T_UPD_COARSE_DINV[] = 0.0
    _AMG_T_UPD_LU[]          = 0.0
    nothing
end

# ─── Galerkin update: Ac = R·A·P ─────────────────────────────────────────────
# CPU path: download A.nzval, two in-place CPU SpGEMMs, upload Ac.nzval (no allocation).

function _galerkin_update!(L::MultigridLevel, Lc::MultigridLevel,
                            backend::KernelAbstractions.CPU, workgroup)
    ex = L.extras
    nzval_dev, _, _ = get_sparse_fields(L.A)
    copyto!(ex.A_cpu.nzval, nzval_dev)
    _spgemm_nzval!(ex.AP_cpu, ex.A_cpu, ex.P_cpu, ex.cpu_tmps, ex.col_to_local)
    _spgemm_nzval!(ex.Ac_cpu, ex.R_cpu, ex.AP_cpu, ex.cpu_tmps, ex.col_to_local)
    nzval_dev_c, _, _ = get_sparse_fields(Lc.A)
    # copyto! handles Float64→Float32 truncation implicitly at fine→coarse boundary.
    copyto!(nzval_dev_c, ex.Ac_cpu.nzval)
end

# GPU path: KA kernels scatter contributions on-device (no PCIe transfer).
# Both smooth_P=false (1 nnz/row P) and smooth_P=true (multi-nnz P) use on-device kernels.
function _galerkin_update!(L::MultigridLevel, Lc::MultigridLevel, backend, workgroup)
    if L.extras.smooth_P
        # GPU: cuSPARSE SpGEMM (via CUDA override) ~6× faster than KA for large matrices.
        amg_rap_update_smooth!(Lc, L, backend, workgroup)
    else
        amg_rap_update!(Lc, L, backend, workgroup)
    end
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
            # Float type conversion (e.g. Float64 → Float32).
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

# Coarse solve: dense LU if available, else Jacobi sweeps.
function amg_coarse_solve!(level::MultigridLevel, coarse_sweeps::Int, backend)
    ex = level.extras
    if !isnothing(ex.lu_factor)
        t = @elapsed begin
            copyto!(ex.lu_rhs, level.b)
            ldiv!(ex.lu_factor, ex.lu_rhs)
            copyto!(level.x, ex.lu_rhs)
        end
        _AMG_T_COARSE_SOLVE[] += t
        _AMG_N_COARSE_SOLVE[] += 1
    else
        amg_smooth!(level, coarse_sweeps, eltype(level.x)(2/3), backend, 1024)
    end
end
