export update!

# ─── Workspace constructor ────────────────────────────────────────────────────

"""
    _workspace(amg::AMG, b) → AMGWorkspace

Builds a skeleton AMGWorkspace. The actual hierarchy is constructed lazily on the
first call to `solve_system!`.  `b` is the reference RHS vector on the target backend.
"""
function _workspace(amg::AMG, b::AbstractVector{Tv}) where {Tv}
    x = similar(b)
    fill!(x, zero(Tv))
    # levels is empty until first solve; setup_valid=false triggers amg_setup!
    return AMGWorkspace(Any[], x, amg, false, 0)
end

# ─── Utility: Tv and Ti from a device matrix ────────────────────────────────

_amg_Tv(A) = eltype(_nzval(A))
_amg_Ti(A) = eltype(_rowptr(A))

# ─── Full hierarchy setup ─────────────────────────────────────────────────────

"""
    amg_setup!(ws, A_device, backend, workgroup)

Build the complete multigrid hierarchy from `A_device`.  All coarsening and Galerkin
products are performed on CPU; only the work vectors and final matrices are kept on
the target device.  Calling this again replaces the entire hierarchy.
"""
function amg_setup!(ws::AMGWorkspace, A_device, backend, workgroup)
    opts   = ws.opts
    Tv     = _amg_Tv(A_device)
    Ti     = _amg_Ti(A_device)
    n_fine = size(A_device, 1)

    mk_vec(n) = KernelAbstractions.zeros(backend, Tv, n)

    # ── Finest level ───────────────────────────────────────────────────────────
    Dinv_fine = mk_vec(n_fine)
    amg_build_Dinv!(Dinv_fine, A_device, backend, workgroup)

    # Level 1 holds A_device directly (no copy); P and R set when next level is built
    lvl1 = MultigridLevel(A_device, nothing, nothing,
                           Dinv_fine,
                           mk_vec(n_fine), mk_vec(n_fine),
                           mk_vec(n_fine), mk_vec(n_fine))

    # Spectral radius estimate on CPU
    A_cpu   = _to_cpu_csr(A_device)
    D_cpu   = Array(Dinv_fine)
    lvl1.rho[] = estimate_spectral_radius(A_cpu, D_cpu)

    levels = [lvl1]
    A_cur  = A_cpu

    # Track fine-level diagonal scale for the numerical breakdown check below.
    # D_cpu[i] = 1/A[i,i], so 1/D_cpu[i] = A[i,i].
    _fine_diag_max = maximum(1.0 ./ D_cpu)

    # ── Coarsening loop ────────────────────────────────────────────────────────
    for _ in 2:opts.max_levels
        n_cur = size(A_cur, 1)
        n_cur <= opts.coarsest_size && break

        agg, nagg = amg_coarsen(A_cur, opts.strength, opts.coarsening)
        nagg >= n_cur && break   # no coarsening achieved

        # Stop if coarsening ratio is too poor (< 20% DOF reduction).
        # On non-uniform CFD meshes SA aggregation can stall, producing near-identity
        # Galerkin products whose diagonals decay to machine epsilon within a few levels.
        nagg > 0.8 * n_cur && break

        # Diagonal and spectral radius of current level
        D_cur = _extract_dinv_cpu(A_cur)
        ρ_cur = estimate_spectral_radius(A_cur, D_cur)

        # Clamp ω_P: for near-zero spectral radius the un-clamped value would
        # produce enormous P entries (Inf-propagation through the Galerkin product).
        # A safe upper bound of 4/3 (ρ=1) keeps the smoother well-behaved.
        ω_P = min(4.0 / (3.0 * max(ρ_cur, eps(Float64))), 4.0/3.0)

        # Tentative prolongation (piecewise-constant aggregates)
        P_tent = build_tentative_P(n_cur, nagg, agg)

        # Smoothed prolongation: P = (I - ω_P D⁻¹ A) P̂
        P_cpu = smooth_prolongation(A_cur, P_tent, ω_P)

        # Guard: if smoothing produced non-finite entries stop coarsening here
        any(!isfinite, P_cpu.nzval) && break

        # Restriction R = Pᵀ
        R_cpu = build_restriction(P_cpu)

        # Galerkin coarse matrix Ac = R A P
        Ac_cpu = galerkin_product(R_cpu, A_cur, P_cpu)

        nc = size(Ac_cpu, 1)
        nc == 0 && break
        any(!isfinite, Ac_cpu.nzval) && break

        # Stop if the coarse diagonal has decayed to near machine epsilon relative to
        # the fine level — this signals numerical breakdown of the hierarchy.
        Ac_diag_min = minimum(v for i in 1:nc
                              for (j,v) in zip(Ac_cpu.colval[Ac_cpu.rowptr[i]:Ac_cpu.rowptr[i+1]-1],
                                               Ac_cpu.nzval[Ac_cpu.rowptr[i]:Ac_cpu.rowptr[i+1]-1])
                              if j == i)
        Ac_diag_min < sqrt(eps(Float64)) * _fine_diag_max && break

        # Coarse level Dinv and spectral radius
        D_c = _extract_dinv_cpu(Ac_cpu)
        ρ_c = estimate_spectral_radius(Ac_cpu, D_c)

        # Transfer to device
        P_dev  = _csr_to_device(P_cpu,  backend, Tv, Ti)
        R_dev  = _csr_to_device(R_cpu,  backend, Tv, Ti)
        Ac_dev = _csr_to_device(Ac_cpu, backend, Tv, Ti)

        Dinv_c = mk_vec(nc)
        KernelAbstractions.copyto!(backend, Dinv_c, D_c)

        # Assign P and R to the current finest level (levels[end]).
        # Also cache the CPU copies so update! can reuse them without a GPU→CPU round-trip.
        levels[end].P     = P_dev
        levels[end].R     = R_dev
        levels[end].P_cpu = P_cpu
        levels[end].R_cpu = R_cpu

        # Create new coarse level (P,R filled in next iteration or left as nothing)
        lvl_c = MultigridLevel(Ac_dev, nothing, nothing,
                                Dinv_c,
                                mk_vec(nc), mk_vec(nc), mk_vec(nc), mk_vec(nc))
        lvl_c.rho[] = ρ_c
        push!(levels, lvl_c)

        A_cur = Ac_cpu
    end

    # ── Coarsest-level LU ──────────────────────────────────────────────────────
    _build_coarsest_lu!(levels[end], A_cur)

    # ── Commit to workspace ────────────────────────────────────────────────────
    ws.levels = levels
    ws.x      = levels[1].x
    ws.setup_valid  = true
    ws.setup_count += 1
    nothing
end

# ─── Numerical update (reuse hierarchy structure) ─────────────────────────────

"""
    update!(ws, A_device, backend, workgroup)

Refresh numerical values in the AMG hierarchy after the fine-level matrix `A_device`
has changed in-place (same sparsity pattern, new coefficients). Coarse matrices are
recomputed via Galerkin products; work vectors are reused.  If the hierarchy has not
yet been built, calls `amg_setup!` instead.
"""
function update!(ws::AMGWorkspace, A_device, backend, workgroup)
    if !ws.setup_valid || isempty(ws.levels)
        amg_setup!(ws, A_device, backend, workgroup)
        return
    end

    # Level 1 holds A_device directly — already up-to-date since the outer solver
    # mutates it in place.  Only rebuild Dinv and coarse levels.
    amg_build_Dinv!(ws.levels[1].Dinv, ws.levels[1].A, backend, workgroup)

    A_cur = _to_cpu_csr(A_device)
    for lvl in 1:(length(ws.levels) - 1)
        L  = ws.levels[lvl]
        Lc = ws.levels[lvl + 1]

        # Use cached CPU copies — avoids a GPU→CPU round-trip on GPU backends.
        # On CPU, P_cpu/R_cpu were set to parent(SparseXCSR) (zero-copy) at setup.
        P_cpu = L.P_cpu
        R_cpu = L.R_cpu
        Ac_new = galerkin_product(R_cpu, A_cur, P_cpu)

        # Write updated nzval into the device matrix
        _copy_nzval_to_device!(Lc.A, Ac_new.nzval, backend)

        # Rebuild Dinv for the coarse level
        amg_build_Dinv!(Lc.Dinv, Lc.A, backend, workgroup)

        A_cur = Ac_new
    end

    # Rebuild coarsest-level LU
    _build_coarsest_lu!(ws.levels[end], A_cur)
    nothing
end

# ─── solve_system! dispatch ────────────────────────────────────────────────────

"""
    solve_system!(phiEqn, setup, result, component, config)

AMG-specific override dispatched when `phiEqn.solver isa AMGWorkspace`.
Implements the same interface contract as the Krylov.jl path in Solve_1_api.jl.
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

    L1 = ws.levels[1]

    # Initialise finest level from current field values
    amg_copy!(L1.x, values, backend, workgroup)
    amg_copy!(L1.b, b,      backend, workgroup)

    # Initial residual norm
    amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
    r0 = amg_norm(L1.r)
    r0 = ifelse(r0 > eps(r0), r0, one(r0))

    # Multigrid cycle iterations
    for _ in 1:itmax
        run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)

        amg_residual!(L1.r, L1.A, L1.x, L1.b, backend, workgroup)
        res_norm = amg_norm(L1.r)
        (res_norm < atol || res_norm / r0 < rtol) && break
    end

    # Copy solution back into the field values array
    amg_copy!(values, L1.x, backend, workgroup)

    # Return field residual using the standard XCALibre definition
    return residual(phiEqn, component, config)
end

# ─── CSR ↔ device helpers ─────────────────────────────────────────────────────

# Gather any sparse matrix to a CPU SparseMatrixCSR
_to_cpu_csr(A::SparseXCSR) = parent(A)
_to_cpu_csr(A::SparseMatricesCSR.SparseMatrixCSR) = A

function _to_cpu_csr(A)
    # GPU case: gather from device and reconstruct as CPU SparseMatrixCSR via COO
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
    cols = Vector{Int}(colval)
    return sparsecsr(rows, cols, nzval, m, n)
end

# Transfer a CPU SparseMatrixCSR to the target device
function _csr_to_device(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti},
                         backend, ::Type{Tv2}, ::Type{Ti2}) where {Bi,Tv,Ti,Tv2,Ti2}
    if backend isa CPU
        return SparseXCSR(A)
    else
        # Build COO then delegate to the backend-specific _build_A
        m  = size(A, 1)
        rows = Int[]
        cols = Int[]
        vals = Tv2[]
        for i in 1:m
            for nzi in A.rowptr[i]:(A.rowptr[i+1]-1)
                push!(rows, i)
                push!(cols, A.colval[nzi])
                push!(vals, Tv2(A.nzval[nzi]))
            end
        end
        return _build_A(backend, rows, cols, vals, m)
    end
end

# Copy updated nzval into the device matrix (in-place, no sparsity change)
function _copy_nzval_to_device!(A::SparseXCSR, nzval_cpu::AbstractVector, ::CPU)
    copyto!(parent(A).nzval, nzval_cpu)
end

function _copy_nzval_to_device!(A, nzval_cpu::AbstractVector, backend)
    KernelAbstractions.copyto!(backend, _nzval(A), nzval_cpu)
end

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Extract 1/diagonal from a CPU CSR matrix
function _extract_dinv_cpu(A::SparseMatricesCSR.SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    n = size(A, 1)
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

# Build dense LU for the coarsest level
function _build_coarsest_lu!(level::MultigridLevel{Tv}, A_cpu::SparseMatricesCSR.SparseMatrixCSR) where {Tv}
    n = size(A_cpu, 1)
    Adense = zeros(Tv, n, n)
    for i in 1:n
        for nzi in A_cpu.rowptr[i]:(A_cpu.rowptr[i+1]-1)
            Adense[i, A_cpu.colval[nzi]] = A_cpu.nzval[nzi]
        end
    end
    level.lu_factor = lu!(Adense)
    level.lu_rhs    = zeros(Tv, n)
    nothing
end

# Coarse solve via LU (host-side).
# copyto!(cpu_vec, device_vec) is a single device→host transfer with no intermediate
# allocation; on CPU it is a plain memcpy between two host vectors.
function amg_coarse_solve!(level::MultigridLevel, backend)
    copyto!(level.lu_rhs, level.b)
    ldiv!(level.lu_factor, level.lu_rhs)
    KernelAbstractions.copyto!(backend, level.x, level.lu_rhs)
end
