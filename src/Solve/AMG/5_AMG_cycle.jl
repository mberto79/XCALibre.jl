@kernel function _amg_coarse_diagonal_solve_kernel!(x, rowptr, colval, nzval, b)
    i = @index(Global)
    T = eltype(x)
    value = zero(T)
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        if colval[p] == i
            aii = nzval[p]
            value = abs(aii) > eps(T) ? b[i] / aii : zero(T)
            break
        end
    end
    @inbounds x[i] = value
end

function _is_diagonal_matrix(A)
    _m(A) == _n(A) || return false
    rowptr = _rowptr(A)
    colval = _colval(A)
    @inbounds for i in 1:_m(A)
        row_has_diagonal = false
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            colval[p] == i || return false
            row_has_diagonal = true
        end
        row_has_diagonal || return false
    end
    return true
end

function _coarse_solve_on_device!(hierarchy::AMGHierarchy, level::AMGLevel, b)
    _launch_amg_kernel!(
        hierarchy,
        _amg_coarse_diagonal_solve_kernel!,
        _m(level.A),
        level.x,
        _rowptr(level.A),
        _colval(level.A),
        _nzval(level.A),
        b
    )
    return level.x
end

function _amg_device_coarse_solve_mode()
    return lowercase(get(ENV, "XCALIBRE_AMG_DEVICE_COARSE_SOLVE", ""))
end

function _amg_device_coarse_cg_maxiter(n)
    value = get(ENV, "XCALIBRE_AMG_DEVICE_COARSE_MAXITER", "")
    isempty(value) && return min(max(n, 8), 32)
    return max(1, parse(Int, value))
end

function _amg_device_coarse_cg_rtol(::Type{T}) where {T}
    value = get(ENV, "XCALIBRE_AMG_DEVICE_COARSE_RTOL", "")
    isempty(value) && return sqrt(eps(T))
    return T(parse(Float64, value))
end

function _use_device_coarse_cg(hierarchy::AMGHierarchy, solver::AMG, level::AMGLevel)
    hierarchy.backend isa CPU && return false
    _amg_device_coarse_solve_mode() == "cg" || return false
    hierarchy.is_symmetric || return false
    _m(level.A) == _n(level.A) || return false
    return true
end

function _coarse_solve_on_device_cg!(hierarchy::AMGHierarchy, level::AMGLevel, b)
    T = eltype(level.x)
    x = level.x
    r = level.tmp
    p = level.direction
    Ap = level.coarse_tmp
    _fill_amg!(hierarchy, x, zero(T))
    _copy_amg!(hierarchy, r, b)
    _copy_amg!(hierarchy, p, r)

    rr = dot(r, r)
    rr0 = rr
    tol2 = max(_amg_device_coarse_cg_rtol(T)^2 * rr0, eps(T))
    maxiter = _amg_device_coarse_cg_maxiter(length(x))
    iter = 0
    while iter < maxiter && rr > tol2
        iter += 1
        _matvec!(hierarchy, Ap, level.A, p)
        pAp = dot(p, Ap)
        if !isfinite(pAp) || pAp <= zero(T)
            break
        end
        alpha = rr / pAp
        if !isfinite(alpha)
            break
        end
        _cg_step_amg!(hierarchy, x, r, p, Ap, alpha)
        rr_new = dot(r, r)
        if !isfinite(rr_new)
            break
        elseif rr_new <= tol2
            break
        end
        beta = rr_new / rr
        if !isfinite(beta)
            break
        end
        _xpay_amg!(hierarchy, p, r, beta)
        rr = rr_new
    end
    return x
end

function _coarse_solve!(coarse_cpu::AMGCPUCoarseLevel, b)
    b === coarse_cpu.rhs || copyto!(coarse_cpu.rhs, b)
    if coarse_cpu.use_qr
        copyto!(coarse_cpu.x, coarse_cpu.qr_factor \ coarse_cpu.rhs)
    else
        ldiv!(coarse_cpu.x, coarse_cpu.lu_factor, coarse_cpu.rhs)
    end
    return coarse_cpu.x
end

# Apply the precomputed device-resident coarse direct solver — both branches stay on device:
#  - a factorization (device Cholesky/LU, CUDA ext): triangular solve (potrs/getrs)
#  - a dense inverse (generic non-CUDA GPU fallback): GEMV
_apply_coarse_direct!(x, F::Factorization, b) = (copyto!(x, b); ldiv!(F, x); x)
_apply_coarse_direct!(x, M::AbstractMatrix, b) = (mul!(x, M, b); x)

# NEW SECTION: device Krylov coarse solve (OnDeviceKrylov) — Cg/Bicgstab + Jacobi on the truncated
# coarsest level, in-place via a reused Krylov.jl workspace (no host round-trip, no per-cycle alloc).

# Thin operator over the coarsest CSR matvec kernel: only mul!/size/eltype needed by Krylov.
struct _AMGCoarseOp{RP,CV,NZ,B}
    rowptr::RP
    colval::CV
    nzval::NZ
    m::Int
    n::Int
    backend::B
    workgroup::Int
end

Base.size(A::_AMGCoarseOp) = (A.m, A.n)
Base.size(A::_AMGCoarseOp, d::Integer) = d == 1 ? A.m : d == 2 ? A.n : 1
Base.eltype(::_AMGCoarseOp{RP,CV,NZ}) where {RP,CV,NZ} = eltype(NZ)

function LinearAlgebra.mul!(y, A::_AMGCoarseOp, x)
    _launch_amg_kernel!(A.backend, A.workgroup, _amg_csr_matvec_kernel!, A.m, y, A.rowptr, A.colval, A.nzval, x)
    return y
end

mutable struct AMGKrylovCoarse{WS,OP,M,F,I,NZ}
    workspace::WS
    op::OP
    M::M             # Jacobi preconditioner Diagonal(inv_diag); applied as mul! (ldiv=false)
    rtol::F
    atol::F
    itmax::I
    nzval::NZ        # identity tag: reuse the workspace until the coarsest array is replaced
    total_iters::Int # instrumentation: summed inner iterations across cycles
    calls::Int
end

# Auto-pick Cg for a symmetric coarsest (P'AP with R=P'), Bicgstab otherwise; user override honoured.
_amg_krylov_solver_for(cs::OnDeviceKrylov, A) =
    cs.solver === nothing ? (_is_symmetric(A) ? Cg() : Bicgstab()) : cs.solver
_amg_krylov_coarse_solver(cs::OnDeviceKrylov, hierarchy) =
    _amg_krylov_solver_for(cs, hierarchy.host_levels[end].A)

function _apply_coarse_direct!(x, kb::AMGKrylovCoarse, b)
    krylov_solve!(kb.workspace, kb.op, b; M=kb.M, ldiv=false, atol=kb.atol, rtol=kb.rtol, itmax=kb.itmax)
    copyto!(x, Krylov.solution(kb.workspace))
    kb.total_iters += Krylov.iteration_count(kb.workspace)
    kb.calls += 1
    return x
end

# Build the device-resident coarse direct solver per the solver's `coarse_solve` strategy, so the
# per-cycle coarse solve has no host round-trip. Dispatch:
#  - OnHost / CPU backend → coarse_inv[]=nothing, solved by the host LU/QR round-trip path.
#  - OnDevice on a device backend → backend-dispatched build: CUDA ext factors on device; generic
#    backends use the host dense inverse adapted to device + on-device GEMV (backend-agnostic).
_build_coarse_inverse!(hierarchy::AMGHierarchy, ::OnHost) = (hierarchy.coarse_inv[] = nothing; hierarchy)

function _build_coarse_inverse!(hierarchy::AMGHierarchy, cs::OnDevice)
    hierarchy.backend isa CPU && return (hierarchy.coarse_inv[] = nothing; hierarchy)
    # Mixed precision: build a device-resident dense inverse of the coarsest level. The inverse is
    # computed in FP64 from the host coarsest (accurate factor, no FP32 pivot fragility) but stored
    # in TS for an on-device TS GEMV — keeping the coarse solve fully on device (no per-cycle host
    # sync) while the convergence reflects only the FP32 cycle-SpMV.
    _amg_mixed_precision(hierarchy) && return _build_mixed_coarse_inverse!(hierarchy, cs.max_rows)
    return _build_coarse_inverse!(hierarchy.backend, hierarchy, cs)
end

function _build_mixed_coarse_inverse!(hierarchy::AMGHierarchy, max_rows)
    Acsc = hierarchy.coarse_cpu.Acsc
    n = size(Acsc, 1)
    if n == 0 || n > max_rows
        hierarchy.coarse_inv[] = nothing  # oversized → host LU round-trip
        return hierarchy
    end
    Adense = Matrix(Acsc)
    Minv = try
        inv(Adense)
    catch err
        err isa LinearAlgebra.SingularException || rethrow(err)
        pinv(Adense)
    end
    TS = eltype(_nzval(hierarchy.levels[end].A))
    hierarchy.coarse_inv[] = adapt(hierarchy.backend, TS.(Minv))  # TS device dense inverse → GEMV
    return hierarchy
end

# Generic non-CUDA device backend: host dense inverse (pinv when singular, via use_qr) adapted to
# device, applied per-cycle as on-device GEMV. Needs no vendor direct solver.
_build_coarse_inverse!(::Any, hierarchy::AMGHierarchy, cs::OnDevice) =
    _build_host_coarse_inverse!(hierarchy, cs.max_rows)

function _build_host_coarse_inverse!(hierarchy::AMGHierarchy, max_rows)
    coarse_cpu = hierarchy.coarse_cpu
    Acsc = coarse_cpu.Acsc
    n = size(Acsc, 1)
    if n == 0 || n > max_rows
        hierarchy.coarse_inv[] = nothing
        return hierarchy
    end
    Adense = Matrix(Acsc)
    Minv = if coarse_cpu.use_qr
        pinv(Adense)
    else
        try
            inv(Adense)
        catch err
            err isa LinearAlgebra.SingularException || rethrow(err)
            pinv(Adense)
        end
    end
    # Acsc/Minv are Float64 (host direct solve); store at the cycle storage type so the on-device
    # GEMV matches the TS coarse vector (e.g. Float32 hierarchy).
    TS = eltype(_nzval(hierarchy.levels[end].A))
    hierarchy.coarse_inv[] = adapt(hierarchy.backend, TS === eltype(Minv) ? Minv : TS.(Minv))
    return hierarchy
end

# Build (or reuse) the device Krylov coarse solver. Stored in coarse_inv[] (an AMGKrylovCoarse),
# so the existing _coarse_solve!/_apply_coarse_direct! routing applies it on device. Called every
# coarse refresh: refresh the coarsest Jacobi inv_diag in place; reuse the workspace while the
# coarsest CSR array is unchanged (numeric refresh updates values in place), rebuild on a hierarchy
# rebuild (array replaced / size change). CPU backend keeps coarse_inv[]=nothing → host LU path.
function _build_coarse_inverse!(hierarchy::AMGHierarchy, cs::OnDeviceKrylov)
    if hierarchy.backend isa CPU
        hierarchy.coarse_inv[] = nothing
        return hierarchy
    end
    level = hierarchy.levels[end]
    n = _m(level.A)
    if n == 0
        hierarchy.coarse_inv[] = nothing
        return hierarchy
    end
    _refresh_diag_device!(hierarchy, level)
    existing = hierarchy.coarse_inv[]
    if existing isa AMGKrylovCoarse && existing.nzval === _nzval(level.A)
        return hierarchy
    end
    solver_choice = _amg_krylov_coarse_solver(cs, hierarchy)
    op = _AMGCoarseOp(_rowptr(level.A), _colval(level.A), _nzval(level.A), n, n, hierarchy.backend, hierarchy.workgroup)
    ws = _workspace(solver_choice, level.rhs)
    hierarchy.coarse_inv[] = AMGKrylovCoarse(ws, op, Diagonal(level.inv_diagonal), cs.rtol, cs.atol, cs.itmax, _nzval(level.A), 0, 0)
    return hierarchy
end

# NEW SECTION: device fixed-sweep smoother coarse solve (OnDeviceJacobi / OnDeviceChebyshev) —
# N Jacobi sweeps or a degree-d Chebyshev polynomial on the coarsest level, x init 0. A fixed-sweep
# smoother is a constant linear operator p(A_c)·b, so it is valid as an outer-CG preconditioner
# (both modes). Reuses the existing level smoother kernels; no host round-trip, no per-cycle alloc.
mutable struct AMGSmootherCoarse{S,H,NZ}
    smoother::S      # AMGJacobi or AMGChebyshev applied to the coarsest level
    loops::Int       # Jacobi sweeps; Chebyshev uses its degree internally (loops=1)
    hierarchy::H
    nzval::NZ        # identity tag: reuse the bundle until the coarsest array is replaced
    calls::Int
end

_coarse_smoother(cs::OnDeviceJacobi) = (AMGJacobi(omega=cs.omega), Int(cs.iterations))
_coarse_smoother(cs::OnDeviceChebyshev) =
    (AMGChebyshev(degree=cs.degree, eig_ratio=cs.eig_ratio, lambda_scale=cs.lambda_scale), 1)

function _apply_coarse_direct!(x, sc::AMGSmootherCoarse, b)
    level = sc.hierarchy.levels[end]
    _fill_amg!(sc.hierarchy, level.x, zero(eltype(level.x)))  # x init 0 → fixed linear operator
    _apply_level_smoother_impl!(sc.hierarchy, sc.smoother, level, b, sc.loops)
    sc.calls += 1
    return level.x
end

# Build/reuse the device fixed-sweep coarse solver. Stored in coarse_inv[] so the existing
# _coarse_solve!/_apply_coarse_direct! routing applies it on device. Refreshes the coarsest diag +
# lambda_max each coarse refresh (Chebyshev needs eig bounds); reuses the bundle while the coarsest
# array is unchanged. CPU backend keeps coarse_inv[]=nothing → host LU path.
function _build_coarse_inverse!(hierarchy::AMGHierarchy, cs::Union{OnDeviceJacobi,OnDeviceChebyshev})
    if hierarchy.backend isa CPU
        hierarchy.coarse_inv[] = nothing
        return hierarchy
    end
    level = hierarchy.levels[end]
    n = _m(level.A)
    if n == 0
        hierarchy.coarse_inv[] = nothing
        return hierarchy
    end
    _refresh_level_device!(hierarchy, level)  # coarsest diag + lambda_max on device
    existing = hierarchy.coarse_inv[]
    existing isa AMGSmootherCoarse && existing.nzval === _nzval(level.A) && return hierarchy
    smoother, loops = _coarse_smoother(cs)
    hierarchy.coarse_inv[] = AMGSmootherCoarse(smoother, loops, hierarchy, _nzval(level.A), 0)
    return hierarchy
end

function _coarse_solve!(hierarchy::AMGHierarchy, solver::AMG, level::AMGLevel, b)
    coarse_inv = hierarchy.coarse_inv[]
    if !(hierarchy.backend isa CPU) && coarse_inv !== nothing
        _apply_coarse_direct!(level.x, coarse_inv, b)  # on-device, no host round-trip
        return level.x
    end
    if !(hierarchy.backend isa CPU) && _is_diagonal_matrix(hierarchy.host_levels[end].A)
        return _coarse_solve_on_device!(hierarchy, level, b)
    end
    if _use_device_coarse_cg(hierarchy, solver, level)
        _coarse_solve_on_device_cg!(hierarchy, level, b)
        KernelAbstractions.synchronize(hierarchy.backend)
        return level.x
    end
    coarse_cpu = hierarchy.coarse_cpu
    _cpu_copyto!(coarse_cpu.rhs, b)
    _coarse_solve!(coarse_cpu, coarse_cpu.rhs)
    _device_copyto!(hierarchy.backend, level.x, coarse_cpu.x)
    return level.x
end

function _cycle!(hierarchy::AMGHierarchy, cycle::VCycle, solver::AMG, level_index, rhs)
    levels = hierarchy.levels
    level = levels[level_index]
    _fill_amg!(hierarchy, level.x, zero(eltype(level.x)))

    if level_index == length(levels)
        return _coarse_solve!(hierarchy, solver, level, rhs)
    end

    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.pre_sweeps)
    _residual!(hierarchy, level.rhs, level.A, level.x, rhs)

    coarse_level = levels[level_index + 1]
    _restrict!(hierarchy, coarse_level.rhs, level.R, level.rhs)
    _cycle!(hierarchy, cycle, solver, level_index + 1, coarse_level.rhs)

    _apply_coarse_correction!(hierarchy, solver, level, coarse_level)
    _apply_level_smoother!(hierarchy, solver.smoother, level, rhs, solver.post_sweeps)
    return level.x
end

# Add the prolongated coarse correction. GAMG scale_correction: scale c=P·xc by the
# energy-minimising sf=(r·c)/(c·Ac) (r=level.rhs, the post-presmoothing residual). Falls back to
# plain additive prolongation when disabled. tmp/direction are scratch (overwritten by post-smoother).
# Gated to AMGSolver: a residual-dependent sf makes the preconditioner nonlinear, which breaks the
# fixed-SPD-preconditioner assumption (Fletcher-Reeves β) of amg_cg_solve!.
function _apply_coarse_correction!(hierarchy::AMGHierarchy, solver::AMG, level::AMGLevel, coarse_level::AMGLevel)
    if !(solver.scale_correction && solver.mode isa AMGSolver)
        _prolongate_add!(hierarchy, level.x, level.P, coarse_level.x, level.tmp)
        return level.x
    end
    c = level.tmp
    Ac = level.direction
    _matvec!(hierarchy, c, level.P, coarse_level.x)
    _matvec!(hierarchy, Ac, level.A, c)
    T = eltype(level.x)
    sf = _amg_scale_factor(T(dot(level.rhs, c)), T(dot(c, Ac)))
    _amg_axpy!(hierarchy, level.x, sf, c)
    return level.x
end

# Mixed precision boundary: the working-precision residual r enters the TS cycle hierarchy, the
# TS finest correction root.x leaves as the working-precision z. Both copies are single-assignment
# (no type instability). When TS==TW (default) cycle_input is nothing → r passes through unchanged.
_amg_cycle_input(hierarchy::AMGHierarchy, r) =
    hierarchy.cycle_input[] === nothing ? r : _copy_amg!(hierarchy, hierarchy.cycle_input[], r)

_amg_mixed_precision(hierarchy::AMGHierarchy) = hierarchy.cycle_input[] !== nothing

function amg_apply_preconditioner!(z, hierarchy::AMGHierarchy, solver::AMG, r)
    root = hierarchy.levels[1]
    rin = _amg_cycle_input(hierarchy, r)
    _cycle!(hierarchy, solver.cycle, solver, 1, rin)
    _copy_amg!(hierarchy, z, root.x)
    return z
end

function _update_cycle_factor!(hierarchy::AMGHierarchy, initial_rel, final_rel, iterations, solver::AMG)
    if iterations > 0 && initial_rel > 0
        hierarchy.last_cycle_factor = (final_rel / initial_rel)^(1 / iterations)
    else
        hierarchy.last_cycle_factor = 0.0
    end
    return hierarchy.last_cycle_factor
end

function _reset_residual_history!(workspace::AMGWorkspace)
    empty!(workspace.residual_history)
    return workspace
end

function _push_residual_norm_history!(workspace::AMGWorkspace, residual_norm)
    push!(workspace.residual_history, float(residual_norm))
    return workspace
end

function amg_solve!(workspace::AMGWorkspace, hierarchy::AMGHierarchy, solver::AMG, A, b, x; itmax, atol, rtol)
    T = eltype(x)
    bnorm = max(norm(b), eps(T))
    _residual!(hierarchy, workspace.residual, A, x, b)
    _reset_residual_history!(workspace)
    rnorm = norm(workspace.residual)
    _push_residual_norm_history!(workspace, rnorm)
    ε = _amg_eps(T, atol, rtol, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel
    # Stall guard: a tiny ‖r0‖ can make ε unreachable; stop if the residual stops improving.
    best_rnorm = rnorm
    stall = 0
    stall_limit = 20
    it = 0
    while it < itmax && rnorm > ε
        it += 1
        amg_apply_preconditioner!(workspace.correction, hierarchy, solver, workspace.residual)
        KernelAbstractions.synchronize(hierarchy.backend)
        _add_amg!(hierarchy, x, workspace.correction)
        _residual!(hierarchy, workspace.residual, A, x, b)
        rnorm = norm(workspace.residual)
        _push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if rnorm < best_rnorm * (one(T) - T(1e-4))
            best_rnorm = rnorm
            stall = 0
        else
            stall += 1
            stall >= stall_limit && break
        end
    end
    workspace.iterations = it
    workspace.converged = rnorm <= ε
    workspace.last_relative_residual = rel
    _update_cycle_factor!(hierarchy, initial_rel, rel, it, solver)
    return x
end
