export AbstractSolvePrecision, FullPrecision, MixedF32, MixedF16

abstract type AbstractSolvePrecision end
abstract type AbstractMixedPrecision <: AbstractSolvePrecision end

"""
    FullPrecision()

Default `SolverSetup` precision: the linear system is solved in the mesh's float type.
"""
struct FullPrecision <: AbstractSolvePrecision end

"""
    MixedF32()

`SolverSetup` precision that solves each linear system for its correction in `Float32`. The
residual `r = b - Ax` and the update `x += d` are formed in the mesh's float type, so the solution
keeps full accuracy while the Krylov iterations, matrix and preconditioner use `Float32` storage,
which halves the memory traffic of each iteration. A single `Float32` solve cannot reduce the
residual by much more than `1e-5`; when `rtol` asks for more, the correction is repeated
(iterative refinement) until the full-precision residual meets `rtol`, so tight tolerances, e.g.
in transient runs, are met as in full precision.

Supported with the Krylov solvers (`Cg`, `Cgs`, `Bicgstab`, `Gmres`) and every serial
preconditioner, which is built on the `Float32` matrix. On distributed meshes PETSc solves the
correction with its `Float32` library (PETSc_jll loads it beside the `Float64` one), so PETSc
preconditioners (e.g. `GAMG`) are available too.
"""
struct MixedF32 <: AbstractMixedPrecision end

"""
    MixedF16()

As [`MixedF32`](@ref) with the off-diagonal matrix entries stored in `Float16`, which halves the
matrix traffic again; the Krylov iterations run in `Float32`. The system is scaled symmetrically to
unit diagonal to fit `Float16`'s range, the diagonal is kept in `Float32` and compensated so the
rounded matrix keeps the exact row sums of the scaled system, and each correction is applied with a
minimal-residual step length computed in full precision. A single correction reduces the residual
by about `1e-3` at most, so tighter `rtol` takes several corrections.

Serial meshes and the `Jacobi` preconditioner only: the scaled matrix has a unit diagonal, so
Jacobi is applied as part of the scaling at no extra cost, while factorisation preconditioners
would need their own higher-precision copy of the matrix, removing the saving. Useful on GPUs;
CPUs without native `Float16` arithmetic run it slower than `MixedF32`.
"""
struct MixedF16 <: AbstractMixedPrecision end

_storage_type(::MixedF32) = Float32
_storage_type(::MixedF16) = Float16

# smallest relative residual reduction one correction solve reaches reliably; tighter rtol takes
# several corrections (iterative refinement)
_attainable_rtol(::MixedF32) = 1e-5
_attainable_rtol(::MixedF16) = 1e-3
const MAX_REFINEMENTS = 20

# half-width storage needs the compensated, scaled correction; Krylov vectors stay Float32 since
# half-precision CG loses the sign of pᵀAp on the smooth modes of pressure matrices
_work_type(::MixedF16) = Float32

# low-precision state of a ModelEquation: Krylov workspace, operator sharing A's sparsity and the
# preconditioner (Preconditioner for MixedF32, Jacobi diagonal for MixedF16); for MixedF32 dinv is
# scratch for the scaled residual D⁻¹r and rfull, s and comp are `nothing`; c is the BiCGStab/CGS
# fallback shadow (`nothing` otherwise)
struct MixedWorkspace{PR,W,O,V,R,D,P,F,I,B,C}
    precision::PR
    krylov::W
    opA::O
    nzval::V
    r::R
    dinv::D
    P::P
    rfull::F    # full-precision scaled residual S·(b - Ax)
    s::F        # D^-1/2
    dptr::I     # position of each row's diagonal in nzval
    comp::B     # (; diag, ad): compensated Float32 diagonal and S·A·S·d (b - Ax before the solve)
    c::C
end

Krylov.iteration_count(ws::MixedWorkspace) = Krylov.iteration_count(ws.krylov)

# any serial preconditioner, built on the Float32 copy of the matrix and refreshed every solve
function MixedWorkspace(p::MixedF32, solver::AbstractLinearSolver, preconditioner::PreconditionerType, A, b, dptr)
    nzval = similar(_nzval(A), Float32)
    nzval .= _nzval(A)
    r = _krylov_vector(similar(b, Float32))
    opA = _lowprecision_operator(A, nzval)
    P = Preconditioner{typeof(preconditioner)}(opA)
    krylov = _workspace(solver, r)
    MixedWorkspace(p, krylov, opA, nzval, r, similar(r), P, nothing, nothing, dptr,
        nothing, _shadow_buffer(krylov, r))
end

function MixedWorkspace(p::MixedF16, solver::AbstractLinearSolver, ::Jacobi, A, b, dptr)
    T = _work_type(p)
    nzval = similar(_nzval(A), _storage_type(p))
    diag = similar(b, Float32)
    r = _krylov_vector(similar(b, T))
    dinv = similar(b, T)
    opA = csr_operator(_rowptr(A), _colval(A), nzval, diag, similar(diag, T), Int(_m(A)))
    krylov = _workspace(solver, r)
    MixedWorkspace(p, krylov, opA, nzval, r, dinv, diagonal_operator(dinv), similar(b), similar(b),
        dptr, (; diag, ad=similar(b)), _shadow_buffer(krylov, r))
end

_shadow_buffer(::_ShadowWorkspace, r) = similar(r)
_shadow_buffer(_, r) = nothing

# BiCGStab's shadow is Krylov.jl's initial residual D⁻¹r (d starts from 0), kept in dinv until the solve
_mixed_shadow(ws::MixedWorkspace{MixedF32,<:BicgstabWorkspace}) = (copyto!(ws.c, ws.dinv); (; c=ws.c))
_mixed_shadow(ws::MixedWorkspace{MixedF16,<:BicgstabWorkspace}) = (mul!(ws.c, ws.P, ws.r); (; c=ws.c))
_mixed_shadow(ws) = (;)

# the correction d starts from 0, so a shadow-vector breakdown is retried in `_mixed_correction!`
# with the dense shadow; a second breakdown leaves the field unchanged
function _correction_broke_down(ws, itmax)
    _broke_down(ws.krylov, itmax) || return false
    @warn "$(nameof(typeof(ws.krylov))) broke down twice; the field keeps its previous values" maxlog=10
    return true
end

function _diagonal_positions(A)
    rowptr = _rowptr(A)
    n = Int(_m(A))
    dptr = similar(rowptr, n)
    kernel! = _diagonal_positions!(_setup(get_backend(rowptr), 256, n)...)
    kernel!(dptr, rowptr, _colval(A))
    dptr
end

@kernel function _diagonal_positions!(dptr, @Const(rowptr), @Const(colval))
    i = @index(Global)
    @inbounds for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        colval[nzi] == i && (dptr[i] = nzi)
    end
end


_lowprecision_operator(A::SparseXCSR{Bi}, nzval) where Bi = begin
    Ap = parent(A)
    SparseXCSR(SparseMatrixCSR{Bi}(Ap.m, Ap.n, Ap.rowptr, Ap.colval, nzval))
end
# backends whose sparse libraries lack T use a KernelAbstractions SpMV
_lowprecision_operator(A, nzval) = csr_operator(_rowptr(A), _colval(A), nzval, Int(_m(A)))

function csr_operator(rowptr, colval, nzval::AbstractVector{T}, n) where T
    backend = get_backend(nzval)
    apply! = (res, v, α, β) -> begin
        kernel! = _csr_mul!(_setup(backend, 256, n)...)
        kernel!(res, rowptr, colval, nzval, v, α, β)
        res
    end
    LinearOperator{T,typeof(nzval)}(n, n, false, false, apply!, apply!, apply!)
end

# off-diagonals in nzval (zero on the diagonal) plus a separate diagonal, on vectors like proto
function csr_operator(rowptr, colval, nzval, diag, proto::AbstractVector{T}, n) where T
    backend = get_backend(diag)
    apply! = (res, v, α, β) -> begin
        kernel! = _csr_diag_mul!(_setup(backend, 256, n)...)
        kernel!(res, rowptr, colval, nzval, diag, v, α, β)
        res
    end
    LinearOperator{T,typeof(proto)}(n, n, false, false, apply!, apply!, apply!)
end

@kernel function _csr_mul!(res, @Const(rowptr), @Const(colval), @Const(nzval), @Const(v), α, β)
    i = @index(Global)
    TA = _acc_type(eltype(res))
    acc = zero(TA)
    @inbounds begin
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            acc += TA(nzval[nzi])*TA(v[colval[nzi]])
        end
        # res may hold garbage when β is zero, so the two cases cannot be merged
        res[i] = iszero(β) ? convert(eltype(res), α*acc) : convert(eltype(res), α*acc + β*res[i])
    end
end

@kernel function _csr_diag_mul!(
    res, @Const(rowptr), @Const(colval), @Const(nzval), @Const(diag), @Const(v), α, β)
    i = @index(Global)
    TA = _acc_type(eltype(res))
    @inbounds begin
        acc = TA(diag[i])*TA(v[i])
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            acc += TA(nzval[nzi])*TA(v[colval[nzi]])
        end
        res[i] = iszero(β) ? convert(eltype(res), α*acc) : convert(eltype(res), α*acc + β*res[i])
    end
end

# NEW SECTION: correction solve

# x += d per correction until ‖D⁻¹(b - Ax)‖ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖, one unless rtol is below what the
# low precision attains; returns the total Krylov iterations, whether the target was met, and why not
function mixed_solve!(ws::MixedWorkspace, A, b, values, setup, config, mesh)
    (; atol, rtol) = setup
    inner_rtol = max(rtol, _attainable_rtol(ws.precision))
    if inner_rtol == rtol && _trusts_inner_stop(ws.precision)
        its, _ = _mixed_correction!(ws, A, b, values, inner_rtol, nothing, setup, config, mesh)
        # a zero residual returns before the Krylov solve
        converged = iszero(its) || _krylov_converged(ws.krylov)
        return its, converged, ws.krylov.stats.status
    end
    iterations, rnorm = _mixed_correction!(ws, A, b, values, inner_rtol, 0.0, setup, config, mesh)
    target = _stop_target(atol, rtol, rnorm)
    for _ ∈ 2:MAX_REFINEMENTS
        its, rnorm = _mixed_correction!(ws, A, b, values, inner_rtol, target, setup, config, mesh)
        iterations += its
        rnorm ≤ target && return iterations, true, ""
    end
    # the last correction is checked without solving again
    _, rnorm = _mixed_correction!(ws, A, b, values, inner_rtol, target, setup, config, mesh; solve=false)
    iterations, rnorm ≤ target, _refinement_reason()
end

# MixedF16's Krylov.jl solve monitors the S-scaled residual, which can differ widely from D⁻¹(b - Ax), so
# its result is always checked by a refinement pass
_trusts_inner_stop(::MixedF32) = true
_trusts_inner_stop(::MixedF16) = false

_refinement_reason() = "iterative refinement stopped after $MAX_REFINEMENTS corrections"

# solves A·d = b - Ax from d = 0 and adds d unless ‖D⁻¹(b - Ax)‖ meets target (`nothing`: the setup's
# own); returns the Krylov count and ‖D⁻¹(b - Ax)‖ before the correction. Later corrections reduce only to the
# target: the outer convergence check reads the post-solve residual (overshooting rtol stops a run)
_correction_tol(atol, rtol, target, rnorm) =
    isnothing(target) || iszero(target) ? (atol, rtol) : (zero(atol), max(rtol, target/rnorm))

# no correction once ‖D⁻¹(b - Ax)‖ meets the target; the first one checks the setup's own (x₀ converged)
_skips_correction(rnorm, target, setup) =
    rnorm ≤ (isnothing(target) || iszero(target) ? _stop_target(setup.atol, setup.rtol, rnorm) : target)

function _mixed_correction!(ws::MixedWorkspace{MixedF32}, A, b, values, rtol, target, setup, config, mesh;
        solve=true)
    (; itmax, atol) = setup
    (; backend, workgroup) = config.hardware
    n = length(values)
    kernel! = _sized(_mixed_residual!, backend, workgroup, n)
    kernel!(ws.r, _data(ws.dinv), ws.nzval, _rowptr(A), _colval(A), _nzval(A), ws.dptr, values, b)
    rnorm = Float64(norm(ws.dinv))
    (!solve || _skips_correction(rnorm, target, setup)) && return 0, rnorm
    # A is unchanged by later corrections of the same solve
    (isnothing(target) || iszero(target)) && update_preconditioner!(ws.P, mesh, config)
    atol_c, rtol_c = Float32.(_correction_tol(atol, rtol, target, rnorm))
    scale = (; nzval=ws.nzval, dptr=ws.dptr, tmp=ws.dinv)
    kw = (; _stop_kwargs(ws.krylov, ws.P, scale, atol_c, rtol_c, Float32(rnorm))..., itmax=itmax, history=false)
    krylov_solve!(ws.krylov, ws.opA, ws.r; _mixed_shadow(ws)..., kw...)
    _broke_down(ws.krylov, itmax) && krylov_solve!(ws.krylov, ws.opA, ws.r; c=_fallback_shadow!(ws.c), kw...)
    _correction_broke_down(ws, itmax) && return Krylov.iteration_count(ws.krylov), rnorm
    kernel! = _sized(_add_correction!, backend, workgroup, n)
    kernel!(values, ws.krylov.x, one(eltype(values)))
    Krylov.iteration_count(ws.krylov), rnorm
end

# scaling to unit diagonal fits Float16's range; half-width rounding perturbs A by more than the
# smallest eigenvalues of a pressure matrix, so the diagonal is compensated to keep A's action on
# constant fields, and the minimal-residual step keeps the full-precision residual from growing
function _mixed_correction!(ws::MixedWorkspace{MixedF16}, A, b, values, rtol, target, setup, config, mesh;
        solve=true)
    (; itmax, atol) = setup
    hardware = config.hardware
    (; backend, workgroup) = hardware
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    T = _work_type(ws.precision)
    F = eltype(values)
    n = length(values)
    kernel! = _sized(_inv_sqrt_diagonal!, backend, workgroup, n)
    kernel!(ws.s, ws.dptr, nzval)
    kernel! = _sized(_mixed_residual_compensated!, backend, workgroup, n)
    kernel!(ws.rfull, ws.comp.ad, ws.nzval, ws.comp.diag, ws.dinv, rowptr, colval, nzval, values, b, ws.s)
    # tolerances apply to D⁻¹(b - Ax), not the scaled residual
    rnorm = norm(ws.comp.ad)
    snorm = norm(ws.rfull)
    (!solve || iszero(snorm) || _skips_correction(rnorm, target, setup)) && return 0, rnorm
    # unit 2-norm: Krylov's dot products return T and n·max² overflows Float16
    kernel! = _sized(_scale_cast!, backend, workgroup, n)
    kernel!(ws.r, ws.rfull, inv(snorm))
    # Krylov.jl sees S·(b - Ax)/‖S·(b - Ax)‖, so the target becomes a relative reduction
    atol_c, rtol_c = _correction_tol(atol, rtol, target, rnorm)
    kw = (; _stop_kwargs(ws.krylov, ws.P, nothing, T(atol_c/rnorm), T(rtol_c), one(T))..., itmax=itmax,
        history=false)
    krylov_solve!(ws.krylov, ws.opA, ws.r; _mixed_shadow(ws)..., kw...)
    _broke_down(ws.krylov, itmax) && krylov_solve!(ws.krylov, ws.opA, ws.r; c=_fallback_shadow!(ws.c), kw...)
    _correction_broke_down(ws, itmax) && return Krylov.iteration_count(ws.krylov), rnorm
    d = ws.krylov.x
    ω = _step_length(ws, A, d, hardware)
    kernel! = _sized(_add_scaled_correction!, backend, workgroup, n)
    kernel!(values, d, ws.s, F(ω))
    Krylov.iteration_count(ws.krylov), rnorm
end

# ω minimising ‖S·(b - Ax) - ω·S·A·S·d‖₂
function _step_length(ws, A, d, hardware)
    (; backend, workgroup) = hardware
    (; ad) = ws.comp
    kernel! = _sized(_correction_product!, backend, workgroup, length(ad))
    kernel!(ad, _rowptr(A), _colval(A), _nzval(A), ws.s, d)
    den = dot(ad, ad)
    iszero(den) ? zero(den) : dot(ws.rfull, ad)/den
end

mixed_solve!(ws, A, b, values, setup, config, mesh) =
    throw(ArgumentError("mixed precision is not yet supported by this solver or equation"))

# casts A while forming r = b - Ax and w = D⁻¹r in full precision
@kernel function _mixed_residual!(
    r, w, nzval_lo, @Const(rowptr), @Const(colval), @Const(nzval), @Const(dptr), @Const(x), @Const(b))
    i = @index(Global)
    Ax = zero(eltype(nzval))
    @inbounds begin
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            a = nzval[nzi]
            c = colval[nzi]
            nzval_lo[nzi] = convert(eltype(nzval_lo), a)
            Ax += a*x[c]
        end
        ri = b[i] - Ax
        r[i] = ri
        w[i] = ri/abs(nzval[dptr[i]])
    end
end

# any S is exact once applied consistently, so Float32 avoids slow double-precision sqrt on GPUs
@kernel function _inv_sqrt_diagonal!(s, @Const(dptr), @Const(nzval))
    i = @index(Global)
    @inbounds s[i] = inv(sqrt(abs(Float32(nzval[dptr[i]]))))
end

# rounds the scaled off-diagonals and sets the diagonal so the rounded matrix maps S^-1·1 (a
# constant field) exactly as S·A·S does; diagonal-only, so symmetry is kept. ru = D⁻¹(b - Ax)
@kernel function _mixed_residual_compensated!(r, ru, nzval_lo, diag, dinv,
        @Const(rowptr), @Const(colval), @Const(nzval), @Const(x), @Const(b), @Const(s))
    i = @index(Global)
    F = eltype(nzval)
    Ax = zero(F)
    rowsum = zero(F)
    offsum = zero(F)
    @inbounds begin
        si = s[i]
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            a = nzval[nzi]
            c = colval[nzi]
            Ax += a*x[c]
            rowsum += a
            if c == i
                nzval_lo[nzi] = zero(eltype(nzval_lo))
            else
                l = convert(eltype(nzval_lo), si*a*s[c])
                nzval_lo[nzi] = l
                offsum += F(l)/s[c]
            end
        end
        dii = si*(si*rowsum - offsum)
        diag[i] = convert(eltype(diag), dii)
        dinv[i] = convert(eltype(dinv), inv(abs(dii)))
        r[i] = si*(b[i] - Ax)
        ru[i] = si*r[i]
    end
end

@kernel function _scale_cast!(y, @Const(x), sr)
    i = @index(Global)
    @inbounds y[i] = convert(eltype(y), sr*x[i])
end

# ad = S·A·S·d in full precision
@kernel function _correction_product!(ad, @Const(rowptr), @Const(colval), @Const(nzval), @Const(s), @Const(d))
    i = @index(Global)
    F = eltype(ad)
    acc = zero(F)
    @inbounds begin
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            c = colval[nzi]
            acc += nzval[nzi]*s[c]*F(d[c])
        end
        ad[i] = s[i]*acc
    end
end

@kernel function _add_correction!(values, @Const(d), α)
    i = @index(Global)
    @inbounds values[i] += α*convert(eltype(values), d[i])
end

@kernel function _add_scaled_correction!(values, @Const(d), @Const(s), α)
    i = @index(Global)
    @inbounds values[i] += α*s[i]*convert(eltype(values), d[i])
end
