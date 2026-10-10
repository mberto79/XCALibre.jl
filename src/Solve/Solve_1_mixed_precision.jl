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
residual `r = b - Ax` and the update `x += d` are formed in the mesh's float type, so the outer
(e.g. SIMPLE) iteration keeps full accuracy while the Krylov iterations, matrix and preconditioner
use `Float32` storage. Supported with the Krylov solvers (`Cg`, `Cgs`, `Bicgstab`, `Gmres`); serial
meshes use the `Jacobi` preconditioner. On distributed meshes PETSc solves the correction with its
`Float32` library (PETSc_jll loads it beside the `Float64` one), so PETSc preconditioners (e.g.
`GAMG`) are available too. `MixedF32` gives the best accuracy per unit of speed-up.
"""
struct MixedF32 <: AbstractMixedPrecision end

"""
    MixedF16()

As [`MixedF32`](@ref) with the off-diagonal matrix entries stored in `Float16`; the Krylov
iterations run in `Float32`. The system is scaled symmetrically to unit diagonal to fit `Float16`'s
range, the diagonal is kept in `Float32` and compensated so the rounded matrix keeps the exact row
sums of the scaled system, and each correction is applied with a minimal-residual step length
computed in full precision. Serial meshes only.
"""
struct MixedF16 <: AbstractMixedPrecision end

_storage_type(::MixedF32) = Float32
_storage_type(::MixedF16) = Float16

# half-width storage needs the compensated, scaled correction; Krylov vectors stay Float32 since
# half-precision CG loses the sign of pᵀAp on the smooth modes of pressure matrices
_work_type(::MixedF16) = Float32

# low-precision state of a ModelEquation: Krylov workspace, operator sharing A's sparsity and the
# preconditioner (a Preconditioner for MixedF32, Jacobi's diagonal operator for MixedF16). dinv,
# rfull, s, dptr and comp are `nothing` for MixedF32.
struct MixedWorkspace{PR,W,O,V,R,D,P,F,I,B}
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
    comp::B     # (; diag, ad): compensated Float32 diagonal and S·A·S·d
end

Krylov.iteration_count(ws::MixedWorkspace) = Krylov.iteration_count(ws.krylov)

# a mixed-precision workspace holds its own preconditioner, so the equation keeps an inert one
struct MixedPrecisionSolve <: PreconditionerType end
update_preconditioner!(::Preconditioner{MixedPrecisionSolve}, mesh, config) = nothing

# any serial preconditioner, built on the Float32 copy of the matrix and refreshed every solve
function MixedWorkspace(p::MixedF32, solver::AbstractLinearSolver, preconditioner::PreconditionerType, A, b)
    nzval = similar(_nzval(A), Float32)
    nzval .= _nzval(A)
    r = _krylov_vector(similar(b, Float32))
    opA = _lowprecision_operator(A, nzval)
    P = Preconditioner{typeof(preconditioner)}(opA)
    MixedWorkspace(p, _workspace(solver, r), opA, nzval, r, nothing, P, nothing, nothing, nothing, nothing)
end

function MixedWorkspace(p::MixedF16, solver::AbstractLinearSolver, ::Jacobi, A, b)
    T = _work_type(p)
    nzval = similar(_nzval(A), _storage_type(p))
    diag = similar(b, Float32)
    r = _krylov_vector(similar(b, T))
    dinv = similar(b, T)
    opA = csr_operator(_rowptr(A), _colval(A), nzval, diag, similar(diag, T), Int(_m(A)))
    MixedWorkspace(p, _workspace(solver, r), opA, nzval, r, dinv,
        diagonal_operator(dinv), similar(b), similar(b), _diagonal_positions(A), (; diag, ad=similar(b)))
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

# solves A·d = b - Ax for d from d = 0 and adds α·d to values; returns the largest Krylov count
function _mixed_correction!(ws::MixedWorkspace{MixedF32}, A, b, values, α, setup, config, mesh)
    (; itmax, atol, rtol) = setup
    (; backend, workgroup) = config.hardware
    n = length(values)
    kernel! = _sized(_mixed_residual!, backend, workgroup, n)
    kernel!(ws.r, ws.nzval, _rowptr(A), _colval(A), _nzval(A), values, b)
    update_preconditioner!(ws.P, mesh, config)
    krylov_solve!(ws.krylov, ws.opA, ws.r;
        M=ws.P.P, itmax=itmax, atol=Float32(atol), rtol=Float32(rtol), ldiv=is_ldiv(ws.P), history=false)
    kernel! = _sized(_add_correction!, backend, workgroup, n)
    kernel!(values, ws.krylov.x, eltype(values)(α))
    Krylov.iteration_count(ws.krylov)
end

# scaling to unit diagonal fits Float16's range; half-width rounding perturbs A by more than the
# smallest eigenvalues of a pressure matrix, so the diagonal is compensated to keep A's action on
# constant fields, and the minimal-residual step keeps the full-precision residual from growing
function _mixed_correction!(ws::MixedWorkspace{MixedF16}, A, b, values, α, setup, config, mesh)
    (; itmax, atol, rtol) = setup
    hardware = config.hardware
    (; backend, workgroup) = hardware
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    T = _work_type(ws.precision)
    F = eltype(values)
    n = length(values)
    kernel! = _sized(_inv_sqrt_diagonal!, backend, workgroup, n)
    kernel!(ws.s, ws.dptr, nzval)
    kernel! = _sized(_mixed_residual_compensated!, backend, workgroup, n)
    kernel!(ws.rfull, ws.nzval, ws.comp.diag, ws.dinv, rowptr, colval, nzval, values, b, ws.s)
    # unit 2-norm: Krylov's dot products return T and n·max² overflows Float16
    rnorm = norm(ws.rfull)
    iszero(rnorm) && return 0
    sr = inv(rnorm)
    kernel! = _sized(_scale_cast!, backend, workgroup, n)
    kernel!(ws.r, ws.rfull, sr)
    krylov_solve!(ws.krylov, ws.opA, ws.r;
        M=ws.P, itmax=itmax, atol=T(atol*sr), rtol=T(rtol), ldiv=false, history=false)
    d = ws.krylov.x
    ω = _step_length(ws, A, d, hardware)
    kernel! = _sized(_add_scaled_correction!, backend, workgroup, n)
    kernel!(values, d, ws.s, F(α)*ω)
    Krylov.iteration_count(ws.krylov)
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

_mixed_correction!(ws, A, b, values, α, setup, config, mesh) =
    throw(ArgumentError("mixed precision is not yet supported by this solver or equation"))

# casts A while forming r = b - Ax in full precision
@kernel function _mixed_residual!(
    r, nzval_lo, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x), @Const(b))
    i = @index(Global)
    Ax = zero(eltype(nzval))
    @inbounds begin
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            a = nzval[nzi]
            c = colval[nzi]
            nzval_lo[nzi] = convert(eltype(nzval_lo), a)
            Ax += a*x[c]
        end
        r[i] = b[i] - Ax
    end
end

# any S is exact once applied consistently, so Float32 avoids slow double-precision sqrt on GPUs
@kernel function _inv_sqrt_diagonal!(s, @Const(dptr), @Const(nzval))
    i = @index(Global)
    @inbounds s[i] = inv(sqrt(abs(Float32(nzval[dptr[i]]))))
end

# rounds the scaled off-diagonals and sets the diagonal so the rounded matrix maps S^-1·1 (a
# constant field) exactly as S·A·S does; diagonal-only, so symmetry is kept
@kernel function _mixed_residual_compensated!(r, nzval_lo, diag, dinv,
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
