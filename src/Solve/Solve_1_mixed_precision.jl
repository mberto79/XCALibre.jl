export AbstractSolvePrecision, FullPrecision, MixedPrecision, BFloat16

abstract type AbstractSolvePrecision end

"""
    FullPrecision()

Default `SolverSetup` precision: the linear system is solved in the mesh's float type.
"""
struct FullPrecision <: AbstractSolvePrecision end

"""
    MixedPrecision(T=Float32)

`SolverSetup` precision that solves each linear system for its correction in the low-precision
type `T` (`BFloat16`, `Float16` or `Float32`). The residual `r = b - Ax` and the update `x += d`
are formed in the mesh's float type, so the outer (e.g. SIMPLE) iteration keeps full accuracy
while the Krylov iterations, matrix and preconditioner use `T` storage. On the CPU, dot products
of half types accumulate in `Float32`; `Float16` systems are scaled symmetrically to unit
diagonal internally to fit its range.

Each solve is less accurate than in full precision, so the reported (post-solve) residuals are
higher and outer convergence is slower, most of all for `BFloat16`; a `convergence` target tuned
for `FullPrecision` may not be reached.

Supported with the Krylov solvers (`Cg`, `Cgs`, `Bicgstab`, `Gmres`); serial meshes use the
`Jacobi` preconditioner. On distributed meshes PETSc solves the correction with its `T` library while the
fields stay in the mesh's float type, so `T` is `Float32` (PETSc_jll loads its Float32 library beside
the Float64 one) and PETSc preconditioners (e.g. `GAMG`) are available too.
`Float32` gives the best accuracy per unit of speed-up.
"""
struct MixedPrecision{T<:AbstractFloat} <: AbstractSolvePrecision end
MixedPrecision(::Type{T}=Float32) where T<:AbstractFloat = MixedPrecision{T}()

# low-precision state of a ModelEquation: Krylov workspace, operator sharing A's sparsity, Jacobi;
# rfull and s (D^-1/2) serve the scaled path of types with a narrow exponent range
struct MixedWorkspace{W,O,V,R,D,P,F}
    krylov::W
    opA::O
    nzval::V
    r::R
    dinv::D
    P::P
    rfull::F
    s::F
end

Krylov.iteration_count(ws::MixedWorkspace) = Krylov.iteration_count(ws.krylov)

function MixedWorkspace(::Type{T}, solver::AbstractLinearSolver, A, b) where T
    nzval = similar(_nzval(A), T)
    r = _krylov_vector(similar(b, T))
    dinv = similar(b, T)
    MixedWorkspace(_workspace(solver, r), _lowprecision_operator(A, nzval), nzval, r, dinv,
        diagonal_operator(dinv), similar(b), similar(b))
end

_lowprecision_operator(A::SparseXCSR{Bi}, nzval) where Bi = begin
    Ap = parent(A)
    SparseXCSR(SparseMatrixCSR{Bi}(Ap.m, Ap.n, Ap.rowptr, Ap.colval, nzval))
end
# backends whose sparse libraries lack T (e.g. BFloat16 in cuSPARSE) use a KernelAbstractions SpMV
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

# NEW SECTION: correction solve

# Float16's exponent range under/overflows FVM coefficients and their inverses
_needs_scaling(::Type{T}) where T = floatmax(T) < floatmax(Float32)

# solves A·d = b - Ax for d from d = 0 and adds α·d to values
function _mixed_correction!(ws, A, b, values, α, setup, hardware)
    (; itmax, atol, rtol) = setup
    (; backend, workgroup) = hardware
    TL, F = eltype(ws.nzval), eltype(values)
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    scaled = _needs_scaling(TL)
    n = length(values)
    sr = one(F)
    if scaled
        kernel! = _sized(_inv_sqrt_diagonal!, backend, workgroup, n)
        kernel!(ws.s, rowptr, colval, nzval)
        kernel! = _sized(_mixed_residual_scaled!, backend, workgroup, n)
        kernel!(ws.rfull, ws.nzval, ws.dinv, rowptr, colval, nzval, values, b, ws.s)
        # unit 2-norm, not unit max: Krylov's dot products return T and n·max² overflows Float16
        rnorm = norm(ws.rfull)
        sr = iszero(rnorm) ? one(F) : inv(rnorm)
        kernel! = _sized(_scale_cast!, backend, workgroup, n)
        kernel!(ws.r, ws.rfull, sr)
    else
        kernel! = _sized(_mixed_residual!, backend, workgroup, n)
        kernel!(ws.r, ws.nzval, ws.dinv, rowptr, colval, nzval, values, b)
    end

    krylov_solve!(ws.krylov, ws.opA, ws.r;
        M=ws.P, itmax=itmax, atol=TL(atol*sr), rtol=TL(rtol), ldiv=false, history=false)

    d = ws.krylov.x
    if scaled
        kernel! = _sized(_add_scaled_correction!, backend, workgroup, n)
        kernel!(values, d, ws.s, F(α)/sr)
    else
        kernel! = _sized(_add_correction!, backend, workgroup, n)
        kernel!(values, d, F(α))
    end
    Krylov.iteration_count(ws.krylov)
end

# casts A and its Jacobi inverse diagonal while forming r = b - Ax in full precision
@kernel function _mixed_residual!(
    r, nzval_lo, dinv, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x), @Const(b))
    i = @index(Global)
    Ax = zero(eltype(nzval))
    @inbounds begin
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            a = nzval[nzi]
            c = colval[nzi]
            nzval_lo[nzi] = convert(eltype(nzval_lo), a)
            c == i && (dinv[i] = convert(eltype(dinv), inv(abs(a))))
            Ax += a*x[c]
        end
        r[i] = b[i] - Ax
    end
end

@kernel function _inv_sqrt_diagonal!(s, @Const(rowptr), @Const(colval), @Const(nzval))
    i = @index(Global)
    @inbounds for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        colval[nzi] == i && (s[i] = inv(sqrt(abs(nzval[nzi]))))
    end
end

# D^-1/2 A D^-1/2 has unit diagonal and off-diagonals of order one, keeping symmetry for Cg
@kernel function _mixed_residual_scaled!(
    r, nzval_lo, dinv, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x), @Const(b), @Const(s))
    i = @index(Global)
    Ax = zero(eltype(nzval))
    @inbounds begin
        si = s[i]
        for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
            a = nzval[nzi]
            c = colval[nzi]
            nzval_lo[nzi] = convert(eltype(nzval_lo), si*a*s[c])
            Ax += a*x[c]
        end
        r[i] = si*(b[i] - Ax)
        dinv[i] = one(eltype(dinv))
    end
end

@kernel function _scale_cast!(y, @Const(x), s)
    i = @index(Global)
    @inbounds y[i] = convert(eltype(y), s*x[i])
end

@kernel function _add_correction!(values, @Const(d), α)
    i = @index(Global)
    @inbounds values[i] += α*convert(eltype(values), d[i])
end

@kernel function _add_scaled_correction!(values, @Const(d), @Const(s), α)
    i = @index(Global)
    @inbounds values[i] += α*s[i]*convert(eltype(values), d[i])
end
