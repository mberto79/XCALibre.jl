export _workspace
export Cg, Cgs, Bicgstab, Gmres

abstract type AbstractLinearSolver end

struct Cg <: AbstractLinearSolver end
struct Cgs <: AbstractLinearSolver end
struct Bicgstab <: AbstractLinearSolver end
struct Gmres <: AbstractLinearSolver end

# Krylov.jl workspace constructors
_workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))
_workspace(::Cgs, b) = CgsWorkspace(KrylovConstructor(b))
_workspace(::Bicgstab, b) = BicgstabWorkspace(KrylovConstructor(b))
_workspace(::Gmres, b) = GmresWorkspace(KrylovConstructor(b); memory=GMRES_RESTART)
# equation workspaces hold XVector storage on the CPU so Krylov's vector work runs on Julia's threads
_workspace(solver::AbstractLinearSolver, b, ::Type) = _workspace(solver, _krylov_vector(b))
_krylov_vector(v::Vector) = XVector(v)
_krylov_vector(v) = v
# Krylov requires b and x0 in the workspace's storage type
_like_workspace(::XVector, v::Vector) = XVector(v)
_like_workspace(_, v) = v
_index_type(A) = eltype(_rowptr(A))

# NEW SECTION: stopping rule

# Krylov.jl and PETSc both restart GMRES after this many iterations
const GMRES_RESTART = 30

# every solve path stops when ‖D⁻¹(b - Ax)‖₂ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖₂, D = |diag(A)|: the size of a
# Jacobi correction, in the field's units, whatever the preconditioner
_stop_target(atol, rtol, r0) = atol + rtol*r0

# Jacobi-like preconditioners apply D⁻¹ themselves (a Preconditioner{Jacobi}, or MixedF16's diagonal)
const _JacobiLike = Union{Preconditioner{Jacobi}, LinearOperator}
_prec_op(P::Preconditioner) = P.P
_prec_op(P) = P
_prec_ldiv(P::Preconditioner) = is_ldiv(P)
_prec_ldiv(P) = false

# Krylov.jl monitors M·(b - Ax), so M = D⁻¹ (left) gives the stopping norm; the right preconditioner then
# approximates (D⁻¹A)⁻¹ = P∘D, the identity for Jacobi. CG must keep P on the left for symmetry: its own
# test is disabled and a callback checks ‖D⁻¹r‖. scale = (; nzval, dptr, tmp) reads D from A
_stop_kwargs(::CgWorkspace, P, scale, atol, rtol, r0) = (; M=_prec_op(P), ldiv=_prec_ldiv(P),
    atol=zero(atol), rtol=zero(rtol), callback=_ScaledNormStop(_stop_target(atol, rtol, r0)^2, _cg_scale(P, scale)))
_stop_kwargs(ws::GmresWorkspace, P, scale, atol, rtol, r0) = (; _sides(P, scale)..., atol, rtol, restart=true)
_stop_kwargs(ws, P, scale, atol, rtol, r0) = (; _sides(P, scale)..., atol, rtol)

_sides(P::_JacobiLike, scale) = (; M=_prec_op(P), ldiv=false)
_sides(P, scale) = (; M=_DiagScale(scale.nzval, scale.dptr),
    N=_RightPrec(_prec_op(P), _prec_ldiv(P), scale.nzval, scale.dptr, scale.tmp), ldiv=false)

# Jacobi CG already holds z = D⁻¹r
_cg_scale(P::_JacobiLike, scale) = nothing
_cg_scale(P, scale) = scale

# ‖D⁻¹r‖² by a dot product: GPU BLAS nrm2 is several times slower than dot
struct _ScaledNormStop{T,S}
    target²::T
    scale::S
end
(s::_ScaledNormStop{T,Nothing})(ws) where T = Krylov.kdotr(length(ws.z), ws.z, ws.z) ≤ s.target²
function (s::_ScaledNormStop)(ws)
    (; tmp, nzval, dptr) = s.scale
    _scale_rows!(tmp, ws.r, nzval, dptr, Val(true))
    Krylov.kdotr(length(tmp), tmp, tmp) ≤ s.target²
end

# y = D⁻¹x; the ldiv flag of a Krylov call is shared by M and N, so both apply the same way
struct _DiagScale{V,I}
    nzval::V
    dptr::I
end
LinearAlgebra.mul!(y, S::_DiagScale, x) = _scale_rows!(y, x, S.nzval, S.dptr, Val(true))
LinearAlgebra.ldiv!(y, S::_DiagScale, x) = mul!(y, S, x)

# y = P(Dx)
struct _RightPrec{P,V,I,T}
    P::P
    ldiv::Bool
    nzval::V
    dptr::I
    tmp::T
end
function LinearAlgebra.mul!(y, N::_RightPrec, x)
    _scale_rows!(N.tmp, x, N.nzval, N.dptr, Val(false))
    N.ldiv ? ldiv!(y, N.P, N.tmp) : mul!(y, N.P, N.tmp)
end
LinearAlgebra.ldiv!(y, N::_RightPrec, x) = mul!(y, N, x)

_data(v::XVector) = parent(v)
_data(v) = v

# ‖D⁻¹r‖₂ for d = |diag(A)| (`nothing`: ‖r‖₂)
_scaled_norm(r, ::Nothing) = norm(r)
_scaled_norm(r, d) = sqrt(mapreduce((ri, di) -> abs2(ri/di), +, _data(r), d))

# y = x./|a_ii| (Val(true)) or x.*|a_ii|; CPU rows in the Krylov vectors' static chunks
_scale_rows!(y, x, nzval, dptr, inv) = (_scale_rows_data!(_data(y), _data(x), nzval, dptr, inv); y)
function _scale_rows_data!(y::Vector, x, nzval, dptr, inv)
    _foreach_chunk(length(y)) do r
        @inbounds for i ∈ r
            y[i] = _scale_row(x[i], nzval[dptr[i]], inv)
        end
    end
end
function _scale_rows_data!(y, x, nzval, dptr, inv)
    kernel! = _scale_rows_kernel!(_setup(get_backend(y), 256, length(y))...)
    kernel!(y, x, nzval, dptr, inv)
end
@kernel function _scale_rows_kernel!(y, @Const(x), @Const(nzval), @Const(dptr), inv)
    i = @index(Global)
    @inbounds y[i] = _scale_row(x[i], nzval[dptr[i]], inv)
end
@inline _scale_row(x, a, ::Val{true}) = convert(typeof(x), x/abs(a))
@inline _scale_row(x, a, ::Val{false}) = convert(typeof(x), x*abs(a))

# a callback exit is CG meeting the target
_krylov_converged(ws) = Krylov.issolved(ws)
_krylov_converged(ws::CgWorkspace) = Krylov.issolved(ws) || ws.stats.status == "user-requested exit"

# NEW SECTION: shadow-vector breakdown

# BiCGStab/CGS take ⟨c, r⟩/⟨c, Ap⟩: 0/0 once r ⟂ shadow c (b - Ax₀ in few cells, one-way coupling);
# Krylov.jl flags it after the 0/0 step reached x, so callers re-solve with `_fallback_shadow!`
const _ShadowWorkspace = Union{BicgstabWorkspace, CgsWorkspace}

_broke_down(workspace, itmax) = false
function _broke_down(workspace::_ShadowWorkspace, itmax)
    Krylov.issolved(workspace) && return false
    # not solved before the iteration limit (Krylov.jl reads itmax = 0 as 2n): a breakdown
    limit = iszero(itmax) ? 2*length(workspace.x) : itmax
    return Krylov.iteration_count(workspace) < limit
end

# positive, non-constant and the same on every run and backend: it is not orthogonal to any
# residual that solves meet in practice, and keeps threaded runs bit-reproducible
function _fallback_shadow!(c)
    data = c isa XVector ? parent(c) : c
    kernel! = _sized(_fallback_shadow_kernel!, get_backend(data), 256, length(data))
    kernel!(data)
    return c
end

@kernel function _fallback_shadow_kernel!(c)
    i = @index(Global)
    @inbounds c[i] = convert(eltype(c), 0.5f0 + Float32((i*40503) & 65535)/65536f0)
end
