export _workspace
export Cg, Cgs, Bicgstab, Gmres, Richardson

abstract type AbstractLinearSolver end

struct Cg <: AbstractLinearSolver end
struct Cgs <: AbstractLinearSolver end
struct Bicgstab <: AbstractLinearSolver end
struct Gmres <: AbstractLinearSolver end

"""
    Richardson()

Stationary iteration `x += M⁻¹(b - Ax)` with no Krylov acceleration. Use it with
`preconditioner = AMG(...)` for a standalone multigrid solve (repeated V-cycles); on a distributed
mesh it maps to PETSc's `richardson`.
"""
struct Richardson <: AbstractLinearSolver end

# Krylov.jl workspace constructors
_workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))
_workspace(::Cgs, b) = CgsWorkspace(KrylovConstructor(b))
_workspace(::Bicgstab, b) = BicgstabWorkspace(KrylovConstructor(b))
_workspace(::Gmres, b) = GmresWorkspace(KrylovConstructor(b))
_workspace(::Richardson, b) = throw(ArgumentError(
    "Richardson() runs only with preconditioner=AMG(...) on a serial mesh"))
# equation workspaces hold XVector storage on the CPU so Krylov's vector work runs on Julia's threads
_workspace(solver::AbstractLinearSolver, b, ::Type) = _workspace(solver, _krylov_vector(b))
_krylov_vector(v::Vector) = XVector(v)
_krylov_vector(v) = v
# Krylov requires b and x0 in the workspace's storage type
_like_workspace(::XVector, v::Vector) = XVector(v)
_like_workspace(_, v) = v
_index_type(A) = eltype(_rowptr(A))

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
