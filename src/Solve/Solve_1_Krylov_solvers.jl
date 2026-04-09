export _workspace
export Cg, Cgs, Bicgstab, Gmres

abstract type AbstractLinearSolver end

struct Cg <: AbstractLinearSolver end
struct Cgs <: AbstractLinearSolver end
struct Bicgstab <: AbstractLinearSolver end
struct Gmres <: AbstractLinearSolver end

# Krylov.jl workspace constructors (2-arg: solver, b)
_workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))
_workspace(::Cgs, b) = CgsWorkspace(KrylovConstructor(b))
_workspace(::Bicgstab, b) = BicgstabWorkspace(KrylovConstructor(b))
_workspace(::Gmres, b) = GmresWorkspace(KrylovConstructor(b))

# Generic 3-arg fallback (solver, A, b): Krylov solvers ignore A.
# AMG has its own specific 3-arg method defined in AMG_6_api.jl.
_workspace(s::AbstractLinearSolver, _, b) = _workspace(s, b)