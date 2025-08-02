# export _workspace
# export Cg, Cgs, Bicgstab, Gmres

# abstract type AbstractLinearSolver end

# struct Cg <: AbstractLinearSolver end
# struct Cgs <: AbstractLinearSolver end
# struct Bicgstab <: AbstractLinearSolver end
# struct Gmres <: AbstractLinearSolver end

# # Krylov.jl workspace constructors
# _workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))
# _workspace(::Cgs, b) = CgsWorkspace(KrylovConstructor(b))
# _workspace(::Bicgstab, b) = BicgstabWorkspace(KrylovConstructor(b))
# _workspace(::Gmres, b) = GmresWorkspace(KrylovConstructor(b))