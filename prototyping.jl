using Plots
using FVM_1D
using Krylov
# using CUDA
# using GPUArrays

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)
# mesh = cu(mesh)

# Inlet conditions

velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.6,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.4,
    )
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint)
)

runtime = set_runtime(iterations=600, write_interval=-1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

# GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rp = @enter simple!(model, config) #, pref=0.0)

## SET PRECONDITIONERS
# set_preconditioner_test(PT::T, eqn, BCs, runtime
# ) where T<:PreconditionerType = 
# begin
#     discretise!(
#         eqn, ConstantScalar(zero(_get_int(get_phi(eqn).mesh))), runtime)
#     apply_boundary_conditions_test!(eqn, BCs)
#     P = Preconditioner{T}(eqn.equation.A)
#     update_preconditioner!(P)
#     return P
# end

# ## APPLY BOUNDARY CONDITIONS
# apply_boundary_conditions_test!(eqn, BCs) = begin
#     _apply_boundary_conditions_test!(eqn.model, BCs, eqn)
# end

# @generated function _apply_boundary_conditions_test!(
#     model::Model{T,S,TN,SN}, BCs::B, eqn) where {T,S,TN,SN,B}

#     # Unpack terms that make up the model (not sources)
#     # nTerms = model.parameters[3]
#     nTerms = TN

#     # Definition of main assignment loop (one per patch)
#     assignment_loops = []
#     for bci ∈ 1:length(BCs.parameters)
#         func_calls = Expr[]
#         for t ∈ 1:nTerms 
#             call = quote
#                 (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
#             end
#             push!(func_calls, call)
#         end
#         assignment_loop = quote
#             (; facesID, cellsID) = boundaries[BCs[$bci].ID]
#             @inbounds for i ∈ eachindex(cellsID)
#                 faceID = facesID[i]
#                 cellID = cellsID[i]
#                 face = faces[faceID]
#                 cell = cells[cellID]
#                 $(func_calls...)
#             end
#         end
#         push!(assignment_loops, assignment_loop.args...)
#     end

#     quote
#     (; A, b) = eqn.equation
#     mesh = model.terms[1].phi.mesh
#     (; boundaries, faces, cells) = mesh
#     $(assignment_loops...)
#     nothing
#     end
# end


## SIMPLE ALGORITHM

# @info "Extracting configuration and input fields..."
# (; U, p, nu, mesh) = model
# (; solvers, schemes, runtime) = config

# @info "Preallocating fields..."

# ∇p = Grad{schemes.p.gradient}(model.p)
# mdotf = FaceScalarField(model.mesh)
# # nuf = ConstantScalar(nu) # Implement constant field!
# rDf = FaceScalarField(model.mesh)
# nueff = FaceScalarField(model.mesh)
# initialise!(rDf, 1.0)
# divHv = ScalarField(model.mesh)

# @info "Defining models..."

# ux_eqn = (
#     Time{schemes.U.time}(model.U.x)
#     + Divergence{schemes.U.divergence}(mdotf, model.U.x) 
#     - Laplacian{schemes.U.laplacian}(nueff, model.U.x) 
#     == 
#     -Source(∇p.result.x)
# ) → Equation(model.mesh)

# uy_eqn = (
#     Time{schemes.U.time}(model.U.y)
#     + Divergence{schemes.U.divergence}(mdotf, model.U.y) 
#     - Laplacian{schemes.U.laplacian}(nueff, model.U.y) 
#     == 
#     -Source(∇p.result.y)
# ) → Equation(model.mesh)

# p_eqn = (
#     Laplacian{schemes.p.laplacian}(rDf, model.p) == Source(divHv)
# ) → Equation(model.mesh)

# @info "Initialising preconditioners..."

# @reset ux_eqn.preconditioner = set_preconditioner_test(
#                 solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
# @reset uy_eqn.preconditioner = ux_eqn.preconditioner
# @reset p_eqn.preconditioner = set_preconditioner_test(
#                 solvers.p.preconditioner, p_eqn, p.BCs, runtime)