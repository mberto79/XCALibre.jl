module XCALibre

# using Krylov 
# export Bicgstab(), Cg(), Gmres()

using KernelAbstractions; export CPU
import Adapt: adapt; export adapt


include("Multithread/Multithread.jl")
include("Mesh/Mesh.jl")
include("UNV2/UNV2.jl")
include("UNV3/UNV3.jl")
include("FoamMesh/FoamMesh.jl")
include("Fields/Fields.jl")
include("ModelFramework/ModelFramework.jl")
include("Discretise/Discretise.jl")
include("Solve/Solve.jl")
include("Simulate/Simulate.jl")
include("Calculate/Calculate.jl")
include("IOFormats/IOFormats.jl")
include("ModelPhysics/ModelPhysics.jl")
include("Postprocess/Postprocess.jl")
include("Solvers/Solvers.jl")
include("Preprocess/Preprocess.jl")
include("Mesh/BlockMesher2D/BlockMesher2D.jl")

using Reexport
@reexport using XCALibre.Multithread
@reexport using XCALibre.Mesh
@reexport using XCALibre.FoamMesh
@reexport using XCALibre.Fields
@reexport using XCALibre.ModelFramework
@reexport using XCALibre.Discretise
@reexport using XCALibre.Solve
@reexport using XCALibre.Calculate
@reexport using XCALibre.ModelPhysics
@reexport using XCALibre.Simulate
@reexport using XCALibre.Postprocess
@reexport using XCALibre.Solvers
@reexport using XCALibre.Preprocess
@reexport using XCALibre.IOFormats
@reexport using XCALibre.UNV3
@reexport using XCALibre.UNV2
@reexport using XCALibre.BlockMesher2D

using StaticArrays, LinearAlgebra, SparseMatricesCSR, SparseArrays, LinearOperators
using ProgressMeter, Printf, Adapt

# include("precompile.jl") need new precompile file

# using PrecompileTools: @setup_workload, @compile_workload


# @setup_workload begin
#     grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
#     grid = "backwardFacingStep_10mm.unv"
#     mesh_file = joinpath(grids_dir, grid)

#     # Inlet conditions
#     Umag = 0.5
#     velocity = [Umag, 0.0, 0.0]
#     nu = 1e-3
    
#     k_inlet = 1
#     ω_inlet = 1000
#     ω_wall = ω_inlet

#     println(mesh_file)
#         @compile_workload begin

#             # Incompressible: Laminar and steady 

#             mesh = UNV2D_mesh(mesh_file, scale=0.001)
#             mesh_dev = mesh

#             model = Physics(
#                 time = Steady(),
#                 fluid = Fluid{Incompressible}(nu = nu),
#                 turbulence = RANS{Laminar}(),
#                 energy = Energy{Isothermal}(),
#                 domain = mesh # mesh_dev  # use mesh_dev for GPU backend
#                 )
            
#             @assign! model momentum U (
#                 Dirichlet(:inlet, velocity),
#                 Neumann(:outlet, 0.0),
#                 Wall(:wall, [0.0, 0.0, 0.0]),
#                 Symmetry(:top)
#             )
            
#             @assign! model momentum p (
#                 Neumann(:inlet, 0.0),
#                 Dirichlet(:outlet, 0.0),
#                 Neumann(:wall, 0.0),
#                 Symmetry(:top)
#             )
            
#             schemes = (
#                 U = Schemes(divergence = Linear),
#                 p = Schemes()
#             )
            
#             solvers = (
#                 U = SolverSetup(
#                     model.momentum.U;
#                     solver = Bicgstab(), # Bicgstab(), Gmres()
#                     smoother = JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
#                     preconditioner = Jacobi(),
#                     convergence = 1e-7,
#                     relax = 0.8,
#                     rtol = 1e-1,
#                 ),
#                 p = SolverSetup(
#                     model.momentum.p;
#                     solver = Cg(), # Bicgstab(), Gmres()
#                     smoother = JacobiSmoother(domain=mesh_dev, loops=5, omega=2/3),
#                     preconditioner = Jacobi(),
#                     convergence = 1e-7,
#                     relax = 0.2,
#                     rtol = 1e-2,
#                 )
#             )
            
#             runtime = Runtime(
#                 iterations=10, time_step=1, write_interval=10)
            
#             hardware = Hardware(backend=CPU(), workgroup=1024)
            
#             config = Configuration(
#                 solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
            
#             initialise!(model.momentum.U, velocity) === nothing
#             initialise!(model.momentum.p, 0.0) === nothing
            
#             residuals = run!(model, config)
            
#             inlet = boundary_average(:inlet, model.momentum.U, config)
#             outlet = boundary_average(:outlet, model.momentum.U, config)

#             # Incompressible: Laminar and unsteady

#             model = Physics(
#             time = Transient(),
#             fluid = Fluid{Incompressible}(nu = nu),
#             turbulence = RANS{KOmega}(),
#             energy = Energy{Isothermal}(),
#             domain = mesh_dev
#             )

#             @assign! model momentum U (
#                 Dirichlet(:inlet, velocity),
#                 Neumann(:outlet, 0.0),
#                 Wall(:wall, [0.0, 0.0, 0.0]),
#                 Dirichlet(:top, [0.0, 0.0, 0.0])
#             )

#             @assign! model momentum p (
#                 Neumann(:inlet, 0.0),
#                 Dirichlet(:outlet, 0.0),
#                 Neumann(:wall, 0.0),
#                 Neumann(:top, 0.0)
#             )

#             @assign! model turbulence k (
#                 Dirichlet(:inlet, k_inlet),
#                 Neumann(:outlet, 0.0),
#                 Dirichlet(:wall, 0.0),
#                 Dirichlet(:top, 0.0)
#             )

#             @assign! model turbulence omega (
#                 Dirichlet(:inlet, ω_inlet),
#                 Neumann(:outlet, 0.0),
#                 OmegaWallFunction(:wall),
#                 OmegaWallFunction(:top)
#             )

#             @assign! model turbulence nut (
#                 Dirichlet(:inlet, k_inlet/ω_inlet),
#                 Neumann(:outlet, 0.0),
#                 Dirichlet(:wall, 0.0), 
#                 Dirichlet(:top, 0.0)
#             )

#             schemes = (
#                 U = Schemes(gradient=Midpoint, time=Euler),
#                 p = Schemes(gradient=Midpoint),
#                 k = Schemes(gradient=Midpoint, time=Euler),
#                 omega = Schemes(gradient=Midpoint, time=Euler)
#             )

#             solvers = (
#                 U = SolverSetup(
#                     model.momentum.U;
#                     solver      = Bicgstab(), # Bicgstab(), Gmres()
#                     preconditioner = Jacobi(), 
#                     convergence = 1e-7,
#                     relax       = 1.0,
#                     rtol = 1e-3
#                 ),
#                 p = SolverSetup(
#                     model.momentum.p;
#                     solver      = Cg(), # Bicgstab(), Gmres()
#                     preconditioner = Jacobi(), 
#                     convergence = 1e-7,
#                     relax       = 1.0,
#                     rtol = 1e-3
#                 ),
#                 k = SolverSetup(
#                     model.turbulence.k;
#                     solver      = Bicgstab(), # Bicgstab(), Gmres()
#                     preconditioner = Jacobi(), 
#                     convergence = 1e-7,
#                     relax       = 1.0,
#                     rtol = 1e-3
#                 ),
#                 omega = SolverSetup(
#                     model.turbulence.omega;
#                     solver      = Bicgstab(), # Bicgstab(), Gmres()
#                     preconditioner = Jacobi(), 
#                     convergence = 1e-7,
#                     relax       = 1.0,
#                     rtol = 1e-3
#                 )
#             )

#             runtime = Runtime(
#                 iterations=10, write_interval=10, time_step=0.01)

#             # hardware = Hardware(backend=CUDABackend(), workgroup=32)
#             hardware = Hardware(backend=CPU(), workgroup=1024)

#             config = Configuration(
#                 solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

#             GC.gc()

#             initialise!(model.momentum.U, velocity) === nothing
#             initialise!(model.momentum.p, 0.0) === nothing
#             initialise!(model.turbulence.k, k_inlet) === nothing
#             initialise!(model.turbulence.omega, ω_inlet) === nothing
#             initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

#             residuals = run!(model, config);
#         end
# end

end # module
