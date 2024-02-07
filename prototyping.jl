using Plots
using FVM_1D
using Krylov
using CUDA, KernelAbstractions
using Accessors
using LoopVectorization
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
# using GPUArrays

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

# Inlet conditions

velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

CUDA.allowscalar(false)
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

# Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

# @info "Extracting configuration and input fields..."
(; U, p, nu, turbulence, energy, mesh, boundary_info) = model
(; solvers, schemes, runtime) = config

# @info "Preallocating fields..."

∇p = Grad{schemes.p.gradient}(p)
mdotf = FaceScalarField(mesh)
# nuf = ConstantScalar(nu) # Implement constant field!
rDf = FaceScalarField(mesh)
nueff = FaceScalarField(mesh)
initialise!(rDf, 1.0)
divHv = ScalarField(mesh)

# @info "Defining models..."

ux_eqn = (
    Time{schemes.U.time}(U.x)
    + Divergence{schemes.U.divergence}(mdotf, U.x) 
    - Laplacian{schemes.U.laplacian}(nueff, U.x) 
    == 
    -Source(∇p.result.x)
) → Equation(mesh)

uy_eqn = (
    Time{schemes.U.time}(U.y)
    + Divergence{schemes.U.divergence}(mdotf, U.y) 
    - Laplacian{schemes.U.laplacian}(nueff, U.y) 
    == 
    -Source(∇p.result.y)
) → Equation(mesh)

p_eqn = (
    Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
) → Equation(mesh)

# @info "Initialising preconditioners..."

@reset ux_eqn.preconditioner = set_preconditioner(
                solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
@reset uy_eqn.preconditioner = ux_eqn.preconditioner
@reset p_eqn.preconditioner = set_preconditioner(
                solvers.p.preconditioner, p_eqn, p.BCs, runtime)

# @info "Pre-allocating solvers..."

@reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
@reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
@reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

if isturbulent(model)
    @info "Initialising turbulence model..."
    turbulence = initialise_RANS(mdotf, p_eqn, config, model)
    config = turbulence.config
else
    turbulence = nothing
end

CUDA.allowscalar(false)

test = cu(model)
cu(∇p)
ux_eqn = cu(ux_eqn)
uy_eqn = cu(uy_eqn)
p_eqn = cu(p_eqn)
turbulence = cu(turbulence)
config = cu(config)

cu((
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
))

mesh = cu(mesh)

_get_backend(mesh)

@kernel function test(var) 

R_ux, R_uy, R_p  = SIMPLE_loop(
    model, ∇p, ux_eqn, uy_eqn, p_eqn, turbulence, config ; resume=resume, pref=pref)