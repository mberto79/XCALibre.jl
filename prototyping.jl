# using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions

using Accessors
using LoopVectorization
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
using CUDA
using KernelAbstractions
using Atomix
using Adapt

# BFS_U CASE
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
# mesh = update_mesh_format(mesh, integer = Int32, float = Float32)
mesh = update_mesh_format(mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(time=Euler),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    )
)

runtime = set_runtime(
    iterations=1000, time_step=0.005, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

@info "Extracting configuration and input fields..."
model = adapt(backend, model)
(; U, p, nu, mesh) = model
(; solvers, schemes, runtime) = config

@info "Preallocating fields..."

∇p = Grad{schemes.p.gradient}(p)
mdotf = FaceScalarField(mesh)
# nuf = ConstantScalar(nu) # Implement constant field!
rDf = FaceScalarField(mesh)
nueff = FaceScalarField(mesh)
initialise!(rDf, 1.0)
divHv = ScalarField(mesh)

@info "Defining models..."

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

uz_eqn = (
    Time{schemes.U.time}(U.z)
    + Divergence{schemes.U.divergence}(mdotf, U.z) 
    - Laplacian{schemes.U.laplacian}(nueff, U.z) 
    == 
    -Source(∇p.result.z)
) → Equation(mesh)

p_eqn = (
    Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
) → Equation(mesh)

CUDA.allowscalar(false)
# model = _convert_array!(model, backend)
# ∇p = _convert_array!(∇p, backend)
# ux_eqn = _convert_array!(ux_eqn, backend)
# uy_eqn = _convert_array!(uy_eqn, backend)
# p_eqn = _convert_array!(p_eqn, backend)

@info "Initialising preconditioners..."

@reset ux_eqn.preconditioner = set_preconditioner(
                solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
@reset uy_eqn.preconditioner = ux_eqn.preconditioner
@reset uz_eqn.preconditioner = ux_eqn.preconditioner
@reset p_eqn.preconditioner = set_preconditioner(
                solvers.p.preconditioner, p_eqn, p.BCs, runtime)

if isturbulent(model)
    @info "Initialising turbulence model..."
    turbulence = initialise_RANS(mdotf, p_eqn, config, model)
    config = turbulence.config
else
    turbulence = nothing
end

@info "Pre-allocating solvers..."
 
@reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
@reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
@reset uz_eqn.solver = solvers.U.solver(_A(uz_eqn), _b(uz_eqn))
@reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

 # Extract model variables and configuration
 (;mesh, U, p, nu) = model
 # ux_model, uy_model = ux_eqn.model, uy_eqn.model
 p_model = p_eqn.model
 (; solvers, schemes, runtime) = config
 (; iterations, write_interval) = runtime
 
 mdotf = get_flux(ux_eqn, 2)
 nueff = get_flux(ux_eqn, 3)
 rDf = get_flux(p_eqn, 1)
 divHv = get_source(p_eqn, 1)
 
 @info "Allocating working memory..."

 # Define aux fields 
 gradU = Grad{schemes.U.gradient}(U)
 gradUT = T(gradU)
 S = StrainRate(gradU, gradUT)
 S2 = ScalarField(mesh)

 # Temp sources to test GradUT explicit source
 # divUTx = zeros(Float64, length(mesh.cells))
 # divUTy = zeros(Float64, length(mesh.cells))

 n_cells = length(mesh.cells)
 Uf = FaceVectorField(mesh)
 pf = FaceScalarField(mesh)
 gradpf = FaceVectorField(mesh)
 Hv = VectorField(mesh)
 rD = ScalarField(mesh)

 # Pre-allocate auxiliary variables

 # Consider using allocate from KernelAbstractions 
 # e.g. allocate(backend, Float32, res, res)
 TF = _get_float(mesh)
 prev = zeros(TF, n_cells)
 prev = _convert_array!(prev, backend) 

 # Pre-allocate vectors to hold residuals 

 R_ux = ones(TF, iterations)
 R_uy = ones(TF, iterations)
 R_uz = ones(TF, iterations)
 R_p = ones(TF, iterations)
 
 interpolate!(Uf, U)   
 correct_boundaries!(Uf, U, U.BCs)
 flux!(mdotf, Uf)
 grad!(∇p, pf, p, p.BCs)

 update_nueff!(nueff, nu, turbulence)
 
 @info "Staring SIMPLE loops..."

#  progress = Progress(iterations; dt=1.0, showspeed=true)

#  CUDA.@tinzval_cpu = 2.2844029e-8
for iteration in 1:1000
    @. prev = U.x.values
    discretise!(ux_eqn, prev, runtime)
    apply_boundary_conditions!(ux_eqn, U.x.BCs)
    implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
    update_preconditioner!(ux_eqn.preconditioner, mesh)
    run!(ux_eqn, solvers.U, U.x) #opP=Pu.P, solver=solver_U)
end

nzval_cpu = ux_eqn.equation.A.nzval
b_cpu = ux_eqn.equation.b
precon_cpu = ux_eqn.preconditioner.storage
values_cpu = U.x.values


CUDA.allowscalar(true)

error_check(nzval_cpu, ux_eqn.equation.A.nzVal, 0)
error_check(b_cpu, ux_eqn.equation.b, 0)
error_check(ux_eqn.equation.A.nzVal, ux_eqn.preconditioner.A.nzVal, 0)
error_check(precon_cpu, ux_eqn.preconditioner.storage, 0)
error_check(values_cpu, U.x.values, 0)

function error_check(arr_cpu, arr_gpu, min_error)
    sum = 0

    error_array = eltype(arr_cpu)[]
    
    for i in eachindex(arr_cpu)

        varcpu = arr_cpu[i]
        vargpu = arr_gpu[i]

        diff = varcpu - vargpu
        
        if abs(diff) > min_error
            println("Index  = $i:\nnzval_cpu = $varcpu\nnzval_gpu = $vargpu\ndifference = $diff\n")
            sum += 1
            push!(error_array, abs(diff))
        end
    end

    if length(error_array) > 0
        max_error = maximum(error_array)
        println("max error = $max_error")
        min_error = minimum(error_array)
        println("min error = $min_error")
    end

    println("number errored = $sum")
end

using KernelAbstractions
backend = CPU()
backend = CUDABackend()