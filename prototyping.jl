using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions

mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_2mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_10-7.5-2mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh, integer=Int32, float=Float32)
mesh = update_mesh_format(mesh)

# Inlet conditions

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu));

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
);

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
);

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-3
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #CgLanczosSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3
    )
);

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint)
);

runtime = set_runtime(iterations=600, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime);

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

@info "Extracting configuration and input fields..."
model = adapt(backend, model_in)
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

 progress = Progress(iterations; dt=1.0, showspeed=true)

 CUDA.@time for iteration ∈ 1:iterations

     @. prev = U.x.values
     # type = typeof(ux_eqn)
     # println("$type")
     discretise!(ux_eqn, prev, runtime)
     apply_boundary_conditions!(ux_eqn, U.x.BCs)
     # ux_eqn.b .-= divUTx
     implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
     update_preconditioner!(ux_eqn.preconditioner, mesh)
     run!(ux_eqn, solvers.U, U.x) #opP=Pu.P, solver=solver_U)
     residual!(R_ux, ux_eqn.equation, U.x, iteration)

     @. prev = U.y.values
     discretise!(uy_eqn, prev, runtime)
     apply_boundary_conditions!(uy_eqn, U.y.BCs)
     # uy_eqn.b .-= divUTy
     implicit_relaxation!(uy_eqn, prev, solvers.U.relax, mesh)
     update_preconditioner!(uy_eqn.preconditioner, mesh)
     run!(uy_eqn, solvers.U, U.y)
     residual!(R_uy, uy_eqn.equation, U.y, iteration)

     @. prev = U.z.values
     discretise!(uz_eqn, prev, runtime)
     apply_boundary_conditions!(uz_eqn, U.z.BCs)
     # uy_eqn.b .-= divUTy
     implicit_relaxation!(uz_eqn, prev, solvers.U.relax, mesh)
     update_preconditioner!(uz_eqn.preconditioner, mesh)
     run!(uz_eqn, solvers.U, U.z)
     residual!(R_uz, uz_eqn.equation, U.z, iteration)
       
     inverse_diagonal!(rD, ux_eqn)
     interpolate!(rDf, rD)
     remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p)
     H!(Hv, U, ux_eqn, uy_eqn, uz_eqn)
     
     interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
     correct_boundaries!(Uf, Hv, U.BCs)
     div!(divHv, Uf)
     
     @. prev = p.values
     discretise!(p_eqn, prev, runtime)
     apply_boundary_conditions!(p_eqn, p.BCs)
     setReference!(p_eqn, pref, 1)
     update_preconditioner!(p_eqn.preconditioner, mesh)
     run!(p_eqn, solvers.p, p)

     explicit_relaxation!(p, prev, solvers.p.relax)
     residual!(R_p, p_eqn.equation, p, iteration)

     grad!(∇p, pf, p, p.BCs) 

    #  correct = false
    #  if correct
    #      ncorrectors = 1
    #      for i ∈ 1:ncorrectors
    #          discretise!(p_eqn)
    #          apply_boundary_conditions!(p_eqn, p.BCs)
    #          setReference!(p_eqn.equation, pref, 1)
    #          # grad!(∇p, pf, p, pBCs) 
    #          interpolate!(gradpf, ∇p, p)
    #          nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
    #          correct!(p_eqn.equation, p_model.terms.term1, pf)
    #          run!(p_model, solvers.p)
    #          grad!(∇p, pf, p, pBCs) 
    #      end
    #  end

     correct_velocity!(U, Hv, ∇p, rD)
     interpolate!(Uf, U)
     correct_boundaries!(Uf, U, U.BCs)
     flux!(mdotf, Uf)

    #  if isturbulent(model)
    #      grad!(gradU, Uf, U, U.BCs)
    #      turbulence!(turbulence, model, S, S2, prev) 
    #      update_nueff!(nueff, nu, turbulence)
    #  end
 end