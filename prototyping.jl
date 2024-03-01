using Plots
using FVM_1D
using Krylov

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

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
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
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

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

# Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

using Accessors
using Adapt
using LoopVectorization
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
using CUDA
using KernelAbstractions

    @info "Extracting configuration and input fields..."
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

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
    ) → Equation(mesh)

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, runtime)

    @info "Pre-allocating solvers..."
     
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
    model = adapt(CuArray, model)
    ∇p = adapt(CuArray, ∇p)
    ux_eqn = adapt(CuArray, ux_eqn)
    uy_eqn = adapt(CuArray, uy_eqn)
    p_eqn = adapt(CuArray, p_eqn)
    turbulence = adapt(CuArray, turbulence)
    config = adapt(CuArray, config)
    
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

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)

    Uf = adapt(CuArray,Uf)
    rDf = adapt(CuArray, rDf)
    rD = adapt(CuArray, rD)
    pf = adapt(CuArray, pf)

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)


    # nueff.values

    update_nueff!(nueff, nu, turbulence)

    prev = adapt(CuArray, prev)
    # solvers = adapt(CuArray, solvers)

    @. prev = U.x.values
    discretise!(ux_eqn, prev, runtime)
    apply_boundary_conditions!(ux_eqn, U.x.BCs)
    implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
    @time begin update_preconditioner!(ux_eqn.preconditioner, mesh) end
    # run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
    ux_eqn.preconditioner.A.nzVal
    ux_eqn.preconditioner.storage

    test = cu(ux_eqn.solver.R)
    test

    GmresSolver()


    using CUDA
    using Adapt
    mutable struct Test{I,F}
        test1::I
        test2::F
        test3::I
        test4::Vector{Int}
    end
    Adapt.@adapt_structure Test
    function ahh(test1, test2, test3, test4::Vector{Int})
        return Test{typeof(test1), typeof(test2)}(test1, test2, test3, test4)
    end
    
    my_test = ahh(10, 101.23, 200, [1, 2, 3, 4, 5, 6])
    my_test = cu(my_test)
    my_test.test4