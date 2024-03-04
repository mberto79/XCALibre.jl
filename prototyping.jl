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
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.6,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
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

    @. prev = U.x.values
    discretise!(ux_eqn, prev, runtime)
    apply_boundary_conditions!(ux_eqn, U.x.BCs)
    implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
    @time begin update_preconditioner!(ux_eqn.preconditioner, mesh) end
    D_gpu = ux_eqn.preconditioner.storage.D

    CUDA.allowscalar(true)

    sum = 0
    for i in eachindex(D_cpu)
        if D_gpu[i] != D_cpu[i]
            sum += 1
            diff = D_cpu[i] - D_gpu[i]
            println("ID = $i\nDifference = $diff\n")
        end
    end
    sum

    # run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
    ux_eqn.preconditioner.A.nzVal
    ux_eqn.preconditioner.storage

    using SparseArrays

    function sparse_array_deconstructor_preconditioners(arr::SparseArrays.SparseMatrixCSC)
        (; rowval, colptr, nzval, m, n) = arr
        return rowval, colptr, nzval, m ,n
    end
    
    function sparse_array_deconstructor_preconditioners(arr::CUDA.CUSPARSE.CuSparseMatrixCSC)
        (; rowVal, colPtr, nzVal, dims) = arr
        return rowVal, colPtr, nzVal, dims[1], dims[2]
    end

    function extract_diagonal!(D, Di, A::AbstractSparseArray{Tf,Ti}, backend) where {Tf,Ti}
        rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    
        kernel! = extract_diagonal_kernel!(backend)
        kernel!(D, Di, nzval, ndrange = n)
    end
    
    @kernel function extract_diagonal_kernel!(D, Di, nzval)
        i = @index(Global)
        
        @inbounds begin
            D[i] = nzval[Di[i]]
        end
    end


    P = ux_eqn.preconditioner

    backend = _get_backend(mesh)

    (; A, storage) = P
    # (; colptr, n, nzval, rowval) = A
    rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    (; Di, Ri, D, upper_indices_IDs) = storage
    
    extract_diagonal!(D, Di, A, backend)

    kernel! = update_dilu_diagonal_kernel8!(backend)
    kernel!(upper_indices_IDs, Di, colptr, Ri, rowval, D, nzval, ndrange = n)
    # D .= 1.0./D # store inverse
    nothing


    @kernel function update_dilu_diagonal_kernel8!(upper_indices_IDs, Di, colptr, Ri, rowval, D, nzval)
        i = @index(Global)
        
        @inbounds begin
            # D[i] = nzval[Di[i]]
            upper_index_ID = upper_indices_IDs[i] 
            c_start = Di[i] + 1 
            c_end = colptr[i+1] - 1
            r_count = 0
            @inbounds for c_pointer ∈ c_start:c_end
                j = rowval[c_pointer]
                r_count += 1
                r_pointer = Ri[upper_index_ID[r_count]]
                D[j] -= nzval[c_pointer]*nzval[r_pointer]/D[i]
            end
            D[i] = 1/D[i] # store inverse
        end
    end

    
    
    
    
    
    
    
    
    
    
    
    
    
    @test = cu(ux_eqn.solver.R)
    test

    GmresSolver()


    using CUDA
    using Adapt
    using KernelAbstractions

    backend = _get_backend(mesh)

    redundant = cu([0 2 34 34 23 12 2])
    kernel = test1(backend)
    for i in 1:10
    kernel(redundant, ndrange = length(redundant))
    end

    @kernel function test1(redundant)
        i = @index(Global)
        
        @inbounds begin
            for j in 1:2
                @cushow i
            end
        end
    end