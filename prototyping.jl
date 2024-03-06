# using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions


# Backend selection

backend = CPU()
# backend = CUDABackend()
iteration = 1

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

    CUDA.allowscalar(false)
    model = _convert_array!(model, backend)
    ∇p = _convert_array!(∇p, backend)
    ux_eqn = _convert_array!(ux_eqn, backend)
    uy_eqn = _convert_array!(uy_eqn, backend)
    p_eqn = _convert_array!(p_eqn, backend)

    @info "Initialising preconditioners..."

    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
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

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_p = ones(TF, iterations)

    Uf = _convert_array!(Uf, backend)
    rDf = _convert_array!(rDf, backend)
    rD = _convert_array!(rD, backend)
    pf = _convert_array!(pf, backend)
    Hv = _convert_array!(Hv, backend)
    prev = _convert_array!(prev, backend)

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)


    # nueff.values

    update_nueff!(nueff, nu, turbulence)

    @. prev = U.x.values
    
    for i in 1:600
        discretise!(ux_eqn, prev, runtime)
        # KernelAbstractions.synchronize(backend)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        # KernelAbstractions.synchronize(backend)
        implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
        # KernelAbstractions.synchronize(backend)
        update_preconditioner!(ux_eqn.preconditioner, mesh)
        run!(ux_eqn, solvers.U, U.x)
        residual!(R_ux, ux_eqn.equation, U.x, i) 

        @. prev = U.y.values
        discretise!(uy_eqn, prev, runtime)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        # uy_eqn.b .-= divUTy
        implicit_relaxation!(uy_eqn, prev, solvers.U.relax, mesh)
        update_preconditioner!(uy_eqn.preconditioner, mesh)
        run!(uy_eqn, solvers.U, U.y)
        residual!(R_uy, uy_eqn.equation, U.y, i)
    end
    # ncpu = ux_eqn.equation.A.nzVal
    # bcpu = ux_eqn.equation.b
    # npreconcpu = ux_eqn.preconditioner.A.nzVal
    # valuescpu = U.x.values
    Ruycpu = R_uy

    # error_check(valuesgpu, U.x.values, 10^-15)
    error_check(Ruxcpu, R_ux, 10^-20)

    function error_check(arr1, arr2, target)
        CUDA.allowscalar(true)
        sum = 0
        for i in eachindex(arr1)
            # if arr1[i] != arr2[i]
                diff = arr1[i]-arr2[i]
                if abs(diff) > target
                    sum += 1
                    println("At index $i: diff = $diff")
                end
            # end
        end
        
        println("$sum elements are out of the specified error")
    end


    #MATMUL
    using Atomix

    @kernel function sparse_matmul6!(nzval, rowval, colptr, mulvec, res)
        i = @index(Global)

        # res[i] = zero(eltype(res))

        # @inbounds begin
            @synchronize
            start = colptr[i]
            fin = colptr[i+1]
    
            for j in start:fin-1
                val = nzval[j] #A[j,i]
                row = rowval[j] #Row index of non-zero element in A
                Atomix.@atomic res[row] += mulvec[i] * val
            end
        # end
    end

    @kernel function matmul_copy_zeros_kernel!(c, fzero)
        i = @index(Global)

        @inbounds begin
            c[i] = fzero
        end
    end

    function matmul_sparse!(a, b, c, backend)
        if size(a)[2] != length(b)
            error("Matrix size mismatch!")
            return nothing
        end

        nzval_array = nzval(a)
        colptr_array = colptr(a)
        rowval_array = rowval(a)
        fzero = zero(eltype(c))

        kernel! = matmul_copy_zeros_kernel!(backend)
        kernel!(c, fzero, ndrange = length(c))
        KernelAbstractions.synchronize(backend)

        kernel! = sparse_matmul6!(backend)
        kernel!(nzval_array, rowval_array, colptr_array, b, c, ndrange=length(c))
        KernelAbstractions.synchronize(backend)
    end

    # GPU
    # CUDA.allowscalar(false)
    (; A, b, R, Fx) = ux_eqn.equation
    # Fx_test = CUDA.zeros(eltype(Fx),length(Fx))

    # for i in 1:2
    matmul_sparse!(A, U.x.values, Fx, backend)
    # KernelAbstractions.synchronize(backend)
    # end
    Fxgpu = Fx

    # backend = _get_backend(phi.mesh)

    # CPU
    (; A, b, R, Fx) = ux_eqn.equation
    vals = U.x.values
    mul!(Fx, A, vals)
    Fxcpu = Fx

    # ERROR CHECK
    Fxgpu == Fxcpu
    error_check(Fxgpu, Fxcpu, 10^-16)



    A.rowval
    A.colptr
    A.nzval



    using SparseArrays
    A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [6, 1, 2, 12])

    mulvec = Array{eltype(A.nzval)}(undef, size(A)[1])
    for i in eachindex(mulvec)
        mulvec[i] = rand(1:10)
    end
    mulvec

    res1 = zeros(eltype(A.nzval), size(A)[1])

    for i in 1:length(A.colptr) - 1
        start = A.colptr[i]
        fin = A.colptr[i+1]

        for j in start:fin-1
            val = A.nzval[j] #A[j,i]
            row = A.rowval[j] #Row index of non-zero element in A
            res1[row] += mulvec[i] * val
        end
    end

    using Atomix

    @kernel function sparse_matmul1!(nzval, rowval, colptr, mulvec, res1)
        i = @index(Global)

        # @inbounds begin
            start = colptr[i]
            fin = colptr[i+1]
    
            @synchronize
            for j in start:fin-1
                val = nzval[j] #A[j,i]
                row = rowval[j] #Row index of non-zero element in A
                Atomix.@atomic res1[row] += mulvec[i] * val
            end
        # end
    end

    nzval_array = nzval(A)
    colptr_array = colptr(A)
    rowval_array = rowval(A)

    mulvec = Array{eltype(A.nzval)}(undef, size(A)[1])
    for i in eachindex(mulvec)
        mulvec[i] = rand(1:10)
    end

    backend = CUDABackend()

    res1 = zeros(eltype(A.nzval), size(A)[1])

    nzval_array = _convert_array!(nzval_array, backend)
    colptr_array = _convert_array!(colptr_array, backend)
    rowval_array = _convert_array!(rowval_array, backend)
    mulvec = _convert_array!(mulvec, backend)
    res1 = _convert_array!(res1, backend)

    kernel! = sparse_matmul1!(backend)
    kernel!(nzval_array, rowval_array, colptr_array, mulvec, res1, ndrange = length(colptr_array) - 1)


    res1

    A.rowval
    A.colptr
    A.nzval

    res2 = zeros(eltype(A.nzval), size(A)[1])
    mul!(res2, A, mulvec)

    error_check(res1, res2, 10^-15)