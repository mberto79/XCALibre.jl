using Plots
using FVM_1D
using Krylov
using KernelAbstractions

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

cID_array = _get_int(mesh)[]

for i in 1:length(mesh.cells)
    faces_range_current = mesh.cells[i].faces_range
    
    if i > 1
        faces_range_prev = mesh.cells[i-1].faces_range
    else
        faces_range_prev = 0
    end

    cID = i + maximum(faces_range_prev)
    push!(cID_array, cID)

    for fi in 1:length(faces_range_current)
        index = cID + fi
        push!(cID_array, index)
    end
end

cID_array
sum = 1

for i in eachindex(cID_array)
    if i < length(cID_array)
        if cID_array[i+1] - cID_array[i] == 1
            sum += 1
        end
    end
end
sum == length(cID_array)

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
    U = set_schemes(),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=500)

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
    # model_backend = _convert_array!(model, backend)
    model_backend = adapt(backend, model)
    (; U, p, nu, mesh) = model_backend
    (; solvers, schemes, runtime) = config

    # Test that backends from KernelAbstractions can be used to adapt directly
    # no need for _convert_array function!
    mesh_cpu = adapt(CPU(), mesh) 
    mesh_1 = adapt(CUDABackend(), mesh_cpu) 


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
    ) → Equation(mesh);

    CUDA.allowscalar(false)
    # model = _convert_array!(model, backend)
    # ∇p = _convert_array!(∇p, backend)
    # ux_eqn = _convert_array!(ux_eqn, backend)
    # uy_eqn = _convert_array!(uy_eqn, backend)
    # p_eqn = _convert_array!(p_eqn, backend)

    @info "Initialising preconditioners..."

    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, uy_eqn, U.y.BCs, runtime)
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
    (;mesh, U, p, nu) = model_backend # TEMP!!! inside function model_backend = model
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime
    
    # mdotfx = get_flux(ux_eqn, 2) # THIS GAVE ME A CLUE!
    # nueffx = get_flux(ux_eqn, 3)

    # mdotfy = get_flux(uy_eqn, 2)
    # nueffy = get_flux(uy_eqn, 3)

    mdotf = get_flux(ux_eqn, 2) # THIS GAVE ME A CLUE!
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
    R_p = ones(TF, iterations)

    # Uf = _convert_array!(Uf, backend)
    # rDf = _convert_array!(rDf, backend)
    # rD = _convert_array!(rD, backend)
    # pf = _convert_array!(pf, backend)
    # Hv = _convert_array!(Hv, backend)
    

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    # flux!(mdotfx, Uf)
    # flux!(mdotfy, Uf)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)


    # nueff.values

    update_nueff!(nueff, nu, turbulence)

    # uy_eqn.preconditioner.A.nzval
    
    # for i in 1:length(R_ux)
        @. prev = U.x.values
        # type = typeof(ux_eqn)
        # println("$type")
        discretise!(ux_eqn, prev, runtime)
        # apply_boundary_conditions!(ux_eqn, U.x.BCs)
        # ux_eqn.b .-= divUTx
        # implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
        # update_preconditioner!(ux_eqn.preconditioner, mesh)
        # run!(ux_eqn, solvers.U, U.x) #opP=Pu.P, solver=solver_U)
        # residual!(R_ux, ux_eqn.equation, U.x, i)

        @. prev = U.y.values
        discretise!(uy_eqn, prev, runtime)


        model = uy_eqn.model
        eqn = uy_eqn
        
            # quote
                # Extract number of terms and sources
                nTerms = length(model.terms)
                nSources = length(model.sources)
        
                # Define variables for function
                mesh = model.terms[1].phi.mesh
                # precon = eqn.preconditioner
        
                # Deconstructors to get lower-level variables for function
                # (; A, b) = eqn.equation
                A_array = _A(eqn)
                b_array = _b(eqn)
                (; terms, sources) = model
                (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
                
                # Get types and create float(zero) and integer(one)
                integer = _get_int(mesh)
                float = _get_float(mesh)
                backend = _get_backend(mesh)
                fzero = zero(float) # replace with func to return mesh type (Mesh module)
                ione = one(integer)
        
                # Deconstruct sparse array dependent on sparse arrays type
                rowval_array = _rowval(A_array)
                colptr_array = _colptr(A_array)
                nzval_array = _nzval(A_array)
                
                # Kernel to set nzval array to 0
                kernel! = set_nzval(backend)
                kernel!(nzval_array, fzero, ndrange = length(nzval_array))
                KernelAbstractions.synchronize(backend)
                nzval_array
        
                # Set initial values for indexing of nzval array
                cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
                nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
                offset = zero(integer)
        
                # Assign storage for sources arrays
                # sources_field = Array{typeof(sources[1].field)}(undef, length(sources))
                # sources_sign = Array{typeof(sources[1].sign)}(undef, length(sources))
        
                # # Populate sources arrays
                # for i in eachindex(sources)
                #     sources_field[i] = sources[i].field
                #     sources_sign[i] = sources[i].sign
                # end
        
                # # Copy sources arrays to required backend
                # sources_field = _convert_array!(sources_field, backend)
                # sources_sign = _convert_array!(sources_sign, backend)
                # KernelAbstractions.synchronize(backend)
        
                # Set b array to 0
                kernel! = set_b!(backend)
                kernel!(fzero, b_array, ndrange = length(b_array))
                KernelAbstractions.synchronize(backend)
                b_array
                # Run schemes and sources calculations on all terms
        
                for i in 1:nTerms
                    schemes_and_sources!(model.terms[i], 
                                        nTerms, nSources, offset, fzero, ione, terms, rowval_array,
                                        colptr_array, nzval_array, cIndex, nIndex, b_array,
                                        faces, cells, cell_faces, cell_neighbours, cell_nsign, integer,
                                        float, backend, runtime, prev)
                    KernelAbstractions.synchronize(backend)
                end
        
                # Free unneeded backend memory
                sources_field = nothing
                sources_sign = nothing 
                nzval_array = nothing
                rowval_array = nothing
                colptr_array = nothing
        
                # Run sources calculations on all sources
                kernel! = sources!(backend)
                for i in 1:nSources
                    (; field, sign) = sources[i]
                    kernel!(field, sign, cells, b_array, ndrange = length(cells))
                    KernelAbstractions.synchronize(backend)
                end
                nothing
            # end
        
        
        # function sparse_array_deconstructor(arr::SparseArrays.SparseMatrixCSC)
        #     (; rowval, colptr, nzval) = arr
        #     return rowval, colptr, nzval
        # end
        
        # function sparse_array_deconstructor(arr::CUDA.CUSPARSE.CuSparseMatrixCSC)
        #     (; rowVal, colPtr, nzVal) = arr
        #     return rowVal, colPtr, nzVal
        # end
        
        @kernel function set_nzval(nzval, fzero)
            i = @index(Global)
        
            @inbounds begin
                nzval[i] = fzero
            end
        end
        
        # function check_for_precon!(nzval, precon::Tuple, backend)
        #     nothing
        # end
        
        # function check_for_precon!(nzval, precon, backend)
        #     (; A) = precon
        #     A_precon = A
        #     copy_to_precon!(nzval, A_precon, backend)
        # end
        
        # function copy_to_precon!(nzval, A::SparseArrays.SparseMatrixCSC, backend)
        #     nothing
        # end
        
        # function copy_to_precon!(nzval, A::CUDA.CUSPARSE.CuSparseMatrixCSC, backend)
        #     rowval_precon, colptr_precon, nzval_precon = sparse_array_deconstructor(A)
        
        #     kernel! = copy_to_precon_kernel!(backend)
        #     kernel!(nzval, nzval_precon, ndrange = length(nzval))
        # end
        
        # @kernel function copy_to_precon_kernel!(nzval, nzval_precon)
        #     i = @index(Global)
        
        #     @inbounds begin
        #         nzval_precon[i] = nzval[i]
        #     end
        # end
        using SparseArrays
        _nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
        _nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval
        
        _colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
        _colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr
        
        _rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
        _rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval 





        # apply_boundary_conditions!(uy_eqn, U.y.BCs)
        # # uy_eqn.b .-= divUTy
        # implicit_relaxation!(uy_eqn, prev, solvers.U.relax, mesh)
        # update_preconditioner!(uy_eqn.preconditioner, mesh)
        # run!(uy_eqn, solvers.U, U.y)
        # residual!(R_uy, uy_eqn.equation, U.y, iteration)
          
        # inverse_diagonal!(rD, ux_eqn.equation)
        # interpolate!(rDf, rD)
        # remove_pressure_source!(ux_eqn, uy_eqn, ∇p)
        # H!(Hv, U, ux_eqn, uy_eqn)
        
        # interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        # correct_boundaries!(Uf, Hv, U.BCs)
        # div!(divHv, Uf)
        
        # @. prev = p.values
        # discretise!(p_eqn, prev, runtime)
        # apply_boundary_conditions!(p_eqn, p.BCs)
        # setReference!(p_eqn, nothing, 1)
        # update_preconditioner!(p_eqn.preconditioner, mesh)
        # run!(p_eqn, solvers.p, p)

        # explicit_relaxation!(p, prev, solvers.p.relax)
        # residual!(R_p, p_eqn.equation, p, i)

        # grad!(∇p, pf, p, p.BCs) 

        # correct = false
        # if correct
        #     ncorrectors = 1
        #     for i ∈ 1:ncorrectors
        #         discretise!(p_eqn)
        #         apply_boundary_conditions!(p_eqn, p.BCs)
        #         setReference!(p_eqn.equation, pref, 1)
        #         # grad!(∇p, pf, p, pBCs) 
        #         interpolate!(gradpf, ∇p, p)
        #         nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
        #         correct!(p_eqn.equation, p_model.terms.term1, pf)
        #         run!(p_model, solvers.p)
        #         grad!(∇p, pf, p, pBCs) 
        #     end
        # end

        # correct_velocity!(U, Hv, ∇p, rD)
        # interpolate!(Uf, U)
        # correct_boundaries!(Uf, U, U.BCs)
        # flux!(mdotf, Uf)
    # end

    xncpu = ux_eqn.equation.A.nzval
    xpreconcpu = ux_eqn.preconditioner.A.nzval
    xbcpu = ux_eqn.equation.b

    yncpu = uy_eqn.equation.A.nzVal
    ypreconcpu = uy_eqn.preconditioner.A.nzVal
    ybcpu = uy_eqn.equation.b

    (; A) = uy_eqn.equation
    nzval_array = _nzval(A)

    kernel! = test(backend)
    kernel!(nzval_array, ndrange = length(nzval_array))
    nzval_array

    @kernel function test(nzval_test)
        i = @index(Global)

        nzval_test[i] += nzval_test[i] + 1
    end


    pcpu = p_eqn.equation.A.nzval
    ppreconcpu = p_eqn.preconditioner.A.nzval
    # npreconcpu = p_eqn.preconditioner.A.nzval
    # pstoragecpu = p_eqn.preconditioner.storage
    # xstoragecpu = ux_eqn.preconditioner.storage
    # ystoragecpu = uy_eqn.preconditioner.storage
    # valuescpu = U.x.values
    # Ruycpu = R_uy
    # valscpu = rD.values
    # vals2cpu = rDf.values
    # vals = divHv.values
    # yvals = Uf.y.values
    cyncpu
    uy_eqn.equation.A.nzVal
    uy_eqn.preconditioner.A.nzVal

    error_check(xncpu, ux_eqn.equation.A.nzVal, 10^-15)
    error_check(xpreconcpu, ux_eqn.preconditioner.A.nzVal, 10^-15)
    error_check(xbcpu, ux_eqn.equation.b, 10^-15)


    error_check(yncpu, uy_eqn.equation.A.nzVal, 0)
    error_check(ypreconcpu, uy_eqn.preconditioner.A.nzVal, 0)
    error_check(ybcpu, uy_eqn.equation.b, 10^-15)

    function error_check(arr1, arr2, target)
        CUDA.allowscalar(true)
        sum = 0
        for i in eachindex(arr1)
            diff = arr1[i]-arr2[i]
            if abs(diff) > target
                sum += 1
                println("At index $i: diff = $diff")
            end
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


    model_cpu = similar(model)