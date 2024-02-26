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
    # type = typeof(ux_eqn)
    # println("$type")

    # CUDA.allowscalar(true)
    ux_eqn.equation.b
    ux_eqn.equation.A.nzval
    # _discretise!(ux_eqn.model, ux_eqn, prev, runtime)

    ux_eqn.equation.b
    ux_eqn.equation.A.nzval

    using SparseArrays
    using CUDA
    using KernelAbstractions

    # @generated function _discretise!(
    #     model::Model{TN,SN,T,S}, eqn, prev, runtime
    #     ) where {TN,SN,T,S}
    
        # assignment_block_1 = Expr[] # Ap
        # assignment_block_2 = Expr[] # An or b
        # assignment_block_3 = Expr[] # b (sources)
    
        # # Loop for operators
        # for t ∈ 1:nTerms
        #     function_call = quote
        #         scheme!(
        #             model.terms[$t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
        #             )
        #     end
        #     push!(assignment_block_1, function_call)
        #     # _convert_array(assignment_block_1, backend)
        #     # type = typeof(assignment_block_1)
        #     # println("$type")
    
        #     assign_source = quote
        #         scheme_source!(model.terms[$t], b, nzval, cell, cID, cIndex, prev, runtime)
        #     end
        #     push!(assignment_block_2, assign_source)
        #     # backend = _get_backend(mesh)
        #     # _convert_array(assignment_block_2, backend)
        # end
    
        # # Loop for sources
        # for s ∈ 1:nSources
        #     add_source = quote
        #         (; field, sign) = model.sources[$s]
        #         b[cID] += sign*field[cID]*volume
        #     end
        #     push!(assignment_block_3, add_source)
        #     # backend = _get_backend(mesh)
        #     # _convert_array(assignment_block_3, backend)
        # end
    
        # quote
            nTerms = 3
            nSources = 1

            model = ux_eqn.model
            eqn = ux_eqn

            (; A, b) = eqn.equation
            (; terms, sources) = model
            mesh = model.terms[1].phi.mesh
            integer = _get_int(mesh)
            float = _get_float(mesh)
            backend = _get_backend(mesh)
            # (; faces, cells, ) = mesh
            (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
            # (; rowval, colptr, nzval) = A
            rowval, colptr, nzval = sparse_array_deconstructor(A)
            fzero = zero(float) # replace with func to return mesh type (Mesh module)
            ione = one(integer)
            # @inbounds for i ∈ eachindex(nzval)
            #     nzval[i] = fzero
            # end

            kernel! = set_nzval(backend)
            kernel!(nzval, fzero, ndrange = length(nzval))

            # CUDA.allowscalar(true)
            # start = colptr[1]
            # offset = findfirst(isequal(1),@view rowval[start:end]) - ione
            # typeof(offset)

            cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
            nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
            offset = zero(integer)

            CUDA.allowscalar(false)

            sources_field = Array{typeof(sources[1].field)}(undef, length(sources))
            sources_sign = Array{typeof(sources[1].sign)}(undef, length(sources))

            for i in eachindex(sources)
                sources_field[i] = sources[i].field
                sources_sign[i] = sources[i].sign
            end

            sources_field = _convert_array(sources_field, backend)
            sources_sign = _convert_array(sources_sign, backend)

            kernel! = schemes_and_sources!(backend)
            kernel!(nTerms, nSources, offset, fzero, ione, terms, sources_field,
                    sources_sign, rowval, colptr, nzval, cIndex, nIndex,  b, faces,
                    cells, cell_faces, cell_neighbours, cell_nsign, integer, float,
                    backend, ndrange = length(cells))



    @kernel function schemes_and_sources!(nTerms, nSources, offset, fzero, ione, terms, sources_field, sources_sign, rowval, colptr, nzval, cIndex, nIndex, b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer, float, backend)
        i = @index(Global)
        # (; terms) = model

        @inbounds begin
            cell = cells[i]
            (; faces_range, volume) = cell

            for fi in faces_range
                fID = cell_faces[fi]
                ns = cell_nsign[fi] # normal sign
                face = faces[fID]
                nID = cell_neighbours[fi]
                cellN = cells[nID]

                start = colptr[i]
                # offset = findfirst(isequal(i),@view rowval[start:end]) - ione
                for j in start:length(rowval)
                    val = rowval[start + j - ione]
                    if val == i
                        offset = j - ione
                        break
                    end
                end
                cIndex = start + offset

                start = colptr[nID]
                # offset = findfirst(isequal(i),@view rowval[start:end]) - ione
                for j in start:length(rowval)
                    val = rowval[start + j - 1]
                    if val == i
                        offset = j - ione
                        break
                    end
                end
                nIndex = start + offset

                for t ∈ 1:nTerms
                    # @cushow(t)
                    scheme!(terms[t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime)
                end

            end

        
        # b[i] = fzero

        # for t ∈ 1:nTerms
        #     scheme_source!(terms[t], b, nzval, cell, cID, cIndex, prev, runtime)
        # end

        # for s ∈ 1:nSources
        #     (; field, sign) = sources[s]
        #     b[i] += sign*field[i]*volume
        # end

        end
    end


    # Loop for terms
    for t ∈ 1:nTerms
        scheme!(model.terms[t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime)
        scheme_source!(model.terms[t], b, nzval, cell, cID, cIndex, prev, runtime)
    end

    # Loop for sources
    for s ∈ 1:nSources
        (; field, sign) = model.sources[s]
        b[cID] += sign*field[cID]*volume
    end
    
    
    function sparse_array_deconstructor(arr::SparseArrays.SparseMatrixCSC)
        (; rowval, colptr, nzval) = arr
        return rowval, colptr, nzval
    end
    
    function sparse_array_deconstructor(arr::CUDA.CUSPARSE.CuSparseMatrixCSC)
        (; rowVal, colPtr, nzVal) = arr
        return rowVal, colPtr, nzVal
    end

    @kernel function set_nzval(nzval, fzero)
        i = @index(Global)
    
        @inbounds begin
            nzval[i] = fzero
        end
    end



    a = [1 2 3 4]
    res = add(a...)

    function add(a,b,c,d)
        return a + b + c +d
    end


    @kernel function test10(tuple, res)
        i = @index(Global)

        start_index = 2

        # @inbounds begin
            for j in start_index:length(tuple) # Use enumerate with the sliced tuple
                val = tuple[start_index + j - 1]
                if val == 4
                    res[1] = j - 1  # Adjust the index to account for the starting index
                    # break    # Break out of the loop since we found the first occurrence
                end
            end
        # end
    end

    arr = [0 2 4 5 6 7 8 9]
    start = 4
    for i in eachindex(arr[start:end])
        val = arr[i+start-1]
        println("$val")
    end
    isequal(10)(10)

    tuple = cu([0,1,2,3,4,10,6])
    res = CUDA.zeros(Int64, 1)
    kernel = test10(backend)
    @device_code_warntype kernel(tuple, res, ndrange = 1)
    res



    tuple = [0,1,2,3,4,10,6]

    start = 2

    findfirst(isequal(4),@view tuple[start:end]) - ione

    tuple = [0,1,2,3,4,10,6]
    res = nothing  # Initialize res outside the loop
    
    start_index = 2  # Define the starting index
    
    for (j, val) in enumerate(tuple[start_index:end])  # Use enumerate with the sliced tuple
        if val == 4
            res = j - ione  # Adjust the index to account for the starting index
            break    # Break out of the loop since we found the first occurrence
        end
    end
    
    println(res)

