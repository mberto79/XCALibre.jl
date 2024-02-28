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
    # model = adapt(CuArray, model)
    # ∇p = adapt(CuArray, ∇p)
    # ux_eqn = adapt(CuArray, ux_eqn)
    # uy_eqn = adapt(CuArray, uy_eqn)
    # p_eqn = adapt(CuArray, p_eqn)
    # turbulence = adapt(CuArray, turbulence)
    # config = adapt(CuArray, config)
    
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

    # Uf = adapt(CuArray,Uf)
    # rDf = adapt(CuArray, rDf)
    # rD = adapt(CuArray, rD)
    # pf = adapt(CuArray, pf)

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)


    # nueff.values

    update_nueff!(nueff, nu, turbulence)

    # prev = adapt(CuArray, prev)

    @. prev = U.x.values
    discretise!(ux_eqn, prev, runtime)
    # ux_eqn.equation.A.nzVal
    # ux_eqn.equation.b
    @time begin apply_boundary_conditions!(ux_eqn, U.x.BCs) end
    # ux_eqn.equation.A.nzVal
    # ux_eqn.equation.b

    eqn = ux_eqn
    model = ux_eqn.model
    BCs = U.x.BCs

    (; A, b) = eqn.equation
    mesh = model.terms[1].phi.mesh
    (; boundaries, faces, cells, boundary_cellsID, cell_neighbours) = mesh
    rowval, colptr, nzval = sparse_array_deconstructor(A)
    backend = _get_backend(mesh)
    integer = _get_int(mesh)
    ione = one(integer)
    kernel! = Dirichlet_laplacian_linear!(backend)
    kernel!(model.terms[3], BCs[1], ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)

    @kernel function Dirichlet_laplacian_linear!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
        i = @index(Global)
        i = BC.ID
        # @cushow i
        
        @inbounds begin
            (; IDs_range) = boundaries[i]
            for k ∈ IDs_range 
                faceID = k
                cellID = boundary_cellsID[k]
                face = faces[faceID]
                cell = cells[cellID] 
                @cushow cellID
                # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
                J = term.flux[faceID]
                (; area, delta) = face 
                flux = J*area/delta
                ap = term.sign[1]*(-flux)
                # @synchronize
                
                start = colptr[cellID]
                offset = 0
                for j in start:length(rowval)
                    offset += 1
                    if rowval[j] == cellID
                        break
                    end
                end
                nIndex = start + offset - ione

                nzval[nIndex] += ap
                b[cellID] += ap*BC.value
            end
        end
    end

    for i in boundaries[1].IDs_range
        cID = boundary_cellsID[i]
        A[cID,cID] = A[cID,cID] + 0.5
        val = A[cID, cID]
        println("$val")
    end

    i = 2
    nID = boundary_cellsID[i]
    start = colptr[nID]
    # cell = cells[boundary_cellsID[i]]
    # offset = findfirst(isequal(i),@view rowval[start:end]) - ione
    offset = 0
    for j in start:length(rowval)
        offset += 1
        if rowval[j] == i
            break
        end
    end
    cIndex = start + offset - ione
    nzval[cIndex]
    A[nID, nID]

    fi = 20
    cellID = boundary_cellsID[fi]
    start = colptr[cellID]
    # offset = findfirst(isequal(i),@view rowval[start:end]) - ione
    offset = 0
    for j in start:length(rowval)
        offset += 1
        if rowval[j] == cellID
            # nIndex = j
            break
        end
    end
    # nIndex = offset - ione
    nIndex = start + offset - ione
    rowval[nIndex]
    nzval[nIndex]
    A[cellID, cellID]

    1
    2
    3
  160


    rowval
    colptr

    ## Unpacking quote

    using KernelAbstractions
    using StaticArrays

    eqn = ux_eqn
    model = ux_eqn.model
    BCs = U.x.BCs


    (; A, b) = eqn.equation
    mesh = model.terms[1].phi.mesh
    (; boundaries, faces, cells, boundary_cellsID) = mesh

    (bc::Dirichlet)(
        term::Operator{F,P,I,Laplacian{Linear}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I} = begin
            kernel! = Dirichlet_laplacian_linear!(backend)
            kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = length(BCs))
        end

    Execute_apply_boundary_condition_kernel!(BCs[1], model.terms[3], backend, boundaries, faces, cells, boundary_cellsID, A, b)



    backend = _get_backend(mesh)
    # IDs_range_array = Array{UnitRange{_get_int(mesh)}}(undef, length(BCs))
    # IDs_range_array = _convert_array!(IDs_range_array, backend)
    # kernel! = Get_IDs_range4!(backend)
    # kernel!(boundaries, IDs_range_array, ndrange = length(IDs_range_array))





    # @kernel function Get_IDs_range4!(boundaries, IDs_range_array)
    #     i = @index(Global)
        
    #     @inbounds begin
    #         (; IDs_range) = boundaries[i]
    #         IDs_range_array[i] = IDs_range
    #     end
    # end

    function Get_IDs_range(backend::CUDABackend, boundaries)
        IDs_range_array = Array{UnitRange{_get_int(mesh)}}(undef, length(BCs))
        IDs_range_array_GPU = _convert_array!(IDs_range_array, backend)
        kernel! = Get_IDs_range_kernel!(backend)
        kernel!(boundaries, IDs_range_array_GPU, ndrange = length(IDs_range_array))
        copyto!(IDs_range_array, IDs_range_array_GPU)
        IDs_range_array_GPU = nothing
        return IDs_range_array
    end

    function Get_IDs_range(backend::CPU, boundaries)
        IDs_range_array = Array{UnitRange{_get_int(mesh)}}(undef, length(BCs))
        for i in eachindex(IDs_range_array)
            IDs_range_array[i] = boundaries[i].IDs_range
        end
        return IDs_range_array
    end

    boundary_IDs_range = Get_IDs_range(backend, boundaries)

    @kernel function Get_IDs_range_kernel!(boundaries, IDs_range_array)
        i = @index(Global)
        
        @inbounds begin
            (; IDs_range) = boundaries[i]
            IDs_range_array[i] = IDs_range
        end
    end



    # TRANSIENT TERM 
    function Execute_apply_boundary_condition_kernel!(
        bc::AbstractBoundary, term::Operator{F,P,I,Time{T}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I,T}
        nothing
    end

    # LAPLACIAN TERM (NON-UNIFORM)
    function Execute_apply_boundary_condition_kernel!(
        bc::Dirichlet, term::Operator{F,P,I,Laplacian{Linear}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Dirichlet_laplacian_linear!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Dirichlet_laplacian_linear!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            J = term.flux[faceID]
            (; area, delta) = face 
            flux = J*area/delta
            ap = term.sign[1]*(-flux)
            A[cellID,cellID] += ap
            b[cellID] += ap*bc.value
        end
    end

    function Execute_apply_boundary_condition_kernel!(
        bc::Neumann, term::Operator{F,P,I,Laplacian{Linear}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Neumann_laplacian_linear!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Neumann_laplacian_linear!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            phi = term.phi
        end
    end

    # DIVERGENCE TERM (NON-UNIFORM)
    function Execute_apply_boundary_condition_kernel!(
        bc::Dirichlet, term::Operator{F,P,I,Divergence{Linear}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Dirichlet_divergence_linear!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Dirichlet_divergence_linear!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            b[cellID] += term.sign[1]*(-term.flux[faceID]*bc.value)
        end
    end

    function Execute_apply_boundary_condition_kernel!(
        bc::Neumann, term::Operator{F,P,I,Divergence{Linear}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Neumann_divergence_linear!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Neumann_divergence_linear!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            ap = term.sign[1]*(term.flux[faceID])
            A[cellID,cellID] += ap
        end
    end

    function Execute_apply_boundary_condition_kernel!(
        bc::Dirichlet, term::Operator{F,P,I,Divergence{Upwind}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Dirichlet_divergence_upwind!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Dirichlet_divergence_upwind!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            # A[cellID,cellID] += 0.0 
            # b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)

            ap = term.sign[1]*(term.flux[faceID])
            # A[cellID,cellID] += max(ap, 0.0)
            b[cellID] -= ap*bc.value

            # ap = term.sign[1]*(term.flux[fID])
            # b[cellID] += A[cellID,cellID]*bc.value
            # A[cellID,cellID] += A[cellID,cellID]
        end
    end

    function Execute_apply_boundary_condition_kernel!(
        bc::Neumann, term::Operator{F,P,I,Divergence{Upwind}}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        
        kernel! = Neumann_divergence_upwind!(backend)
        kernel!(term, bc, boundaries, faces, cells, boundary_cellsID, A, b, ndrange = 1)
    end

    @kernel function Neumann_divergence_upwind!(term, BC, boundaries, faces, cells, boundary_cellsID, A, b)
        i = @index(Global)
        i = BC.ID
        
        (; IDs_range) = boundaries[i]
        @inbounds for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            phi = term.phi 
            # ap = term.sign[1]*(term.flux[fID])
            # A[cellID,cellID] += ap
            # ap = term.sign[1]*(term.flux[fID])
            # A[cellID,cellID] += ap
            # b[cellID] -= ap*phi[cellID]
            ap = term.sign[1]*(term.flux[fID])
            A[cellID,cellID] += max(ap, 0.0)
            # b[cellID] -= max(ap*phi[cellID], 0.0)
        end
    end

    # IMPLICIT SOURCE

    function Execute_apply_boundary_condition_kernel!(
        bc::Dirichlet, term::Operator{F,P,I,Si}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        nothing
    end

    function Execute_apply_boundary_condition_kernel!(
        bc::Neumann, term::Operator{F,P,I,Si}, 
        backend, boundaries, faces, cells,
        boundary_cellsID, A, b) where {F,P,I}
        nothing
    end

    (; A, b) = eqn.equation
    mesh = model.terms[1].phi.mesh
    (; boundaries, faces, cells, boundary_cellsID) = mesh
    for bci ∈ 1:length(BCs.parameters)
        func_calls = Expr[]
        for t ∈ 1:nTerms 
            call = quote
                (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            end
            push!(func_calls, call)
        end
        assignment_loop = quote
            (; IDs_range) = boundaries[BCs[$bci].ID]
            @inbounds for i ∈ IDs_range
                faceID = i
                cellID = boundary_cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                $(func_calls...)
            end
        end
        push!(assignment_loops, assignment_loop.args...)
    end

    # @generated function _apply_boundary_conditions!(
    #     model::Model{TN,SN,T,S}, BCs::B, eqn) where {T,S,TN,SN,B}
    
        # Unpack terms that make up the model (not sources)
        # nTerms = model.parameters[3]
        nTerms = 3
    
        # Definition of main assignment loop (one per patch)
        assignment_loops = []
        for bci ∈ 1:length(BCs.parameters)
            func_calls = Expr[]
            for t ∈ 1:nTerms 
                call = quote
                    (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
                end
                push!(func_calls, call)
            end
            assignment_loop = quote
                (; IDs_range) = boundaries[BCs[$bci].ID]
                @inbounds for i ∈ IDs_range
                    faceID = i
                    cellID = boundary_cellsID[i]
                    face = faces[faceID]
                    cell = cells[cellID]
                    $(func_calls...)
                end
            end
            push!(assignment_loops, assignment_loop.args...)
        end
    
        quote
        (; A, b) = eqn.equation
        mesh = model.terms[1].phi.mesh
        (; boundaries, faces, cells, boundary_cellsID) = mesh
        $(assignment_loops...)
        nothing
        end
    end