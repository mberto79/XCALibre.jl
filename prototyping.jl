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

# CUDA.allowscalar(false)
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

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)

    sum = 0

    CUDA.allowscalar(true)

    for i in eachindex(U)
        num = U[i][1]
        if num != 0.5
            sum += 1
            println("Sum incremented")
        end
    end

    println("$sum")
    length(U)
    length(Uf.x)
    
    # IDs = Array{typeof(BCs[1].ID)}(undef,length(BCs))
    # values = _get_float(mesh)[]

    function _convert_array(arr, backend::CPU)
        return arr
    end
    function _convert_array(arr, backend::CUDABackend)
        return adapt(CuArray, arr)
    end

    BCs = U.BCs
    psif = Uf
    psi = U
    (; x, y, z, mesh) = psif
    (; boundaries) = mesh 
    

    boundaries_cpu = Array{eltype(mesh.boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)

    backend = _get_backend(psif.mesh)
    kernel_dirichlet = adjust_boundary_dirichlet_vector!(backend)
    kernel_neumann = adjust_boundary_neumann_vector!(backend)

    for i in eachindex(BCs)
        (; ID) = BCs[i]
        (; facesID, cellsID) = boundaries_cpu[ID]
        facesID = _convert_array(facesID, backend)
        cellsID = _convert_array(cellsID, backend)
        #KERNEL LAUNCH
        call_adjust_boundary(BCs[i], psif, psi, facesID, cellsID, x, y, z)
    end

    CUDA.allowscalar(true)

    psif.x[50]
    sum = 0

    for i in eachindex(psi)
        num = psi[i][1]
        if num != 0.5
            sum += 1
        end
    end

    println("$sum") # aim for 158 and 0
    length(x)

    function call_adjust_boundary(BC::Dirichlet, psif::FaceVectorField, psi::VectorField, facesID, cellsID, x, y, z)
        kernel_dirichlet(BC, psif, psi, facesID, cellsID, x, y, z, ndrange = length(facesID))
    end

    function call_adjust_boundary(BC::Neumann, psif, psi, facesID, cellsID, x, y, z)
        kernel_neumann(BC, psif, psi, facesID, cellsID, x, y, z, ndrange = length(facesID))
    end

    @kernel function adjust_boundary_dirichlet_vector!(BC, psif, psi, facesID, cellsID, x, y, z)
        i = @index(Global)
        @inbounds begin
            fID = facesID[i]
            x[fID] = BC.value[1]
            y[fID] = BC.value[2]
            z[fID] = BC.value[3]
        end
    end

    @kernel function adjust_boundary_neumann_vector!(BC, psif, psi, facesID, cellsID, x, y, z)
        i = @index(Global)
        @inbounds begin
            fID = facesID[i]
            cID = cellsID[i]
            psi_cell = psi[cID]
            # normal = faces[fID].normal
            # Line below needs sorting out for general user-defined gradients
            # now only works for zero gradient
            # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
            x[fID] = psi_cell[1]
            y[fID] = psi_cell[2]
            z[fID] = psi_cell[3]
        end
    end

    function adjust_boundary!( 
        BC::Neumann, psif::FaceVectorField, psi::VectorField, boundary, faces
        ) 
    
        (; x, y, z) = psif
        (; facesID, cellsID) = boundary
    
        @inbounds for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            cID = cellsID[fi]
            psi_cell = psi[cID]
            # normal = faces[fID].normal
            # Line below needs sorting out for general user-defined gradients
            # now only works for zero gradient
            # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
            x[fID] = psi_cell[1]
            y[fID] = psi_cell[2]
            z[fID] = psi_cell[3]
        end
    end


    CUDA.allowscalar(true)

    (; facesID) = mesh.boundaries[IDs[1]]


    values


    kernel(BC, IDs, ndrange = length(IDs))

    @kernel function adjust_boundary(BC::Dirichlet, IDs)
        i = @index(Global)
        
        # boundary = boundaries[ID]

        ID = IDs[i]

        @cushow(ID)
        
        # adjust_boundary!(BCs[i], phif, phi, boundary, faces)
    end
    
    function adjust_boundary!( 
        BC::Dirichlet, psif::FaceVectorField, psi::VectorField, boundary, faces
        )
    
        (; x, y, z) = psif
        (; facesID) = boundary
    
        @inbounds for fID ∈ facesID
            x[fID] = BC.value[1]
            y[fID] = BC.value[2]
            z[fID] = BC.value[3]
        end
    end

    U.BCs.ID
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)



## Tuple Test

using KernelAbstractions
using Adapt
using CUDA

A = (1, 2, 3, 4, 5)
typeof(A)

A = cu(A)
result = CUDA.zeros(eltype(A), length(A))

backend = get_backend(result)
kernel! = test1(backend)
kernel!(A, result, ndrange = length(A))

result

@kernel function test1(A, result)
    i = @index(Global)

    @inbounds result[i] = A[i]

end


struct tuple_parent1{V<:AbstractArray{tuple_test}}
    tuple_test::V
    int1::Integer
    int2::Integer
end
Adapt.@adapt_structure tuple_parent1

struct tuple_test{S<:Symbol, NT<:NTuple}
    name::S
    vec1::NT
    vec2::NT
end
Adapt.@adapt_structure tuple_test

child = tuple_test[]

for i in 1:5
    name = Symbol("child")
    I = rand(5:10)
    type = NTuple{I, Integer}
    vec1 = type(rand(1:10) for j in 1:I)
    vec2 = type(rand(1:10) for j in 1:I)
    push!(child, tuple_test(name, vec1, vec2))
end

child

parent = tuple_parent1(child, 100, 200)

parent.tuple_test = adapt(CuArray, parent.tuple_test)

parent = adapt(CuArray, parent)
tuple = parent.tuple_test

result1 = CUDA.zeros(Int64, 5)
result2 = similar(result1)

backend = get_backend(tuple[1].vec1)
kernel! = tuple_test_kernel(backend)
@device_code_warntype kernel!(tuple, result1, result2, ndrange = length(tuple))


@kernel function tuple_test_kernel(tuple, result1, result2)
    i = @index(Global)

    @inbounds begin
        (; vec1, vec2) = tuple[i]
        result1[i] = length(vec1)
        result2[i] = length(vec2)
    end
end



A = CUDA.zeros(Int64, 100)
B = []
for i in 1:10
    push!(B, A)
end
B
cu(B)