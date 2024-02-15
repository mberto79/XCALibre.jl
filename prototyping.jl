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

cu(model)
cu(∇p)
cu(ux_eqn)
cu(uy_eqn)
cu(p_eqn)

# model constructor
function model_to_GPU(model)

    # Deconstruct model to constituent variables and individually transfer them to GPU
    (; U, energy, model, p, boundary_info, mesh, nu, turbulence) = model
    energy = cu(energy)
    model = cu(model)
    p = cu(p)
    boundary_info = cu(boundary_info)
    mesh = cu(mesh)
    nu = cu(nu)
    turbulence = cu(turbulence)

    # Deconstruct U to constituent parts and transfer only x, y, z to GPU
    # BCs cannot be transferred to GPU as it is not scalar or vector
    (; x, y, z, BCs) = U
    x = cu(x)
    y = cu(y)
    z = cu(z)

    # Reconstruct U with GPU-based variables
    U = VectorField(
        x,
        y,
        z,
        mesh,
        BCs
    )

    # Reconstruct model with GPU-based variables
    model = RANS(
        model,
        U,
        p,
        nu,
        turbulence,
        energy,
        mesh,
        boundary_info
    )

    return model
end

# function ∇p_to_GPU(∇p)

    # Deconstruct ∇p to constitutent variables and move them to GPU
    # (; correct, correctors, field, mesh, result) = ∇p 
    # correct = cu(correct)
    # correctors = cu(correctors)
    # field = cu(field)
    # mesh = cu(mesh)
    # result = cu(result)

    # S = typeof(schemes.p.gradient)
    # F = typeof(field)
    # R = typeof(result)
    # Int = _get_int(mesh)
    # M = typeof(mesh)


    # # Reconstruct ∇p with GPU-based variables
    # ∇p = Grad{S,F,R,I,M}(
    #     field,
    #     result,
    #     correctors,
    #     correct,
    #     mesh
    # )

    # return ∇p
# end

model = model_to_GPU(model)
∇p = Grad{schemes.p.gradient}(cu(p)) #make this multiple dispatch earlier in code

#equation to GPU
function Equation_to_GPU(eqn, mesh, Precon::PT) where PT<:PreconditionerType
    
    # Deconstruct equation to constituent parts
    (; equation, model, preconditioner, solver) = eqn
    
    # Equation constructor
    equation = cu(equation)

    # (; A, Fx, R, b) = equation
    # A = cu(A) 
    # b = cu(b)
    # R = cu(R)
    # Fx = cu(Fx)

    # SMCSC = typeof(A)
    # VTf = typeof(b)

    # Equation{SMCSC,VTf}(
    #     A,
    #     b,
    #     R,
    #     Fx
    #     )

    # Model constructor
    (; sources, terms) = model

    # Defining array to get around terms indexing issue during deconstruction and reconstruction
    # Completed as "type" variable cannot be converted to GPU
    new_terms = []
    # Loop to convert variables to GPU
    for i in eachindex(terms)
        # Deconstructing the tuple
        (; flux, phi, sign, type) = terms[i]
        flux = cu(flux)
        phi = cu(phi)
        sign = cu(sign)

        # Creating a new Operator and adding it to the new_terms array
        push!(new_terms, Operator(flux, phi, sign, type))
    end

    # Reconstructing terms as a tuple
    terms = tuple(new_terms...)

    # Converting sources to GPU
    sources = cu(sources)

    # Retrieving types for model reconstruction
    int = _get_int(mesh)
    T = typeof(terms)
    S = typeof(sources)

    # Reconstructing model
    model = Model{T, S, int, int}(
        terms,
        sources
        )

    # Preconditioner constructor
    (; A, P, storage) = preconditioner
    A = cu(A)
    P = cu(P)
    storage = Storage_to_GPU(storage, Precon)

    preconditioner = Preconditioner{PT,typeof(A),typeof(P),typeof(storage)}(
        A,
        P,
        storage
    )

    # Solver constructor
    cu(solver)

    # Reconstruct equation
    eqn = ModelEquation(
        model,
        equation,
        solver,
        preconditioner
    )

    return eqn
end

# storage constructor
struct ILU0Precon_GPU{N, VN, VT, VM}
        m::N
        n::N
        l_colptr::VN
        l_rowval::VN
        l_nzval::VT
        u_colptr::VN
        u_rowval::VN
        u_nzval::VT
        l_map::VN
        u_map::VN
        wrk::VM
end

function Storage_to_GPU(storage,Precon::ILU0)
    (; m, n, l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map, wrk) = storage
    m = cu(m)
    n = cu(n)
    l_colptr = cu(l_colptr)
    l_rowval = cu(l_rowval)
    l_nzval = cu(l_nzval)
    u_colptr = cu(u_colptr)
    u_rowval = cu(u_rowval)
    u_nzval = cu(u_nzval)
    l_map = cu(l_map)
    u_map = cu(u_map)
    wrk = cu(wrk)

    storage = ILU0Precon_GPU(
        m,
        n,
        l_colptr,
        l_rowval,
        l_nzval,
        u_colptr,
        u_rowval,
        u_nzval,
        l_map,
        u_map,
        wrk
        )

    return storage
end

function Storage_to_GPU(storage, Precon::T) where T<:Union{NormDiagonal, LDL, Jacobi}
    storage = cu(storage)
    return storage
end

function Storage_to_GPU(storage, Precon::DILU)
    (; A, D, Di, Ri, J) = S
    A = cu(A)
    D = cu(D)
    Di = cu(Di)
    
    Ri_temp = []
    for i in eachindex(Ri)
        temp = cu(Ri[i])
        push!(Ri_temp, temp)
    end
    Ri = [Ri_temp...]
    
    J_temp = []
    for i in eachindex(J)
        temp = cu(J[i])
        push!(J_temp, temp)
    end
    J = [J_temp...]
    
    S = DILUprecon(
        A,
        D,
        Di,
        Ri,
        J
    )
end

# # DILU preconditioner constructor
# using SparseArrays
# A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])
# m, n = size(A)
# m == n || throw("Matrix not square")
# D = zeros(Float32, m)
# Di = zeros(Int32, m)
# diagonal_indices!(Di, A)
# @time Ri, J = upper_row_indices(A, Di)
# S = DILUprecon(A, D, Di, Ri, J)
# P  = LinearOperator(
#     Float32, m, n, false, false, (y, v) -> ldiv!(y, S, v)
#     )
# prep = Preconditioner{DILU,typeof(A),typeof(P),typeof(S)}(A,P,S)

ux_eqn = Equation_to_GPU(ux_eqn, mesh, solvers.U.preconditioner)
uy_eqn = Equation_to_GPU(uy_eqn, mesh, solvers.U.preconditioner)
p_eqn = Equation_to_GPU(p_eqn, mesh, solvers.p.preconditioner)
turbulence = cu(turbulence)
config = cu(config)

@kernel function test3(var)
    i = @index(Global)
    (; cells,
    cell_nodes,
    cell_faces,
    cell_neighbours,
    cell_nsign,
    faces,
    face_nodes,
    boundaries,
    nodes,
    node_cells,
    get_float,
    get_int) = var

    @inbounds cell_neighbours[i] = cell_neighbours[i]+cell_neighbours[i]

    var = Mesh2(cells,
    cell_nodes,
    cell_faces,
    cell_neighbours,
    cell_nsign,
    faces,
    face_nodes,
    boundaries,
    nodes,
    node_cells,
    get_float,
    get_int)

end

backend = _get_backend(model.mesh)
kernel = test3(backend)
kernel(ux_eqn.model.terms[1].phi.mesh, ndrange = length(ux_eqn.model.terms[1].phi.mesh.cells))
ux_eqn.model.terms[1].phi.mesh.cell_neighbours


# R_ux, R_uy, R_p  = SIMPLE_loop(
    # model, ∇p, ux_eqn, uy_eqn, p_eqn, turbulence, config ; resume=resume, pref=pref)


    ## ADAPT TESTING
import Adapt
function Adapt.adapt_structure(to, itp::test)
    test()
end

struct test end

T = test()

cu(T)

struct test0{F,T} 
    Bool::T
end

using Adapt
using CUDA

function Adapt.adapt_structure(to, itp::test0{F,T}) where {F,T}
    Bool = Adapt.adapt_structure(to, itp.Bool)
    test0{F,T}(Bool)
end

a = 1
F = Float32
T = typeof(a)

test0{F}(val) where {F} = begin
    T = typeof(val)
    test0{F,T}(val)
end

T = test0{Float32}(1)

cu(T)