using Plots
using FVM_1D
using Krylov
using CUDA
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

# backend = CPU()
backend = CUDABackend()

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_2mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
# mesh = update_mesh_format(mesh; integer=Int32, float=Float32)
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


@info "Extracting configuration and input fields..."
model_in = model
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

# Consider using allocate from KernelAbstractions 
# e.g. allocate(backend, Float32, res, res)
TF = _get_float(mesh)
prev = zeros(TF, n_cells)
prev = _convert_array!(prev, backend) 

# Pre-allocate vectors to hold residuals 

R_ux = ones(TF, iterations)
R_uy = ones(TF, iterations)
R_p = ones(TF, iterations)

# Convert arrays to selected backend

# Uf = _convert_array!(Uf, backend)
# rDf = _convert_array!(rDf, backend)
# rD = _convert_array!(rD, backend)
# pf = _convert_array!(pf, backend)
# Hv = _convert_array!(Hv, backend)
# prev = _convert_array!(prev, backend)

interpolate!(Uf, U)   
correct_boundaries!(Uf, U, U.BCs)
flux!(mdotf, Uf)
grad!(∇p, pf, p, p.BCs)

update_nueff!(nueff, nu, turbulence)

@. prev = U.x.values

discretise_test!(ux_eqn, prev, runtime)

A = _A(ux_eqn.equation)

## TESTING IF NUMBER TYPES WORK IN KERNELS
using Adapt

struct test{TN, SN, VI}
    val::VI
end
function Adapt.adapt_structure(to, itp::test{TN,SN}) where {TN,SN}
    value = Adapt.adapt_structure(to, itp.val); VI = typeof(value)
    test{TN, SN, VI}(value)
end
test{TN,SN}(value) where {TN, SN} = begin
    VI = typeof(value)
    test{TN,SN, VI}(value)
end

Test = test{3,1}([1 2 3 4 5])
Test = adapt(backend, Test)

kernel = test_struct(backend)
kernel(Test, ndrange = length(Test.val))

@kernel function test_struct(Test_sruct)
    i = @index(Global)

    (; val) = Test_sruct

    val[i] += 1
end


## TESTING IF SOURCES WORKS IN KERNELS - BOTH MODEL AND SOURCES WORK WHEN REMOVING TYPE FROM SRC STRUCT
model = ux_eqn.model
sources = model.sources

kernel = source_test4(backend)
kernel(model, ndrange = length(sources[1].field.values))

@kernel function source_test4(model)
    i = @index(Global)

    (; sources) = model
    (; field, sign) = sources[1]
    (; values) = field

    @inbounds begin
        values[i] += sign
    end
end

model = ux_eqn.model

mesh = model.terms[1].phi.mesh

kernel = test2!(backend)
kernel(mesh, ndrange = 1)

## TESTING IF MODEL WORKS WITH NEW SOURCES STRUCTURE
model = ux_eqn.model

## TESTING NZVAL ACCESSOR
kernel = test_nzval1(backend)
kernel(_A(ux_eqn.equation), ndrange = length(ux_eqn.equation.A.nzVal))

@kernel function test_nzval1(A_arr)
    i = @index(Global)

    # A_arr = _A(eqn)
    
    nzval_arr = _nzval(A_arr)

    nzval_arr[i] += 1
end

## DISCRETISE ALTERATIONS

discretise_test!(ux_eqn, prev, runtime)

function discretise_test!(eqn, prev, runtime)
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    integer = _get_int(mesh)
    float = _get_float(mesh)
    fzero = zero(float)
    ione = one(integer)
    # cIndex = zero(integer)
    # nIndex = zero(inetger)

    A_array = _A(eqn)
    b_array = _b(eqn)

    nzval_array = _nzval(A_array)
    rowval_array = _rowval(A_array)
    colptr_array = _colptr(A_array)

    backend = _get_backend(mesh)
    kernel! = _discretise2!(backend)
    @device_code_warntype kernel!(model, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione, ndrange = length(mesh.cells))
end

## GENERATED FUNCTION W/ KERNEL
@kernel function _discretise2!(model, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione)
    i = @index(Global)
    
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh


    @inbounds begin
        cell = cells[i]
        (; faces_range, volume) = cell

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            _scheme!(model, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
        end
        b_array[i] = fzero
        _scheme_source!(model, b_array, nzval_array, cell, i, cIndex, prev, runtime)
        _sources!(model, b_array, volume, i)
    end
end

@generated function _scheme!(model::Model{TN,SN,T,S}, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
    nTerms = TN
    
    assignment_block = Expr[]
    
    for t in 1:nTerms
        function_call_scheme = quote
            scheme!(model.terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
        end
        push!(assignment_block, function_call_scheme)
    end
    

    quote
        $(assignment_block...)
    end
end

@generated function _scheme_source!(model::Model{TN,SN,T,S}, b, nzval_array, cell, cID, cIndex, prev, runtime) where {TN,SN,T,S}
    nTerms = TN
    
    assign_source = Expr[]

    for t in 1:nTerms
        function_call_scheme_source = quote
            scheme_source!(model.terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
        end
        push!(assign_source, function_call_scheme_source)
    end

    quote
        $(assign_source...)
    end
end

@generated function _sources!(model::Model{TN,SN,T,S}, b, volume, cID) where {TN,SN,T,S}
    nSources = SN
    
    add_source = Expr[]

    for s in 1:nSources
        expression_call_sources = quote
            (; field, sign) = model.sources[$s]
            b[cID] += sign*field[cID]*volume
        end
        push!(add_source, expression_call_sources)
    end

    quote
        $(add_source...)
    end
end

## DISCRETISE FUNCTION

# Define variables for function
nTerms = length(model.terms)
nSources = length(model.sources)
mesh = model.terms[1].phi.mesh

# Deconstructors to get lower-level variables for function
# (; A, b) = eqn.equation
(; terms, sources) = model
(; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
backend = _get_backend(mesh)
A_array = _A(eqn)
b_array = _b(eqn)

# Get types and set float(zero) and integer(one)
integer = _get_int(mesh)
float = _get_float(mesh)
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
# println(typeof(eqn))

# Set initial values for indexing of nzval array
cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
offset = zero(integer)

# Set b array to 0
kernel! = set_b!(backend)
kernel!(fzero, b_array, ndrange = length(b_array))
KernelAbstractions.synchronize(backend)

# Run schemes and sources calculations on all terms

for i in 1:nTerms
    schemes_and_sources!(model.terms[i], 
                        nTerms, nSources, offset, fzero, ione, terms, rowval_array,
                        colptr_array, nzval_array, cIndex, nIndex, b_array,
                        faces, cells, cell_faces, cell_neighbours, cell_nsign, integer,
                        float, backend, runtime, prev)
    # KernelAbstractions.synchronize(backend)
end

# Free unneeded backend memory 
nzval_array = nothing
rowval_array = nothing
colptr_array = nothing

# Run sources calculations on all sources
kernel! = sources!(backend)
for i in 1:nSources
    (; field, sign) = sources[i]
    kernel!(field, sign, cells, b_array, ndrange = length(cells))
    # KernelAbstractions.synchronize(backend)
end






arr = [1 2 3]
test(arr)
function test(arr)
    test_gen{length(arr)}(arr)
end


@generated function test_gen(arr) where {TN}

    print_arr = Expr[]

    for i in 1:TN
        print = quote
            println(arr[i])
        end
        push!(print_arr, print)
    end

    quote
       $(print_arr...) 
    end
end










## TESTING ACCESSORS AND GENERATED FUNCTIONS IN KERNELS
backend = CUDABackend()
# backend = CPU()

struct test{VI}
    A::VI
    B::VI
    res::VI
end
Adapt.@adapt_structure test

_A(Struct) = Struct.A
_B(Struct) = Struct.B
_res(Struct) = Struct.res

@kernel function test_gen3(Test)
    i = @index(Global)

    A = _A(Test)
    B = _B(Test)
    res = _res(Test)

    @inbounds begin
        res[i] = gen_add!(A[i],B[i])
    end
end

@generated function gen_add!(a,b)

    assignment_block = Expr[]

    function_call = quote
        add!(a,b)
    end
    push!(assignment_block, function_call)

    quote
        res = $(assignment_block...)
    end
end

function add!(a,b)
    res = a + b
    return res
end


A = ones(Int64, 100)
B = A.+1
res = zeros(Int64, 100)

Test = test(A, B, res)
Test = adapt(backend, Test)

Test.res

kernel! = test_gen3(backend)
kernel!(Test, ndrange = length(Test.A))

Test.res