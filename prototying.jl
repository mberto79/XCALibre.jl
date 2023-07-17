using Plots

using FVM_1D

using LinearAlgebra
using SparseArrays
using Krylov
using LinearOperators
using ILUZero
using LoopVectorization


# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

p = ScalarField(mesh)
U = VectorField(mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = (
    Dirichlet(U, :inlet, velocity),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, [0.0, 0.0, 0.0]),
    Dirichlet(U, :top, [0.0, 0.0, 0.0])
    # Neumann(U, :top, 0.0)
    )

uxBCs = (
    Dirichlet(U, :inlet, velocity[1]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, 0.0),
    Dirichlet(U, :top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(U, :inlet, velocity[2]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, 0.0),
    Dirichlet(U, :top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(p, :inlet, 0.0),
    Dirichlet(p, :outlet, 0.0),
    Neumann(p, :wall, 0.0),
    Neumann(p, :top, 0.0)
)

setup_U = SolverSetup(
    solver      = BicgstabSolver,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-2
)

using Profile, PProf

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)


# Pre-allocate fields
ux = ScalarField(mesh)
uy = ScalarField(mesh)
∇p = Grad{Linear}(p)
mdotf = FaceScalarField(mesh)
nuf = ConstantScalar(nu) # Implement constant field! Priority 1
rDf = FaceScalarField(mesh)
divHv_new = ScalarField(mesh)


# Define models 
ux_model = (
    Divergence{Linear}(mdotf, ux) - Laplacian{Linear}(nuf, ux) 
    == 
    Source(∇p.x)
)

uy_model = (
    Divergence{Linear}(mdotf, uy) - Laplacian{Linear}(nuf, uy) 
    == 
    Source(∇p.y)
)

p_model = (
    Laplacian{Linear}(rDf, p) == Source(divHv_new)
)

# Extract model variables
ux = ux_model.terms[1].phi
mdotf = ux_model.terms[1].flux
uy = uy_model.terms[1].phi
nuf = ux_model.terms[2].flux
rDf = p_model.terms[1].flux 
rDf.values .= 1.0
divHv_new = ScalarField(p_model.sources[1].field, mesh)

# Define aux fields 
n_cells = m = n = length(mesh.cells)

# U = VectorField(mesh)
Uf = FaceVectorField(mesh)
# mdot = ScalarField(mesh)

pf = FaceScalarField(mesh)
# ∇p = Grad{Midpoint}(p)
gradpf = FaceVectorField(mesh)

Hv = VectorField(mesh)
Hvf = FaceVectorField(mesh)
Hv_flux = FaceScalarField(mesh)
divHv = Div(Hv, FaceVectorField(mesh), zeros(Float64, n_cells), mesh)
rD = ScalarField(mesh)

# Define equations
ux_eqn  = Equation(mesh)
uy_eqn  = Equation(mesh)
p_eqn    = Equation(mesh)

n_cells = m = n = length(mesh.cells)

# Define preconditioners and linear operators
opAx = LinearOperator(ux_eqn.A)
opAy = LinearOperator(uy_eqn.A)
opAp = LinearOperator(p_eqn.A)

discretise!(ux_eqn, ux_model)
apply_boundary_conditions!(ux_eqn, ux_model, uxBCs)
Pu = set_preconditioner(NormDiagonal(), ux_eqn, ux_model, uxBCs)
Pp = set_preconditioner(LDL(), p_eqn, p_model, pBCs)

A = ux_eqn.A
Da = Diagonal(A)
m, n = size(A)

D0 = zeros(eltype(A), m)
D1 = zeros(eltype(A), m)
D2 = zeros(eltype(A), m)
b = ones(eltype(A), m)

x = A\b
A*x

extract_diagonal!(D0, A)
@time extract_diagonal!(D1, A)
@time dilu_diagonal1!(D1,A)
@time dilu_diagonal2!(D2,A)

DD1 = Diagonal(D0)
La =LowerTriangular(A) - DD1 + I
Ua = UpperTriangular(A) #- DD1

DD1i =inv(DD1)
L1 = (La + DD1)*DD1i# *(DD1 + Ua)
U1 = (DD1 + Ua)
L1*U1
c1 = L1\b
x1 = U1\c1
L1*U1*x1

DD2 = Diagonal(D2)
DD2i =inv(DD2)
L2 = (La + DD2)*DD2i# *(DD1 + Ua)
U2 = (DD2 + Ua)
L2*U2

c2 = L2\b
x2 = U2\c2

A*x2

x1 = zeros(eltype(A), m)
left_div!(x1, A, D1, b)
x1

P = Preconditioner{DILU}(A)
dilu_diagonal2!(P.storage.diagonal,P.A)
ldiv!(x, P.storage, b)
x

P.P*ones(m)