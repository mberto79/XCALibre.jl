using Plots

using FVM_1D

using LinearAlgebra
using Krylov
using LinearOperators
using ILUZero
using LoopVectorization


# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# struct Mesh2{I,F} <: AbstractMesh
#     cells::Vector{Cell{I,F}}
#     faces::Vector{Face2D{I,F}}
#     boundaries::Vector{Boundary{I}}
#     nodes::Vector{Node{F}}
# end

# mesh = Mesh2(mesh.cells, mesh.faces, (mesh.boundaries...), mesh.nodes)

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
model_ux = (
    Divergence{Linear}(mdotf, ux) - Laplacian{Linear}(nuf, ux) 
    == 
    Source(∇p.x)
)

model_uy = (
    Divergence{Linear}(mdotf, uy) - Laplacian{Linear}(nuf, uy) 
    == 
    Source(∇p.y)
)

model_p = (
    Laplacian{Linear}(rDf, p) == Source(divHv_new)
)

# Extract model variables
ux = model_ux.terms[1].phi
mdotf = model_ux.terms[1].flux
uy = model_uy.terms[1].phi
nuf = model_ux.terms[2].flux
rDf = model_p.terms[1].flux 
rDf.values .= 1.0
divHv_new = ScalarField(model_p.sources[1].field, mesh)

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

# Pre-allocated auxiliary variables
ux0 = zeros(Float64, n_cells)
uy0 = zeros(Float64, n_cells)
p0 = zeros(Float64, n_cells)

ux0 .= velocity[1]
uy0 .= velocity[2]
p0 .= zero(Float64)


discretise!(p_eqn, model_p)

p_eqn.A[10,10]
Ad = Diagonal(p_eqn.A)

Ad[10,10]