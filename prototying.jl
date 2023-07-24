using Plots

using FVM_1D

using LinearAlgebra
using SparseArrays
using Krylov
using LinearOperators
using ILUZero
using LoopVectorization
using BenchmarkTools


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

Pu = set_preconditioner(DILU(), ux_eqn, ux_model, uxBCs)

Da = zeros(eltype(A), m)
b = ones(eltype(A), m)

# @benchmark dilu_diagonal2!(Pu) # 11.615 ms, 23 ms, 48.687 μs, 14.14 μs

@time extract_diagonal!(Da, Pu.storage.Di, A)
Da
@time dilu_diagonal2!(Pu)
Da
DDa = Diagonal(Da)
# D_dilu = Pu.storage.D
D_dilu = 1.0./Pu.storage.D
DD = Diagonal(D_dilu)
# rDD = Diagonal(1.0./D_dilu)
La =sparse(LowerTriangular(A - DDa))
Ua = sparse(UpperTriangular(A - DDa)) #- DD1
LL = (La + DD)*inv(DD)
# LL = (La + DD)*rDD
UU = (DD + Ua)
Q = LL*UU
Diagonal(Q).diag

@benchmark c = forward_substitution($A, $D0, $b) # 5.264 ms
@benchmark c = forward_substitution($Pu, $b) # 16 μs
@time c = forward_substitution(Pu, b)

LL*c

@benchmark $d = backward_substitution($Pu, $c) # 6.2 ms
@benchmark $d = backward_substitution($Pu, $c) # 18 μs
c
@time d = backward_substitution(Pu, b)
d
UU*d

c = zeros(eltype(b), length(b))
xx = zeros(eltype(b), length(b))
b
left_div!(xx, Pu.storage, b)

xx
Q*xx

c .= LL\b 
xx .= UU\c
Q*xx

for i ∈ 4:2
    println(i)
end
