using Plots
using FVM_1D
using Krylov
using CUDA
# using GPUArrays

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)
mesh = mesh |> cu

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

CUDA.allowscalar(false)

Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

## Vector Field function
VectorField_test(mesh::Mesh2) = begin
    ncells = length(mesh.cells)
    F = eltype(mesh.nodes[1].coords)
    VectorField(
        ScalarField(zeros(F, ncells), mesh, ()),
        ScalarField(zeros(F, ncells), mesh, ()), 
        ScalarField(zeros(F, ncells), mesh, ()), 
        mesh,
        () # to hold x, y, z and combined BCs
        )
end


using Plots
using FVM_1D
using Krylov
using KernelAbstractions, CUDA

mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)
mesh = mesh |> cu

_get_float(mesh)
_get_int(mesh)
_get_backend(mesh)
