using Plots

using FVM_1D.Mesh2D
using FVM_1D.UNV
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK

using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# BOUNDARY CONDITIONS 

velocity = [0.1, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Dirichlet(:bottom, velocity),
    Dirichlet(:top, velocity)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip[1]),
    Dirichlet(:bottom, velocity[1]),
    Dirichlet(:top, velocity[1])
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip[2]),
    Dirichlet(:bottom, velocity[2]),
    Dirichlet(:top, velocity[2])
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

setup_U = SolverSetup(
    iterations  = 1,
    solver      = BicgstabSolver,
    tolerance   = 1e-1,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    iterations  = 1,
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    tolerance   = 1e-1,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-2
)

GC.gc()

ux = ScalarField(mesh)
uy = ScalarField(mesh)
p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 1000
Rx, U = isimple!(
    mesh, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations
)

write_vtk("results", mesh, ("U", U), ("p", p))

# plotly(size=(400,400), markersize=1, markerstrokewidth=1)
niterations = length(Rx)
plot(collect(1:niterations), Rx[1:niterations], yscale=:log10)
