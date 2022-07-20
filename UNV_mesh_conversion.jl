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
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh_file = "unv_sample_meshes/quad100.unv"
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = build_mesh(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    # Neumann(:bottom, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

# CAVITY BOUNDARY CONDITIONS 

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = 1*velocity[1]/nu
UBCs = ( 
    Dirichlet(:inlet, noSlip),
    Dirichlet(:outlet, noSlip),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, velocity)
)

uxBCs = (
    Dirichlet(:inlet, noSlip[1]),
    Dirichlet(:outlet, noSlip[1]),
    Dirichlet(:bottom, noSlip[1]),
    Dirichlet(:top, velocity[1])
)

uyBCs = (
    Dirichlet(:inlet, noSlip[2]),
    Dirichlet(:outlet, noSlip[2]),
    Dirichlet(:bottom, noSlip[2]),
    Dirichlet(:top, velocity[2])
)

pBCs = (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
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

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 3000
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations, pref=0.0)

write_vtk("results", mesh, ("U", U), ("p", p))

# plotly(size=(400,400), markersize=1, markerstrokewidth=1)
niterations = length(Rx)
plot(collect(1:niterations), Rx[1:niterations], yscale=:log10, label="Ux")
plot!(collect(1:niterations), Ry[1:niterations], yscale=:log10, label="Uy")
plot!(collect(1:niterations), Rp[1:niterations], yscale=:log10, label="p")

scatter(xf(mesh), yf(mesh), Uf.x)