using Plots

using FVM_1D

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
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip[1]),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip[2]),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

setup_U = SolverSetup(
    solver      = BicgstabSolver,
    relax       = 0.7,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.3,
    itmax       = 100,
    rtol        = 1e-2
)

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 2000
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

# plotly(size=(400,400), markersize=1, markerstrokewidth=1)
niterations = length(Rx)
plot(collect(1:niterations), Rx[1:niterations], yscale=:log10, label="Ux")
plot!(collect(1:niterations), Ry[1:niterations], yscale=:log10, label="Uy")
plot!(collect(1:niterations), Rp[1:niterations], yscale=:log10, label="p")