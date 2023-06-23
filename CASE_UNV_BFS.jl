using Plots

using FVM_1D

using Krylov


# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
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
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))