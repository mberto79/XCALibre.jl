using Plots

using FVM_1D

using Krylov


# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = build_mesh(mesh_file, scale=0.001)

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