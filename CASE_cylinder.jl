using Plots

using FVM_1D

using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# Fields 
p = ScalarField(mesh)
U = VectorField(mesh)

# BOUNDARY CONDITIONS 

velocity = [0.1, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

UBCs = ( 
    Dirichlet(U, :inlet, velocity),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :cylinder, noSlip),
    Neumann(U, :bottom, 0.0),
    Neumann(U, :top, 0.0)
)

uxBCs = (
    Dirichlet(U, :inlet, velocity[1]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :cylinder, noSlip[1]),
    Neumann(U, :bottom, 0.0),
    Neumann(U, :top, 0.0)
)

uyBCs = (
    Dirichlet(U, :inlet, velocity[2]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :cylinder, noSlip[2]),
    Neumann(U, :bottom, 0.0),
    Neumann(U, :top, 0.0)
)

pBCs = (
    Neumann(p, :inlet, 0.0),
    Dirichlet(p, :outlet, 0.0),
    Neumann(p, :cylinder, 0.0),
    Neumann(p, :bottom, 0.0),
    Neumann(p, :top, 0.0)
)

setup_U = SolverSetup(
    solver      = GmresSolver,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-1
)

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 1000
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

plot(; xlims=(0,1000), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10)
plot!(1:length(Ry), Ry, yscale=:log10)
plot!(1:length(Rp), Rp, yscale=:log10)