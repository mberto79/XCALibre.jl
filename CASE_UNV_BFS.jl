using Plots

using FVM_1D

using Krylov


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
    solver      = GmresSolver, # BicgstabSolver, GmresSolver
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, # GmresSolver, FomSolver, DiomSolver
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-1
)

using Profile, PProf

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 1000
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)
end

PProf.Allocs.pprof()

write_vtk("results", mesh, ("U", U), ("p", p))

plot(; xlims=(0,92))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")