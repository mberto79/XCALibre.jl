using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

U = VectorField(mesh)
p = ScalarField(mesh)
k = ScalarField(mesh)
ω = ScalarField(mesh)
νt = ScalarField(mesh)

velocity = [0.5, 0.0, 0.0]
k_inlet = 0.2
ω_inlet = 200
ω_wall = 1e3 # ω_inlet
nu = 1e-3
Re = velocity[1]*0.1/nu

U = assign(
    U,
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
    )

p = assign(
    p,
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

k = assign(
    k,
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
)

ω = assign(
    ω,
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, ω_wall),
    Dirichlet(:top, ω_wall)
)

setup_U = SolverSetup(
    solver      = GmresSolver, # BicgstabSolver, GmresSolver
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, # GmresSolver, FomSolver, DiomSolver
    relax       = 0.3,
    itmax       = 100,
    rtol        = 1e-1
)

GC.gc()

initialise!(U, velocity)
initialise!(p, 0.0)
initialise!(k, k_inlet)
initialise!(ω, ω_inlet)
initialise!(νt, k_inlet/ω_inlet)

iterations = 1
Rx, Ry, Rp = isimple!( # 123 its, 4.68k allocs
    mesh, nu, U, p, k, ω, νt, 
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)

write_vtk(
    "results", mesh, 
    ("U", U), 
    ("p", p),
    ("k", k),
    ("omega", ω),
    ("nut", νt)
    )

plot(; xlims=(0,123))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

# using Profile, PProf

# GC.gc()
# initialise!(U, velocity)
# initialise!(p, 0.0)

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=1 begin Rx, Ry, Rp = isimple!(
#     mesh, nu, U, p,
#     # setup_U, setup_p, iterations, pref=0.0)
#     setup_U, setup_p, iterations)
# end

# PProf.Allocs.pprof()