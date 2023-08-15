using Plots

using FVM_1D

using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# Fields 
U = VectorField(mesh)
p = ScalarField(mesh)
k = ScalarField(mesh)
ω = ScalarField(mesh)
νt = ScalarField(mesh)

# BOUNDARY CONDITIONS 

Umag = 1.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
νR = 50
Tu = 0.05
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (0.2*velocity[1])/nu

U = assign(
    U, 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    # Dirichlet(:top, velocity),
    # Dirichlet(:bottom, velocity),
    Dirichlet(:cylinder, noSlip)
)

p = assign(
    p,
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:cylinder, 0.0)
)

k = assign(
    k,
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Dirichlet(:cylinder, 1e-15)
)

ω = assign(
    ω,
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    OmegaWallFunction(:cylinder, (κ=0.41, cmu=0.09, k=k))
)

νt = assign(
    νt,
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0), 
    Dirichlet(:cylinder, 0.0)
)

setup_U = SolverSetup(
    solver      = GmresSolver, # GmresSolver, BicgstabSolver
    relax       = 0.9,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-1
)

setup_turb = SolverSetup(
    solver      = GmresSolver, # BicgstabSolver, GmresSolver
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1,
)

GC.gc()

initialise!(U, velocity)
initialise!(p, 0.0)
initialise!(k, k_inlet)
initialise!(ω, ω_inlet)
initialise!(νt, k_inlet/ω_inlet)

iterations = 500
Rx, Ry, Rp = isimple!( 
    mesh, nu, U, p, k, ω, νt, 
    setup_U, setup_p, setup_turb, iterations)

Fp = pressure_forces(:cylinder, p, 1.25)
Reff = stress_tensor(U, nu, νt)
Fv = viscous_forces(:cylinder, U, 1.25, nu, νt)

write_vtk(
    "results", mesh, 
    ("U", U), 
    ("p", p),
    ("k", k),
    ("omega", ω),
    ("nut", νt)
    )

plot(; xlims=(0,iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")