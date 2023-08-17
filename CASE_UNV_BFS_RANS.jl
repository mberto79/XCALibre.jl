using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh_file = "unv_sample_meshes/backwardFacingStep_5mm.unv"
# mesh_file = "unv_sample_meshes/backwardFacingStep_2mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

U = VectorField(mesh)
p = ScalarField(mesh)
k = ScalarField(mesh)
ω = ScalarField(mesh)
νt = ScalarField(mesh)

# velocity = [0.5, 0.0, 0.0]
nu = 1e-3
# u_mag = 1.5
u_mag = 2.5
velocity = [u_mag, 0.0, 0.0]
Tu = 0.1
k_inlet = 1 # 3/2*(Tu*u_mag)^2
νR = 0.1 # nut/nu
ω_inlet = 1000 # k_inlet/(nu*νR) # nut = k/ω thus w = k/nut 
# ω_wall = 10*6*1e-3/(0.075*mesh.faces[61].delta^2)
ω_wall = ω_inlet
Re = velocity[1]*0.1/nu

@assign! U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

@assign! p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    # KWallFunction(:wall, (κ=0.41, cmu=0.09, k=k)),
    # KWallFunction(:top, (κ=0.41, cmu=0.09, k=k))
    Dirichlet(:wall, 1e-15),
    Dirichlet(:top, 1e-15)
)

@assign! ω (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall, (κ=0.41, cmu=0.09, k=k)),
    OmegaWallFunction(:top, (κ=0.41, cmu=0.09, k=k))
    # Dirichlet(:wall, ω_wall), 
    # Dirichlet(:top, ω_wall)
)

@assign! νt (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    # OmegaWallFunction(:wall, (κ=0.41, cmu=0.09, k=k)),
    # OmegaWallFunction(:top, (κ=0.41, cmu=0.09, k=k))
    Dirichlet(:wall, 0.0), 
    Dirichlet(:top, 0.0)
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

setup_turb = SolverSetup(
    solver      = GmresSolver, # BicgstabSolver, GmresSolver
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1,
)


initialise!(U, velocity)
initialise!(p, 0.0)
initialise!(k, k_inlet)
initialise!(ω, ω_inlet)
initialise!(νt, k_inlet/ω_inlet)

iterations = 1500
Rx, Ry, Rp = isimple!( 
    mesh, nu, U, p, k, ω, νt, 
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, setup_turb, iterations)

write_vtk(
    "results", mesh, 
    ("U", U), 
    ("p", p),
    ("k", k),
    ("omega", ω),
    ("nut", νt)
    )

# Fp = pressure_forces(:wall, p, 1.25)
# Reff = stress_tensor(U, nu, νt)
# Fv = viscous_forces(:wall, U, 1.25, nu, νt)


plot(; xlims=(0,1500))
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