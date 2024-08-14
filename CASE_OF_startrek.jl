using Plots
using FVM_1D
using CUDA


mesh_file = "unv_sample_meshes/OF_startrek/polyMesh/"
@time mesh = FOAM3D_mesh(mesh_file, scale=0.001, integer_type=Int64, float_type=Float64)

# mesh_dev = adapt(CPU(), mesh)
mesh_dev = adapt(CUDABackend(), mesh)

Umag = 0.5
nu = 1e-05
d = 100e-3
Re = Umag*d/nu
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
νR = 150
Tu = 0.1
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
nut_inlet = k_inlet/ω_inlet

model = Physics(
    # time = Steady(),
    time = Transient(),
    fluid = FLUID{Incompressible}(nu = nu),
    # turbulence = RANS{KOmega}(),
    turbulence = RANS{Laminar}(),
    # turbulence = LES{Smagorinsky}(),
    energy = ENERGY{Isothermal}(),
    # domain = mesh
    domain = mesh_dev
    )


@assign! model momentum U (
    Dirichlet(:inlet, velocity), 
    Neumann(:outlet, 0.0), 
    Wall(:enterprise, noSlip), 
    # Neumann(:sides, 0.0)
    Dirichlet(:sides, velocity)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0), 
    Dirichlet(:outlet, 0.0), 
    Neumann(:enterprise, 0.0), 
    Neumann.(:sides, 0.0)
)

# @assign! model turbulence k (
#     Dirichlet(:Xmin, k_inlet), # inlet
#     Neumann(:Xmax, 0.0), # outlet
#     # KWallFunction.(walls)..., # walls
#     Dirichlet.(walls, Ref(0.0))..., # walls
#     Neumann.(freestream, Ref(0.0))...,
#     Neumann(:Symmetry, 0.0)
#     # KWallFunction(:Symmetry)
# )

# @assign! model turbulence omega (
#     Dirichlet(:Xmin, ω_inlet), # inlet
#     Neumann(:Xmax, 0.0), # outlet
#     OmegaWallFunction.(walls)..., # walls
#     Neumann.(freestream, Ref(0.0))...,
#     Neumann(:Symmetry, 0.0)
#     # OmegaWallFunction(:Symmetry)
# )

# ~@assign! model turbulence nut (
#     Dirichlet(:Xmin, k_inlet/ω_inlet), # inlet
#     Neumann(:Xmax, 0.0), # outlet
#     # NutWallFunction.(walls)..., # walls
#     Dirichlet.(walls, Ref(0.0))..., # walls
#     Neumann.(freestream, Ref(0.0))...,
#     Neumann(:Symmetry, 0.0)
#     # NutWallFunction(:Symmetry)
# )

schemes = (
    U = set_schemes(time=Euler, divergence=Linear, gradient=Midpoint),
    p = set_schemes(time=Euler, gradient=Midpoint),
    # k = set_schemes(divergence=Upwind, gradient=Midpoint),
    # omega = set_schemes(divergence=Upwind, gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        # relax       = 0.8,
        relax       = 1.0,
        rtol = 1e-5,
        atol = 1e-15
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #LDL(),
        convergence = 1e-7,
        # relax       = 0.2,
        relax       = 0.2,
        # rtol = 1e-3,
        rtol = 1e-5,
        atol = 1e-15
    ),
    # k = set_solver(
    #     # model.turbulence.fields.k;
    #     model.turbulence.k;
    #     solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
    #     preconditioner = Jacobi(),
    #     convergence = 1e-7,
    #     relax       = 0.5,
    #     rtol = 1e-1,
    #     atol = 1e-15
    # ),
    # omega = set_solver(
    #     # model.turbulence.fields.omega;
    #     model.turbulence.omega;
    #     solver      = BicgstabSolver, # BicgstabSolver, GmresSolver, CgSolver
    #     preconditioner = Jacobi(),
    #     convergence = 1e-7,
    #     relax       = 0.5,
    #     rtol = 1e-1,
    #     atol = 1e-15
    # )
)

# runtime = set_runtime(iterations=1000, write_interval=500, time_step=1)
runtime = set_runtime(iterations=5000, write_interval=50, time_step=1e-6)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=8)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut, 2*nu)
# initialise!(model.turbulence.k, k_inlet)
# initialise!(model.turbulence.omega, ω_inlet)

# initialise!(model.momentum.U, [0,0,0])
# initialise!(model.momentum.p, 0.0)
# initialise!(model.turbulence.k, k_inlet)
# initialise!(model.turbulence.omega, ω_inlet)
# initialise!(model.turbulence.nut, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config); #, pref=0.0)

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.momentum.p, 1.25)
Fv = viscous_force(:cylinder, model.momentum.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")