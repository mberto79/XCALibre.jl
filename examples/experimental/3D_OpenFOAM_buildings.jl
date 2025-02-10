using Plots
using XCALibre
using CUDA


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

grid = "OF_buildings/polyMesh"
mesh_file = joinpath(grids_dir, grid)

@time mesh = FOAM3D_mesh(mesh_file, integer_type=Int64, float_type=Float64)

mesh_dev = mesh
mesh_dev = adapt(CUDABackend(), mesh)

Umag = 10
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1.5e-5
νR = 50
Tu = 0.1
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
nut_inlet = k_inlet/ω_inlet
Re = (10*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )


@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Symmetry(:frontAndBack, 0.0),
    Wall(:ground, noSlip),
    Wall(:buildings, noSlip)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:frontAndBack, 0.0),
    Neumann(:ground, 0.0),
    Neumann(:buildings, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:frontAndBack, 0.0),
    KWallFunction(:ground),
    KWallFunction(:buildings)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:frontAndBack, 0.0),
    OmegaWallFunction(:ground),
    OmegaWallFunction(:buildings)
)

@assign! model turbulence nut (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Neumann(:frontAndBack, 0.0),
    NutWallFunction(:ground),
    NutWallFunction(:buildings)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(divergence=Upwind, gradient=Midpoint),
    omega = set_schemes(divergence=Upwind, gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.4,
        rtol = 1e-2,
        atol = 1e-15
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = IC0GPU(), #Jacobi(), 
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2,
        atol = 1e-15
    ),
    k = set_solver(
        # model.turbulence.fields.k;
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-1,
        atol = 1e-15
    ),
    omega = set_solver(
        # model.turbulence.fields.omega;
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver, CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-1,
        atol = 1e-15
    )
)

runtime = set_runtime(iterations=500, write_interval=50, time_step=1)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config, ncorrectors=0)


Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.momentum.p, 1.25)
Fv = viscous_force(:cylinder, model.momentum.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")