using Plots
using XCALibre
# using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_5mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_dev = adapt(CUDABackend(), mesh)

velocity = [600, 0.0, 0.0]
nu = 1e-1
Re = velocity[1]*0.1/nu
Cp = 1005
gamma = 1.4
h_inf = 300*Cp
h_wall = 320*Cp
pressure = 150000


model = Physics(
    time = Steady(),
    fluid = Compressible(
        mu = nu,
        cp = ConstantScalar(cp),
        gamma = ConstantScalar(gamma),
        Pr = ConstantScalar(Pr)
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(),
    domain = mesh
    )


@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0])
)

@assign! model momentum p (
    Dirichlet(:inlet, pressure),
    Neumann(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, model=model.energy),
    Neumann(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),#, gradient=Midpoint),
    p = set_schemes(divergence=Linear, gradient=Midpoint),
    h = set_schemes(divergence=Upwind)#, gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-6,
        atol = 1e-2
    ),
    p = set_solver(
        model.momentum.p;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-6,
        atol = 1e-3
    ),
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-4
    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=100)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

Rx, Ry, Rz, Rp, Re, model_out = run!(model, config) # 9.39k allocs in 184 iterations

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="p")

# # PROFILING CODE

using Profile, PProf

GC.gc()
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rz, Rp, model_out = run!(model, config)
end

PProf.Allocs.pprof()
