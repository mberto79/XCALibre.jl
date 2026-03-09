# using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
# mesh_file = "unv_sample_meshes/OF_squareBend_laminar/constant/polyMesh/"

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_squareBend/polyMesh"
mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file)


backend = CPU(); workgroup = 1024
# backend = CUDABackend(); workgroup = 32 
# backend = ROCBackend(); workgroup = 32


mesh_dev = adapt(backend, mesh)

# Inlet conditions

# Not working

velocity = [50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
temp = 1000 # 300.0
pressure = 110000
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid = Fluid{Compressible}(
    # fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=298.15),
    domain = mesh_dev
    )

boundaries = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            # Extrapolated(:outlet),
            # Wall(:walls, noSlip)
            # Symmetry(:walls)
            Slip(:walls)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, pressure),
            # Dirichlet(:inlet, pressure),
            # Zerogradient(:outlet),

            # Wall(:walls)
            Slip(:walls)
        ],
        h = [
            FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=298.15)),
            Zerogradient(:outlet),
            # Extrapolated(:outlet),
            # Wall(:walls)
            Slip(:walls)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.6,
        rtol = 1e-3,
        # atol = 1e-4
    ),
    p = SolverSetup(
        # solver      = Cg(), # Bicgstab(), Gmres(), Cg() # WeaklyCompressible
        solver      = Bicgstab(), # Bicgstab(), Gmres(), Cg() # Compressible
        preconditioner = Jacobi(), # Jacobi DILU
        convergence = 1e-7,
        relax       = 0.2,
        limit = (1000.0, 1000000.0),
        rtol = 1e-3,
        # atol = 1e-4
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        # limit = (100.0, 1000.0),
        rtol = 1e-2,
        # atol = 1e-4
    )
)

divergence = Upwind # Upwind LUST
schemes = (
    U = Schemes(divergence=BoundedUpwind),
    p = Schemes(divergence=Upwind),
    h = Schemes(divergence=BoundedUpwind)

    # U = Schemes(divergence=divergence),
    # p = Schemes(divergence=divergence),
    # h = Schemes(divergence=divergence)
)

runtime = Runtime(iterations=5000, write_interval=50, time_step=1)

hardware = Hardware(backend=backend, workgroup=workgroup)

config = Configuration(;
    solvers, schemes, runtime, hardware, boundaries)

GC.gc(true)

initialise!(model.momentum.U, [0.0, 0.0, 0.0])
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

residuals = run!(model, config, output=VTK()); #, pref=0.0)



# plot(; xlims=(0,runtime.iterations), ylims=(1e-8,1))
# plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
# plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
# plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")
# plot!(1:length(residuals.e), residuals.e, yscale=:log10, label="h")