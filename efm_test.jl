using XCALibre
using CSV
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs

grids_dir = pkgdir(XCALibre, "Test_Meshes/");
grid = "initial_efm_mesh_CD1_2.unv";
mesh_file = joinpath(grids_dir, grid);

mesh = UNV2D_mesh(mesh_file, scale=0.001);

# Select backend and setup hardware
backend = CPU();
# backend = CUDABackend() # run on NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=1024);
# hardware = Hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh; # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

begin # Model Info
inlet_width = 0.510; # m
inlet_height = 0.05; # m
inlet_area = inlet_width*inlet_height; # m^2

# Case 1
#inlet_flow_rate = 1.69e-5
#ϕ  = 5
#θm = 75
#σ  = 0.069
#β  = 1

# Case 5
#inlet_flow_rate = 3.76e-5
#ϕ  = 90
#θm = 75
#σ  = 0.069
#β  = 3

# Case 8
inlet_flow_rate = 25.53e-5
ϕ = 90
θm = 75
σ = 0.069
β = 3

inlet_rate = inlet_flow_rate/inlet_area; # m\s

inlet_velocity = inlet_rate.*[1.0, 0.0, 0.0]#.*1000;
mu = 0.001003;
rho_l = 998.2; # Density of water @ 43°C kg/m3
#nu = 6.245e-7; # kinematic viscosity of water @ 43°c
nu = mu/rho_l;

h_crit = 1e-10;
h_floor = 1e-15

Δt = 1e-4
end;

model = Physics(
    momentum=Momentum{EFM}(;σ=σ, h_crit = h_crit, h_floor=h_floor, β=β, θm = θm, ϕ=ϕ),
    time = Transient(),
    fluid = Fluid{Incompressible}(; nu = nu, rho = rho_l),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
);

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, inlet_velocity),
            Extrapolated(:outlet),
            Wall(:inlet_sides, [0,0,0]),
            Extrapolated(:top_of_plate),
            Extrapolated(:side_1),
            Extrapolated(:side_2),
        ],
        h = [
            Dirichlet(:inlet, inlet_height),
            Extrapolated(:outlet),
            Extrapolated(:inlet_sides),
            Extrapolated(:top_of_plate),
            Extrapolated(:side_1),
            Extrapolated(:side_2)
        ]
    )
);

schemes = (
    U = Schemes(
        time=Euler,
        #divergence=Upwind
        divergence=LUST
        ),
    h = Schemes(
        time=Euler
    ),
);

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-11,
        relax       = 0.9,
        rtol = 1e-4,
        atol = 1e-5
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-11,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-6
    )
);

adaptive = AdaptiveTimeStepping(
    maxCo=0.01,
    minShrink=0.1,
    maxGrow=1.2
)
begin
runtime = Runtime(iterations=200, time_step=Δt, write_interval=5, adaptive=adaptive);
#runtime = Runtime(iterations=2000, time_step=Δt, write_interval=100, adaptive=adaptive)
#runtime = Runtime(iterations=8000, time_step=1e-6, write_interval=100, adaptive=adaptive)
#runtime = Runtime(iterations=100, time_step=Δt, write_interval=2, adaptive=adaptive)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

GC.gc(true)
  
initialise!(model.momentum.U, [0,0,0]);
h_init = h_floor;
initialise!(model.momentum.h, h_init)

residuals = run!(model, config, inner_loops=5);

end;
