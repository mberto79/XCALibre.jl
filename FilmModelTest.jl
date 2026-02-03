using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs

grids_dir = pkgdir(XCALibre, "Test_Meshes/");
#grid = "quad.unv";
grid = "25x25_grid.unv"
mesh_file = joinpath(grids_dir, grid);

mesh = UNV2D_mesh(mesh_file, scale=0.001);

# Select backend and setup hardware
backend = CPU();
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=1024);
# hardware = Hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh; # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 


inlet_size = 0.01; # m - taken from model in Salome
rho_l = 1;#991.07; # Density of water @ 43°C kg/m3
Γ=200; # g/m/s
Γkg = Γ/1000; # kg/m/s
inlet_flow_rate = Γkg/rho_l; # m2/s
h_inlet = 0.005;
inlet_speed = inlet_flow_rate/h_inlet;
inlet_speed = 0.04;

velocity = inlet_speed*[1, 0.0, 0.0];
nu = 6.245e-7; # Kinematic Viscosity of water @ 43°C
Re = velocity[1]*0.01/nu;


h_crit = 1e-10;



model = Physics(
    momentum=Momentum{EFM}(; h_crit = h_crit, β=6.0, θm = 75, ϕ=90),
    time = Steady(),
    fluid = Fluid{Incompressible}(; nu = nu, rho = rho_l),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
);

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            #Extrapolated(:outlet),
            Zerogradient(:outlet),
            #Extrapolated(:outlet),
            #Wall(:wall, [0.0, 0.0, 0.0]),
            #Zerogradient(:bottom),
            #Zerogradient(:wall),
            Wall(:bottom, [0.0, 0.0, 0.0]),
            #Extrapolated(:bottom),
            Wall(:top, [0.0, 0.0, 0.0])
            #Zerogradient(:top)
            #Extrapolated(:top)
        ],
        h = [
            Dirichlet(:inlet, h_inlet),
            #Zerogradient(:inlet),
            #Extrapolated(:inlet),
            #Dirichlet(:outlet, h_crit),
            #Wall(:outlet),
            Zerogradient(:outlet),
            #Extrapolated(:outlet),
            #Wall(:wall),
            Zerogradient(:bottom),
            #Dirichlet(:bottom, h_crit),
            #Zerogradient(:wall),
            #Extrapolated(:bottom),
            #Wall(:top)
            Zerogradient(:top)
            #Dirichlet(:top, h_crit)
            #Extrapolated(:top)
        ],
        PL = [
            Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:bottom),
            #Zerogradient(:wall),
            Zerogradient(:top)
        ]
    )
);

schemes = (
    U = Schemes(divergence = Linear),
    h = Schemes(), # no input provided (will use defaults)
    PL = Schemes()
);

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-12,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    PL = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
);

runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
#runtime = Runtime(iterations=20, time_step=1, write_interval=1); # hide

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

initialise!(model.momentum.U, velocity);
initialise!(model.momentum.h, 1);

residuals = run!(model, config);