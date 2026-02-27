using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs

grids_dir = pkgdir(XCALibre, "Test_Meshes/");
#grid = "quad.unv";
grid = "25x25_grid.unv"
#grid = "500x500_grid.unv"
mesh_file = joinpath(grids_dir, grid);

mesh = UNV2D_mesh(mesh_file, scale=0.01);

# Select backend and setup hardware
backend = CPU();
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=1024);
# hardware = Hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh; # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 


inlet_size = 0.1; # m - taken from model in Salome
rho_l = 991.07; # Density of water @ 43°C kg/m3
Γ=200; # g/m/s
Γkg = Γ/1000; # kg/m/s
inlet_flow_rate = Γkg/rho_l; # m2/s
h_inlet = 0.05;
inlet_speed = inlet_flow_rate/h_inlet;
#inlet_speed = 0.04;

velocity = inlet_speed*[1, 0.0, 0.0];
nu = 6.245e-7; # Kinematic Viscosity of water @ 43°C
Re = velocity[1]*0.01/nu;


h_crit = 1e-10;
#h_crit = 5e-3


model = Physics(
    momentum=Momentum{EFM}(;σ=0.069, h_crit = h_crit, β=1.0, θm = 30, ϕ=10),
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
            Dirichlet(:inlet, velocity),
            #Extrapolated(:outlet),
            Zerogradient(:outlet),
            #Extrapolated(:outlet),
            #Wall(:wall, [0.0, 0.0, 0.0]),
            #Zerogradient(:bottom),
            #Zerogradient(:wall),
            Wall(:bottom, [0.0, 0.0, 0.0]),
            #Wall(:bottom, velocity),
            #Extrapolated(:bottom),
            Wall(:top, [0.0, 0.0, 0.0])
            #Wall(:top, velocity)
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
            #Dirichlet(:bottom, 1e-18),
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
        ]
    )
);

schemes = (
    U = Schemes(
        time=Euler,
        divergence=Upwind
        ),
    h = Schemes(
        time=Euler,
        divergence=Upwind
    ),
);

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-12,
        relax       = 0.8,
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
    )
);

#runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
#runtime = Runtime(iterations=20000, time_step=1, write_interval=20000)
#runtime = Runtime(iterations=20, time_step=1, write_interval=1); # hide
runtime = Runtime(iterations=2000, time_step=0.01, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

initialise!(model.momentum.U, velocity);
#initialise!(model.momentum.U, [1e-10,1e-10,1e-10]);
#initialise!(model.momentum.h, h_inlet)
initialise!(model.momentum.h, h_crit/10)
#initialise!(model.momentum.h, 0.000005046);
#pow = 18
#for i ∈ eachindex(model.momentum.h)
    #model.momentum.h[i] = ((0.1-mesh.cells[i].centre[1])/0.1)^pow
#    if mesh.cells[i].centre[1] > 0.06
#        model.momentum.h[i] = h_crit/10
#    end
#    println(cells[i].centre)
#end

residuals = run!(model, config);