using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs

grids_dir = pkgdir(XCALibre, "Test_Meshes/");

#grid = "25x25_grid.unv"
grid = "500x500_grid.unv"
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


rho_l = 991.07; # Density of water @ 43°C kg/m3

nu = 6.245e-7; # Kinematic Viscosity of water @ 43°C
#Re = velocity[1]*0.01/nu;

#h_crit = 1e-10;
h_crit = 1e-10


model = Physics(
    momentum=Momentum{EFM}(;σ=0.069, h_crit = h_crit, β=6.0, θm = 70, ϕ=0),
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
            Wall(:inlet, [0.0, 0.0, 0.0]),
            Wall(:bottom, [0.0, 0.0, 0.0]),
            Wall(:outlet, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ],
        h = [
            #Zerogradient(:inlet),
            #Zerogradient(:bottom),
            #Zerogradient(:outlet),
            #Zerogradient(:top)
            Wall(:inlet),
            Wall(:bottom),
            Wall(:outlet),
            Wall(:top)
        ]
    )
);

schemes = (
    U = Schemes(
        time=Euler,
        divergence=Upwind
        #divergence=LUST
        ),
    h = Schemes(
        time=Euler,
        divergence=Upwind
        #divergence=LUST
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
        convergence = 1e-12,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-10
    )
);

#runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
#runtime = Runtime(iterations=20000, time_step=1, write_interval=20000)
#runtime = Runtime(iterations=20, time_step=0.01, write_interval=1); # hide
#runtime = Runtime(iterations=2000, time_step=0.01, write_interval=100)
runtime = Runtime(iterations=50, time_step=0.01, write_interval=5)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

initialise!(model.momentum.U, [0,0,0]);
#initialise!(model.momentum.U, [0,0,0]);
h_init = h_crit/100;
initialise!(model.momentum.h, h_init)
cells = mesh.cells;
h_inner = 1e-3
h_outer = 1e-5
for i ∈ eachindex(mesh.cells)
    if sqrt((cells[i].centre[1]-0.05)^2+(cells[i].centre[2]-0.05)^2)<0.1^2
        #model.momentum.U.x.values[i] = 1e-3
        #model.momentum.U.y.values[i] = 1e-3
        
        model.momentum.h[i] = -((h_inner-h_outer)/0.1)*sqrt((cells[i].centre[1]-0.05)^2+(cells[i].centre[2]-0.05)^2)+h_inner
    end
end


residuals = run!(model, config);

using Plots