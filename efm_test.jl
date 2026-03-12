using XCALibre
using CSV
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs

grids_dir = pkgdir(XCALibre, "Test_Meshes/");
grid = "initial_efm_mesh4.unv";
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

test_case = 5;
input_parameters = CSV.File("Model_Input.csv"); #File containing the different test cases from paper "Modeling of Partially Wetting Liquid Film Using an Enhanced Thin Film Model for Aero-Engine Bearing Chamber Applications" by Kuldeep Singh et. al
inlet_width = 0.510; # m
inlet_height = 0.05; # m
inlet_area = inlet_width*inlet_height; # m^2
inlet_flow_rate = input_parameters.Q[test_case]; # m^3/s
inlet_rate = inlet_flow_rate/inlet_area; # m\s

inlet_velocity = inlet_rate.*[1.0, 0.0, 0.0]#.*1000;
mu = 0.001003;
rho_l = 998.2; # Density of water @ 43°C kg/m3
#nu = 6.245e-7; # kinematic viscosity of water @ 43°c
nu = mu/rho_l;

h_crit = 1e-10;
β = input_parameters.Beta[test_case]; # empisical value from paper
σ = input_parameters.Sigma[test_case]
θm = input_parameters.Theta_m[test_case]
ϕ = input_parameters.Phi[test_case]
Δt = 1e-3
Δx = 0.006
C=inlet_rate*Δt/Δx

model = Physics(
    momentum=Momentum{EFM}(;σ=σ, h_crit = h_crit, β=β, θm = θm, ϕ=ϕ),
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
            Extrapolated(:side_2)
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
        #time=SteadyState,
        #divergence=Linear
        #divergence=Upwind
        divergence=LUST
        ),
    h = Schemes(
        time=Euler,
        #time=SteadyState,
        #divergence=Linear
        #divergence=Upwind
        #divergence=LUST
    ),
);

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-11,
        relax       = 1.0,
        rtol = 0,
        atol = 1e-5
    ),
    h = SolverSetup(
        solver      = Cg(),#Bicgstab(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-11,
        relax       = 1.0,
        rtol = 0,
        atol = 1e-6
    )
);

#runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
#runtime = Runtime(iterations=20000, time_step=1, write_interval=20000)
runtime = Runtime(iterations=20, time_step=Δt, write_interval=1); # hide
#runtime = Runtime(iterations=200, time_step=Δt, write_interval=10);
#runtime = Runtime(iterations=2000, time_step=Δt, write_interval=100)
#runtime = Runtime(iterations=8000, time_step=Δt, write_interval=400)
#runtime = Runtime(iterations=100, time_step=Δt, write_interval=2)
#runtime = Runtime(iterations=100000, time_step=Δt, write_interval=10000)
#runtime = Runtime(iterations=500, time_step=Δt, write_interval=10)
#runtime = Runtime(iterations=4000, time_step=Δt, write_interval=100)
#runtime = Runtime(iterations=15000, time_step=Δt, write_interval=250)
#runtime = Runtime(iterations=500, time_step=Δt, write_interval=50)
#runtime = Runtime(iterations=1, time_step=Δt, write_interval=1);

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

GC.gc(true)
  
initialise!(model.momentum.U, [0,0,0]);
h_init = 1e-13;#h_crit*100;
initialise!(model.momentum.h, h_init)

#for i ∈ eachindex(model.momentum.h.values)
#    if abs(model.momentum.h.mesh.cells[i].centre[2]) < 0.51/2
#        model.momentum.U.x.values[i] = inlet_velocity[1]/1000;
#        model.momentum.U.y.values[i] = inlet_velocity[2]/1000;
#        model.momentum.h.values[i] = inlet_height/100;
#    end
#end



residuals = run!(model, config);

using Plots
plot((residuals.Ux), label="Ux")
plot!((residuals.Uy), label="Uy")
#plot!(residuals.Uz, label="Uz")
plot!((residuals.h), label="h", yaxis=(:log10, [1e-2, 1e-8]))