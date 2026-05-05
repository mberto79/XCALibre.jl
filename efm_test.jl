using XCALibre
# using CSV
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

function env_default!(key, default)
    value = get(ENV, key, string(default))
    ENV[key] = value
    return value
end

function env_value(key, default)
    return get(ENV, key, string(default))
end

EFM_CASE = parse(Int, env_value("EFM_CASE", 1))

# Meredith et al. CD1 cases as reported in ASME GTP 143(4), Table 2.
# The table heading is "Q x 10^5 (m^3/s)", so the physical flow rate is
# table_Q * 1e-5 m^3/s. This scaling reproduces the reported Reynolds numbers.
if EFM_CASE == 1
    inlet_flow_rate = 1.69e-5 # m^3/s
    ϕ  = 5
    θm = 75
    σ  = 0.069
    β  = 1
elseif EFM_CASE == 5
    inlet_flow_rate = 3.76e-5 # m^3/s
    ϕ  = 89
    θm = 75
    σ  = 0.069
    β  = 3
elseif EFM_CASE == 8
    inlet_flow_rate = 25.53e-5 # m^3/s
    ϕ = 89
    θm = 75
    σ  = 0.069
    # GTP Fig. 4 validates case 8 with β = 1; Table 3 later gives β = 3 as the optimized value.
    β = 1
else
    error("Unsupported EFM_CASE=$(EFM_CASE). Supported cases are 1, 5, and 8.")
end

β = parse(Float64, env_value("EFM_BETA", β))

# EFM_DT sets the requested initial time step.
EFM_DT = parse(Float64, env_value("EFM_DT", "1e-4"))
# EFM_ITERS sets the number of solver iterations.
EFM_ITERS = parse(Int, env_value("EFM_ITERS", "50000"))
# EFM_WRITE_INTERVAL sets output frequency; use -1 to disable writes.
EFM_WRITE_INTERVAL = parse(Int, env_value("EFM_WRITE_INTERVAL", "100"))
# EFM_SIGMA_SCALE multiplies surface tension; use 0 to disable capillarity.
EFM_SIGMA_SCALE = parse(Float64, env_value("EFM_SIGMA_SCALE", "1"))
# EFM_ADAPTIVE=1 enables adaptive time stepping from the limiting Courant number.
EFM_ADAPTIVE = env_value("EFM_ADAPTIVE", "1") != "0"
# EFM_MAX_CO is the adaptive target for max ordinary/film Courant number.
EFM_MAX_CO = parse(Float64, env_value("EFM_MAX_CO", "0.1"))
# EFM_MIN_SHRINK is the smallest adaptive dt multiplier per iteration.
EFM_MIN_SHRINK = parse(Float64, env_value("EFM_MIN_SHRINK", "0.1"))
# EFM_MAX_GROW is the largest adaptive dt multiplier per iteration.
EFM_MAX_GROW = parse(Float64, env_value("EFM_MAX_GROW", "1.2"))
# XCALIBRE_EFM_DEBUG=1 prints film diagnostics every iteration.
XCALIBRE_EFM_DEBUG = env_default!("XCALIBRE_EFM_DEBUG", "0") != "0"
# XCALIBRE_EFM_DEBUG_INTERVAL prints diagnostics every N iterations when N > 0.
XCALIBRE_EFM_DEBUG_INTERVAL = parse(Int, env_default!("XCALIBRE_EFM_DEBUG_INTERVAL", "0"))
# XCALIBRE_EFM_WETTING selects hard, smooth, or allwet wetting-mask probes.
XCALIBRE_EFM_WETTING = env_default!("XCALIBRE_EFM_WETTING", "smooth")
# XCALIBRE_EFM_WETTING_WIDTH sets the h_crit-to-width*h_crit smoothing interval.
XCALIBRE_EFM_WETTING_WIDTH = parse(Float64, env_default!("XCALIBRE_EFM_WETTING_WIDTH", "10"))
mu = 0.001003; # Pa s
rho_l = 998.2; # kg/m3
nu = mu/rho_l;

σ *= EFM_SIGMA_SCALE

g = 9.81
inlet_flow_per_width = inlet_flow_rate/inlet_width # m^2/s
gravity_tangential = g*sind(ϕ)
inlet_height = (3*nu*inlet_flow_per_width/gravity_tangential)^(1/3) # m, Nusselt estimate
inlet_rate = inlet_flow_per_width/inlet_height # m/s
# inlet_velocity = inlet_rate.*[1.0, 0.0, 0.0];
inlet_velocity = inlet_rate.*[1.0, 0.0, 0.0];
case_reynolds = inlet_rate*inlet_height/nu

# - Case 1: h = 4.8886e-4 m, U = 0.06778 m/s
#       - Case 5: h = 2.8296e-4 m, U = 0.26056 m/s
#       - Case 8: h = 5.3580e-4 m, U = 0.93428 m/s

h_crit = 1e-10
h_floor = 1e-15

Δt = EFM_DT # 5e-5
end;

@info "EFM paper case setup" case=EFM_CASE inlet_flow_rate inlet_width inlet_flow_per_width inlet_height inlet_rate case_reynolds ϕ θm σ β

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
            # Extrapolated(:inlet),
            Extrapolated(:outlet),
            Wall(:inlet_sides, [0,0,0]),
            Extrapolated(:top_of_plate),
            Extrapolated(:side_1),
            Extrapolated(:side_2),
        ],
        h = [
            Dirichlet(:inlet, inlet_height),
            # Zerogradient(:inlet),
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
        divergence=Upwind
        #divergence=LUST
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
        # relax       = 0.9,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-8
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-11,
        # relax       = 0.7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-8
    )
);

adaptive = if EFM_ADAPTIVE
    AdaptiveTimeStepping(
        maxCo=EFM_MAX_CO,
        minShrink=EFM_MIN_SHRINK,
        maxGrow=EFM_MAX_GROW
    )
else
    nothing
end
begin

runtime = Runtime(
    iterations=EFM_ITERS,
    time_step=Δt,
    write_interval=EFM_WRITE_INTERVAL,
    adaptive=adaptive
)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs);

GC.gc(true)
  
initialise!(model.momentum.U, [0,0,0]);
# Start from a dry plate; the paper cases inject the liquid film from the inlet.
h_init = h_floor;
initialise!(model.momentum.h, h_init)

residuals = run!(model, config, inner_loops=2);

end;
