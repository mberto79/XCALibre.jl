using XCALibre
using CUDA   # uncomment for NVIDIA GPU
# using AMDGPU # uncomment for AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Freestream conditions at M=1.5
gamma = 1.4
cp    = 1005.0       # J/(kg·K)
Pr    = 0.7
nu    = 1e-5      # kinematic viscosity (inviscid here, but required by fluid model)
T_inf = 300.0        # K
p_inf = 101325.0     # Pa
R_gas = cp * (1.0 - 1.0/gamma)   # ≈ 287 J/(kg·K)
a_inf = sqrt(gamma * R_gas * T_inf)  # ≈ 347 m/s
Mach  = 1.2
U_inf = Mach * a_inf              # ≈ 521 m/s

velocity = [U_inf, 0.0, 0.0]
noflow   = [0.0,   0.0, 0.0]

model = Physics(
    time      = Transient(),
    fluid     = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    # turbulence = RANS{Laminar}(),
    turbulence = LES{Smagorinsky}(),
    energy    = Energy{SensibleEnthalpy}(Tref=0.0),
    domain    = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet,    velocity),
            Zerogradient(:outlet),
            # Wall(:cylinder, noflow),
            Dirichlet(:cylinder, noflow),
            Slip(:top),
            Slip(:bottom)
        ],
        p = [
            Dirichlet(:inlet, p_inf),
            Zerogradient(:outlet),
            Zerogradient(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        T = [
            Dirichlet(:inlet, T_inf),
            Zerogradient(:outlet),
            # Dirichlet(:cylinder, 400.0),   # isothermal wall at 400 K
            Zerogradient(:cylinder),          # adiabatic wall
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            # Dirichlet(:cylinder, 0.0),
            Zerogradient(:cylinder),
            # NutMixingLengthWallFunction(:cylinder),
            Symmetry(:top),
            Symmetry(:bottom)
        ]
    )
)

# Density-based solver only needs convergence criterion (no linear solver config)
solvers = (
    rho = (convergence = 1e-15,),
)

# gradient schemes for viscous flux gradients; flux selects the inviscid Riemann solver
schemes = (
    U             = Schemes(gradient=Gauss),
    p             = Schemes(gradient=Gauss),
    T             = Schemes(gradient=Gauss),
    flux          = HLLC(),           # or Rusanov() for more dissipation
    time_stepping = FEuler(),         # or RK2() for 2nd-order in time
    reconstruction = MUSCL{VanLeer}() # Upwind(), MUSCL{VanLeer}(), MUSCL{MinMod}(), MUSCL{Superbee}()
)

runtime = Runtime(
    iterations     = 10000,
    write_interval = 100,
    time_step      = 1e-7,
    adaptive       = AdaptiveTimeStepping(maxCo=0.5)
)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime,
    hardware=hardware, boundaries=BCs
)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, p_inf)
initialise!(model.energy.T,   T_inf)

residuals = run!(model, config)
