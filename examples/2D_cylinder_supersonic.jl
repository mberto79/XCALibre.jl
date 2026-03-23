using XCALibre
# using CUDA   # uncomment for NVIDIA GPU
# using AMDGPU # uncomment for AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Freestream conditions at M=1.5
gamma = 1.4
cp    = 1005.0       # J/(kg·K)
Pr    = 0.7
nu    = 1e-5      # kinematic viscosity (inviscid here, but required by fluid model)
Tref  = 0.0          # enthalpy reference temperature

T_inf = 300.0        # K
p_inf = 101325.0     # Pa
R_gas = cp * (1.0 - 1.0/gamma)   # ≈ 287 J/(kg·K)
a_inf = sqrt(gamma * R_gas * T_inf)  # ≈ 347 m/s
Mach  = 3
U_inf = Mach * a_inf              # ≈ 521 m/s

velocity = [U_inf, 0.0, 0.0]
noflow   = [0.0,   0.0, 0.0]

model = Physics(
    time      = Transient(),
    fluid     = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{Laminar}(),
    energy    = Energy{SensibleEnthalpy}(Tref=Tref),
    domain    = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet,    velocity),
            Zerogradient(:outlet),
            Wall(:cylinder, noflow),
            Symmetry(:top),
            Symmetry(:bottom)
        ],
        p = [
            Dirichlet(:inlet, p_inf),
            Zerogradient(:outlet),
            Zerogradient(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        he = [
            FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
            Zerogradient(:outlet),
            FixedTemperature(:cylinder, T=400, Enthalpy(cp=cp, Tref=Tref)),
            # Zerogradient(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ]
    )
)

# Density-based solver only needs convergence criterion (no linear solver config)
solvers = (
    rho = (convergence = 1e-15,),
)

# gradient schemes for viscous flux gradients; flux selects the inviscid Riemann solver
schemes = (
    U    = Schemes(gradient=Gauss),
    p    = Schemes(gradient=Gauss),
    he   = Schemes(gradient=Gauss),
    flux = HLLC(),   # or Rusanov() for more dissipation
)

runtime = Runtime(
    iterations     = 50000,
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
