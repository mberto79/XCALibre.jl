using XCALibre
# using CUDA # or AMDGPU, to run on GPU

# Mach 1.2 cylinder with pressure-based CPISO; `p` needs divergence=Upwind (Linear oscillates at shocks).
# Solver `limit` clamps on p and he bound the startup transient; for strongly supersonic flow see 2D_cylinder_supersonic.jl.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_25mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024
# backend = CUDABackend(); workgroup = 32
mesh_dev = adapt(backend, mesh)

# Freestream conditions at Mach 1.2
gamma = 1.4
cp = 1005.0
Pr = 0.7
nu = 1e-5
R = cp*(1.0 - 1.0/gamma)
cv = cp - R
T_inf = 300.0
p_inf = 101325.0
Tref = 0.0
Mach = 1.4
U_inf = Mach*sqrt(gamma*R*T_inf)
velocity = [U_inf, 0.0, 0.0]

# Freestream turbulence (KOmega): 5% intensity, nut/nu ratio = 10
Tu = 0.05
nuR = 10
k_inlet = 1.5*(Tu*U_inf)^2
ω_inlet = k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet

model = Physics(
    # time = Transient(),
    time = Steady(),
    fluid = Fluid{Compressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{KOmega}(),
    energy = Energy{SensibleEnthalpy}(Tref=Tref),
    # energy = Energy{InternalEnergy}(Tref=Tref),
    domain = mesh_dev
    )

boundaries = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:cylinder, [0.0, 0.0, 0.0]),
            Slip(:top),
            Slip(:bottom)
        ],
        p = [
            Dirichlet(:inlet, p_inf),
            Zerogradient(:outlet),
            Wall(:cylinder),
            Slip(:top),
            Slip(:bottom)
        ],
        he = [
            FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:inlet, T=T_inf, IEnergy(cv=cv, Tref=Tref)),
            Zerogradient(:outlet),
            FixedTemperature(:cylinder, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:cylinder, T=T_inf, IEnergy(cv=cv, Tref=Tref)),
            Slip(:top),
            Slip(:bottom)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            KWallFunction(:cylinder),
            Slip(:top),
            Slip(:bottom)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:cylinder),
            Slip(:top),
            Slip(:bottom)
        ],
        nut = [
            Dirichlet(:inlet, νt_inlet),
            Extrapolated(:outlet),
            NutWallFunction(:cylinder),
            Slip(:top),
            Slip(:bottom)
        ]
    )
)
time = SteadyState # Euler

relax_p = time() isa SteadyState ? 0.4 : 1.00
relax_U = time() isa SteadyState ? 0.6 : 1.00
convergence = 1e-8
solvers = (
    U = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=relax_U, rtol=1e-1
        ),
    p = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=relax_p, 
        limit=(0.02*p_inf, 50*p_inf), rtol=1e-1
        ),
    he = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=relax_p,
        limit=(50.0, 6000.0), rtol=1e-1
        ),
    k = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=0.6, rtol=1e-1
        ),
    omega = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=0.6, rtol=1e-1
        )
)


schemes = (
    U = Schemes(divergence=Upwind, gradient=Gauss, time=time),
    p = Schemes(divergence=Upwind, gradient=Gauss, time=time),  # Upwind is required (see notes)
    he = Schemes(divergence=Upwind, gradient=Gauss, time=time),
    k = Schemes(divergence=Upwind, gradient=Gauss, time=time),
    omega = Schemes(divergence=Upwind, gradient=Gauss, time=time)
)

dt = time() isa SteadyState ? 1 : 5e-8
runtime = Runtime(iterations=10000, write_interval=100, time_step=dt)
hardware = Hardware(backend=backend, workgroup=workgroup)

config = Configuration(;
    solvers, schemes, runtime, hardware, boundaries)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, p_inf)
initialise!(model.energy.T, T_inf)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

residuals = run!(model, config, output=VTK(), inner_loops=2)
