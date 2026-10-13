# Mach 1.4 flow over a cylinder, as in 2D_cylinder_transonic_RANS.jl, but with `p` solved by
# AMG-preconditioned BiCGStab: upwinded pressure convection makes the pressure matrix
# non-symmetric, so `AMG(mode=Cg())` would reject it.
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_25mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024
# backend = CUDABackend(); workgroup = 32
mesh_dev = adapt(backend, mesh)

# Freestream conditions at Mach 1.4
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
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        he = [
            FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:inlet, T=T_inf, IEnergy(cv=cv, Tref=Tref)),
            Zerogradient(:outlet),
            FixedTemperature(:cylinder, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:cylinder, T=T_inf, IEnergy(cv=cv, Tref=Tref)),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            KWallFunction(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        nut = [
            Dirichlet(:inlet, νt_inlet),
            Extrapolated(:outlet),
            NutWallFunction(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ]
    )
)
time = SteadyState # Euler

relax_p = time() isa SteadyState ? 0.3 : 1.00
relax_U = time() isa SteadyState ? 0.7 : 1.00
convergence = 1e-8
solvers = (
    U = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=relax_U, rtol=1e-2
        ),
    p = SolverSetup(
        solver=AMG(mode=Bicgstab(), coarsening=SmoothAggregation(), smoother=AMGJacobi()),
        preconditioner=Jacobi(), convergence=convergence, relax=relax_p,
        limit=(0.02*p_inf, 50*p_inf), rtol=1e-2 # clamps bound the startup transient
        ),
    he = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=relax_p,
        limit=(50.0, 6000.0), rtol=1e-2
        ),
    k = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=0.6, rtol=1e-2
        ),
    omega = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=convergence, relax=0.6, rtol=1e-2
        )
)


schemes = (
    U = Schemes(divergence=Upwind, gradient=Gauss, time=time),
    p = Schemes(divergence=Upwind, gradient=Gauss, time=time),  # Upwind is required at shocks
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
