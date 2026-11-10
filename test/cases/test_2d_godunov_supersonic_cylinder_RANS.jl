using XCALibre

# RANS turbulence models in the density-based (Godunov) solver: k-omega SST with wall
# functions on a Mach 1.2 cylinder, a few iterations. Checks that the model initialises
# (it needs an equation template) and that the fields stay finite with positive nut.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "cylinder_d10mm_5mm.unv"), scale=0.001)
@test typeof(mesh) <: Mesh2

backend  = CPU()
hardware = Hardware(backend=backend, workgroup=AutoTune())
mesh_dev = adapt(backend, mesh)

gamma = 1.4
cp    = 1005.0
Pr    = 0.7
nu    = 1e-5
T_inf = 300.0
p_inf = 101325.0
R_gas = cp * (1.0 - 1.0/gamma)
U_inf = 1.2 * sqrt(gamma * R_gas * T_inf)
velocity = [U_inf, 0.0, 0.0]
noflow   = [0.0, 0.0, 0.0]
k_inf    = 1.5 * (0.01 * U_inf)^2
ω_inf    = 1000.0

model = Physics(
    time       = Transient(),
    fluid      = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{KOmegaSST}(walls=(:cylinder,)),
    energy     = Energy{SensibleEnthalpy}(Tref=0.0),
    domain     = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
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
            Zerogradient(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        k = [
            Dirichlet(:inlet, k_inf),
            Zerogradient(:outlet),
            Dirichlet(:cylinder, 0.0),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        omega = [
            Dirichlet(:inlet, ω_inf),
            Zerogradient(:outlet),
            OmegaWallFunction(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            NutWallFunction(:cylinder),
            Zerogradient(:top),
            Zerogradient(:bottom)
        ]
    )
)

config = Configuration(
    solvers   = (
        rho   = (convergence = 1e-15,),
        k     = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-2),
        omega = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-2),
    ),
    schemes   = (
        U = Schemes(gradient=Gauss),
        p = Schemes(gradient=Gauss),
        T = Schemes(gradient=Gauss),
        k = Schemes(),
        omega = Schemes(),
        flux = HLLC(),
        time_stepping = FEuler(),
    ),
    runtime   = Runtime(iterations=20, write_interval=-1, time_step=1e-7),
    hardware  = hardware,
    boundaries = BCs
)

GC.gc(true)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, p_inf)
initialise!(model.energy.T,   T_inf)
initialise!(model.turbulence.k, k_inf)
initialise!(model.turbulence.omega, ω_inf)
initialise!(model.turbulence.nut, k_inf/ω_inf)

residuals = run!(model, config; progress=false)

@test residuals isa NamedTuple
@test all(isfinite, residuals.rho)
@test all(isfinite, model.momentum.U.x.values)
@test all(isfinite, model.momentum.p.values)
@test all(isfinite, model.energy.T.values)
@test all(isfinite, model.turbulence.k.values)
@test all(isfinite, model.turbulence.omega.values)
@test all(isfinite, model.turbulence.nut.values)
@test all(>(0), model.turbulence.nut.values)
