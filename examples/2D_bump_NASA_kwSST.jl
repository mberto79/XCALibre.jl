using Plots
using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_bump2d/polyMesh"
mesh_file = joinpath(grids_dir, grid)

mesh = FOAM3D_mesh(mesh_file, scale=1, integer_type=Int64, float_type=Float64)

# mesh_dev = adapt(CUDABackend(), mesh)
# mesh_dev = adapt(CPUBackend(), mesh)
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(CPU(), mesh)

L = 50
nu = 1.388E-5
# u_mag = 1.5 # 5mm mesh
u_mag = 69.44 # 2mm mesh
# u_mag = 5 # 2mm mesh
# u_mag = 5 # 2mm mesh
velocity = [u_mag, 0.0, 0.0]
Tu = 0.01
nuR = 10
ReL = u_mag*L/nu
k_inlet = 0.723
ω_inlet = 8675 #k_inlet/(nuR*nu)

νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaSST}(walls=(:bump,)),
    # turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

patch_group = [:top, :symUp, :symDown]

# set up as walls
group_bcs_U = Wall.(patch_group, Ref([0,0,0]))
group_bcs_p = Wall.(patch_group)
group_bcs_k = Dirichlet.(patch_group, Ref(0.0))
# group_bcs_k = KWallFunction.(patch_group)
group_bcs_omega = OmegaWallFunction.(patch_group)
group_bcs_nut = Dirichlet.(patch_group, Ref(0.0))

# set up as symmetric
group_bcs_U = Symmetry.(patch_group)
group_bcs_p = group_bcs_U
group_bcs_k = group_bcs_U
group_bcs_omega = group_bcs_U
group_bcs_nut = group_bcs_U


BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            # Extrapolated(:outlet),
            Zerogradient(:outlet),
            Wall(:bump, [0.0, 0.0, 0.0]),
            Empty(:frontAndBack),
            group_bcs_U...,
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:bump),
            Empty(:frontAndBack),
            group_bcs_p...,
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            # Extrapolated(:outlet),
            # Dirichlet(:bump, 0.0),
            KWallFunction(:bump),
            Empty(:frontAndBack),
            group_bcs_k...,
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            # Extrapolated(:outlet),
            OmegaWallFunction(:bump),
            Empty(:frontAndBack),
            group_bcs_omega...,
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Dirichlet(:bump, 0.0),
            # NutWallFunction(:bump),
            Empty(:frontAndBack),
            group_bcs_nut...,
        ],
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    y = Schemes(),#gradient=Midpoint),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.6,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        # preconditioner = Jacobi(),
        preconditioner = DILU(),
        convergence = 1e-11,
        relax       = 0.2,
        rtol = 1e-3,
        itmax = 4000
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-10,
        rtol = 1e-5,
        relax       = 0.7,
        itmax = 5000
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # DILU Jacobi
        convergence = 1e-10,
        relax       = 0.6,
        rtol = 1e-3
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-10,
        relax       = 0.6,
        rtol = 1e-3
    )
)

runtime = Runtime(iterations=5000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
# initialise!(model.turbulence.k, 0.0)
initialise!(model.turbulence.k, k_inlet) # k_inlet
initialise!(model.turbulence.omega, ω_inlet) # ω_inlet
initialise!(model.turbulence.nut, k_inlet/ω_inlet) # k_inlet/ω_inlet

residuals = run!(model, config, output=OpenFOAM()) # 36.90k allocs

# Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
# Fp = pressure_force(:wall, model.momentum.p, 1.25)
# Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)


# plot(; xlims=(0,494))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

# using Profile, PProf

# GC.gc()

# initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.p, 0.0)
# initialise!(model.turbulence.k, k_inlet)
# initialise!(model.turbulence.omega, ω_inlet)
# initialise!(model.turbulence.nut, νt_inlet)

# residuals = run!(model, config)

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=0.1 begin 
# residuals = run!(model, config)
# end

# PProf.Allocs.pprof()