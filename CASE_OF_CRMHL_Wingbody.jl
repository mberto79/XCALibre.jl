using Plots
using FVM_1D
using CUDA


mesh_file = "unv_sample_meshes/OF_CRMHL_Wingbody_1v/polyMesh/"
@time mesh = load_foamMesh(mesh_file, scale=0.001, integer_type=Int64, float_type=Float64)

# # check volume calculation
# volumes = ScalarField(mesh)
# vols = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]
# volumes.values .= vols
# write_vtk("cellVolumes", mesh, ("cellVolumes", volumes))

Umag = 10
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-05
νR = 100
Tu = 0.01
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
nut_inlet = k_inlet/ω_inlet

model = Physics(
    time = Steady(),
    fluid = Incompressible(nu = ConstantScalar(nu)),
    turbulence = RANS{KOmega}(),
    # turbulence = RANS{Laminar}(),
    # turbulence = LES{Smagorinsky}(),
    energy = nothing,
    domain = mesh
    )

walls = [:Fuselage, :FuselageAft, :Windshield, :WindshieldFrame, :WingLower, 
            :WingTEIB, :WingTEOB, :WingTip, :WingUpper]

freestream = [:Ymax, :Zmax, :Zmin]

@assign! model momentum U (
    Dirichlet(:Xmin, velocity), # inlet
    Neumann(:Xmax, 0.0), # outlet
    Dirichlet.(walls, Ref(noSlip))..., # walls
    Neumann.(freestream, Ref(0.0))...,
    Dirichlet(:Symmetry, velocity)
)

@assign! model momentum p (
    Neumann(:Xmin, 0.0), # inlet
    Dirichlet(:Xmax, 0.0), # outlet
    Neumann.(walls, Ref(0.0))..., # walls
    Neumann.(freestream, Ref(0.0))...,
    Neumann(:Symmetry, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:Xmin, k_inlet), # inlet
    Neumann(:Xmax, 0.0), # outlet
    KWallFunction.(walls)..., # walls
    Neumann.(freestream, Ref(0.0))...,
    Neumann(:Symmetry, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:Xmin, ω_inlet), # inlet
    Neumann(:Xmax, 0.0), # outlet
    OmegaWallFunction.(walls)..., # walls
    Neumann.(freestream, Ref(0.0))...,
    Neumann(:Symmetry, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:Xmin, k_inlet/ω_inlet), # inlet
    Neumann(:Xmax, 0.0), # outlet
    NutWallFunction.(walls)..., # walls
    Neumann.(freestream, Ref(0.0))...,
    Neumann(:Symmetry, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(divergence=Upwind, gradient=Midpoint),
    omega = set_schemes(divergence=Upwind, gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-3,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #LDL(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = set_solver(
        # model.turbulence.fields.k;
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-3,
        atol = 1e-10
    ),
    omega = set_solver(
        # model.turbulence.fields.omega;
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver, CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-3,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=2000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=8)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

# initialise!(model.momentum.U, [0,0,0])
# initialise!(model.momentum.p, 0.0)
# initialise!(model.turbulence.k, k_inlet)
# initialise!(model.turbulence.omega, ω_inlet)
# initialise!(model.turbulence.nut, 0.0)

Rx, Ry, Rz, Rp, model = run!(model, config); #, pref=0.0)


Reff = stress_tensor(model.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.p, 1.25)
Fv = viscous_force(:cylinder, model.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")