using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend = CPU()
workgroup = 1024
mesh_dev = adapt(backend, mesh)

U0 = 0.3
A = 0.5
frequency = 1
velocity = [1.5*U0, U0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@inline inflow(vec, t, i) = begin
    u = U0*(1 + A*cospi(2*frequency*t))
    v = U0*(1+ A*sinpi(2*frequency*t))
    return velocity = SVector{3}(u, v, 0.0)
end

BCs = assign(
    region=mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, inflow),
            Extrapolated(:outlet),
            # Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:bottom, [0.0, 0.0, 0.0]),
            Symmetry(:top, [0.0, 0.0, 0.0])
    ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            # Extrapolated(:wall),
            Symmetry(:bottom),
            Symmetry(:top)
        ]
    )
)

schemes = (
    U = Schemes(time=Euler),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # DILU(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), #LDL(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    )
)

iterations = 100
timestep = 0.01
runtime = Runtime(iterations=iterations, time_step=timestep, write_interval=-1)
hardware = Hardware(backend=backend,workgroup = workgroup)

# records U at every time step, for an exact reference RMS
struct RMSProbe{F}
    U::F
    ux::Vector{Vector{Float64}}
    uy::Vector{Vector{Float64}}
end
XCALibre.convert_time_to_iterations(p::RMSProbe, model, dt, iterations) = p
function XCALibre.runtime_postprocessing!(p::RMSProbe, iter::Integer, n::Integer, S, time, config)
    push!(p.ux, copy(p.U.x.values)); push!(p.uy, copy(p.U.y.values))
    nothing
end
probe = RMSProbe(model.momentum.U, Vector{Float64}[], Vector{Float64}[])

postprocess = [FieldRMS(model.momentum.U; name="Urms"), FieldRMS(model.momentum.U;name="Urms_stop50", stop = 50*timestep,update_interval = 3*timestep), FieldRMS(model.momentum.U;name="Urms_start51", start = 51*timestep, update_interval = timestep/2), probe]
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
residuals = run!(model, config; progress=false);

#check middle 10 cells of inlet agree with analytical rms 
u_rms_exact = U0*A/sqrt(2) 
v_rms_exact = U0*A/sqrt(2)
u_rms = sum(postprocess[1].rms.x.values[end-25:end-15])/length(postprocess[1].rms.x.values[end-25:end-15])
v_rms = sum(postprocess[1].rms.y.values[end-25:end-15])/length(postprocess[1].rms.y.values[end-25:end-15])

# FieldRMS against the RMS of the recorded samples, in every cell
sample_rms(u) = (m = sum(u)/length(u); sqrt.(max.(sum(x -> x.^2, u)/length(u) .- m.^2, 0.0)))
@test postprocess[1].rms.x.values ≈ sample_rms(probe.ux) rtol = 1e-8
@test postprocess[1].rms.y.values ≈ sample_rms(probe.uy) rtol = 1e-8

# sanity bound only: the analytic value is the RMS of the inlet face, the samples are the first
# cells, whose RMS depends on the discretisation on this coarse grid (on generated 40², 80² and 160²
# grids the first-cell RMS of two face-flux treatments differs by 2.3e-3, 5e-5 and 1e-8)
@test u_rms ≈ u_rms_exact rtol = 0.1
@test v_rms ≈ v_rms_exact atol = 0.03

#testing start and end and update_interval logic
u_rms_first_half = sum(postprocess[2].rms.x.values[end-25:end-15])/length(postprocess[2].rms.x.values[end-25:end-15])
u_rms_second_half = sum(postprocess[3].rms.x.values[end-25:end-15])/length(postprocess[3].rms.x.values[end-25:end-15])

@test u_rms ≈ u_rms_first_half atol = 0.005
@test u_rms ≈ u_rms_second_half atol = 0.005
