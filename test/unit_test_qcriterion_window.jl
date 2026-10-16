using XCALibre
# Qcriterion must honour its start/stop/update_interval window like the other runtime
# post-processing types: Q is computed only on iterations start:update_interval:stop and
# keeps its last computed value (zero before start) outside it.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)
backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh_dev = adapt(backend, mesh)

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=1e-3),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev)

U = model.momentum.U
gradU = Grad{Gauss}(U)
S = StrainRate(gradU, T(gradU), U, FaceVectorField(mesh_dev))

# Solid-body rotation, dU/dy = 1 and dV/dx = -1: zero strain, |Ω|² = 2, so Q = 1
gradU.result.xy.values .= 1.0
gradU.result.yx.values .= -1.0

config = (hardware=hardware,) # magnitude2! only reads config.hardware
iterations = 20
dt = 0.25 # exact in binary, so the time-to-iteration conversion is exact

# window: start = 4 dt, stop = 12 dt, every 3 dt -> iterations 4, 7, 10
QC = convert_time_to_iterations(
    Qcriterion(U; start=4dt, stop=12dt, update_interval=3dt), model, dt, iterations)
@test (QC.start, QC.stop, QC.update_interval) == (4, 12, 3)

sentinel = -7.0
computed_at = Int[]
for iter ∈ 1:iterations
    QC.Q.values .= sentinel
    runtime_postprocessing!(QC, iter, iterations, S, iter*dt, config)
    if all(==(1.0), QC.Q.values)
        push!(computed_at, iter)
    else
        @test all(==(sentinel), QC.Q.values) # untouched outside the window
    end
end
@test computed_at == [4, 7, 10]

# No window given: Q is computed on every iteration
QCall = convert_time_to_iterations(Qcriterion(U), model, dt, iterations)
computed_at = Int[]
for iter ∈ 1:iterations
    QCall.Q.values .= sentinel
    runtime_postprocessing!(QCall, iter, iterations, S, iter*dt, config)
    all(≈(1.0), QCall.Q.values) && push!(computed_at, iter)
end
@test computed_at == collect(1:iterations)

# Through the postprocess-vector method used by the solvers: a window that starts
# after the first iteration leaves Q at zero on that iteration
QClate = convert_time_to_iterations(
    Qcriterion(U; start=(iterations+5)*dt), model, dt, iterations)
@test QClate.start == iterations # clamped
runtime_postprocessing!([QClate], 1, iterations, S, dt, config)
@test all(iszero, QClate.Q.values)
