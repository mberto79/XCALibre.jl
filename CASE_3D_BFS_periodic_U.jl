using FVM_1D
using CUDA
using Adapt

mesh_file = "unv_sample_meshes/BFS_UNV_3D_hex_5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CUDABackend()
mesh_dev = adapt(backend, mesh)
# mesh_dev = mesh

velocity = [1.5, 0.0, 0.8]
nu = 1e-4
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    # turbulence = RANS{Laminar}(),
    turbulence = LES{Smagorinsky}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

periodic = construct_periodic(mesh, backend, :side1, :side2)
    
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0]),
    # Neumann(:top, 0.0),
    periodic...
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    periodic...
)

schemes = (
    U = set_schemes(time=Euler, divergence=LUST, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
    # p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        # relax       = 0.8,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-5
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        # relax       = 0.2,
        relax       = 0.7,
        rtol = 0.0,
        atol = 1e-6
    )
)

runtime = set_runtime(
    iterations=2000, time_step=1e-3, write_interval=50)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, [0.0,0.0,0.0])
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

using Profile, PProf

GC.gc()
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rz, Rp, model = run!(model, config)
end

PProf.Allocs.pprof()

test(::Nothing, a) = print("nothing")
test(b, a) = print(a*a)

test(nothing, 1)