using Plots
using XCALibre
using Krylov
using KernelAbstractions
using CUDA

#mesh_file="src/UNV3/5_cell_new_boundaries.unv"
mesh_file="src/UNV3/5_cell_new_boundaries.unv"
mesh_file="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.08mm.unv"
mesh_file="unv_sample_meshes/3d_streamtube_0.5x0.1x0.1_0.03m.unv"
mesh_file="unv_sample_meshes/3d_streamtube_0.5x0.1x0.1_0.015m.unv" # Converges

@time mesh=UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.01,0.0,0.0]
nu=1e-3
Re=velocity[1]*0.1/nu
noSlip = [0.0, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, noSlip),
    Dirichlet(:side1, noSlip),
    Dirichlet(:side2, noSlip),
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, noSlip),
    # Dirichlet(:top, noSlip),
    # Dirichlet(:side1, noSlip),
    # Dirichlet(:side2, noSlip)
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
)

@assign! model momentum p (
    Neumann(:inet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
)

# schemes = (
#     U = Schemes(time=Euler, divergence=Upwind, gradient=Midpoint),
#     p = Schemes(time=Euler, divergence=Upwind, gradient=Midpoint)
# )

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes()
)

solvers = (
    U = SolverSetup(
        model.momentum.U;
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-4,
        atol = 1e-4
    ),
    p = SolverSetup(
        model.momentum.p;
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        # relax       = 1.0,
        rtol = 1e-4

    )
)

runtime = Runtime(iterations=10, time_step=1, write_interval=5)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.U, [0.0,0.0,0.0])
initialise!(model.momentum.p, 0.0)

backend = CUDABackend()
# backend = CPU()

residuals = run!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rz), Rz, yscale=:log10, label="Uz")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

using Profile, PProf

GC.gc()
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    residuals = run!(model, config)
end

PProf.Allocs.pprof()
