using XCALibre
# using CUDA # uncomment to run on an NVIDIA GPU

# reorder_mesh! renumbers cells, faces and nodes so neighbouring cells sit close in memory.
# This example times the same run on the mesh in file order and reordered.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "bfs_unv_tet_10mm.unv")

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
# backend = CUDABackend(); workgroup = 32

function run_case(mesh, backend, workgroup; iterations=100)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    mesh_dev = adapt(backend, mesh)
    velocity = [0.5, 0.0, 0.0]
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu = 1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev
        )
    BCs = assign(
        region = mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]),
                Zerogradient(:sides),
                Zerogradient(:top)
            ],
            p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:wall),
                Extrapolated(:sides),
                Extrapolated(:top)
            ]
        )
    )
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.8, rtol=0.1),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, relax=0.2, rtol=0.01)
    )
    schemes = (
        U = Schemes(divergence=Upwind, gradient=Gauss),
        p = Schemes(gradient=Gauss)
    )
    config(n) = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=n, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    timed_run(n) = begin
        initialise!(model.momentum.U, velocity)
        initialise!(model.momentum.p, 0.0)
        @elapsed run!(model, config(n); progress=false)
    end
    timed_run(1) # compile
    timed_run(iterations)
end

file_order = UNV3D_mesh(mesh_file, scale=0.001)
reordered = reorder_mesh!(UNV3D_mesh(mesh_file, scale=0.001))

t_file = run_case(file_order, backend, workgroup)
t_reordered = run_case(reordered, backend, workgroup)
println("file order: $(round(t_file, digits=2)) s, reordered: $(round(t_reordered, digits=2)) s")
