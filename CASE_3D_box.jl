using Plots
using FVM_1D
using FVM_1D.UNV_3D
using Krylov

#mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.04m.unv"
mesh_file="unv_sample_meshes/3d_streamtube_0.5x0.1x0.1_0.03m.unv"
mesh_file="unv_sample_meshes/3d_streamtube_0.5x0.1x0.1_0.015m.unv"

@time mesh=build_mesh3D(mesh_file)

_boundary_faces(mesh)
@time faces_checked, results = check_face_owners(mesh)

velocity = [0.01,0.0,0.0]
velocity = [0.05,0.0,0.0]
nu=1e-3
Re=velocity[1]*0.1/nu
noSlip = [0.0, 0.0, 0.0]

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, noSlip),
    Dirichlet(:side1, noSlip),
    Dirichlet(:side2, noSlip)
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),
    # Neumann(:bottom, 0.0),
    # Neumann(:top, 0.0),
    # Neumann(:side1, 0.0),
    # Neumann(:side2, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-4

    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, [0.0,0.0,0.0])
initialise!(model.p, 0.0)

# model2vtk(model, "iteration_0")

# backend = CUDABackend()

Rx, Ry, Rz, Rp = simple!(model, config)

plot(; xlims=(0,400))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rz), Rz, yscale=:log10, label="Uz")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(ylabel="Residuals")
plot!(xlabel="Iterations")

using Profile, PProf

GC.gc()
initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rz, Rp = simple!(model, config)
end

PProf.Allocs.pprof()
