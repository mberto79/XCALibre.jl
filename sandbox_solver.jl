using FVM_1D
using Plots
using Krylov

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"

mesh=build_mesh3D(unv_mesh)
mesh.cell_neighbours

velocity = [0.5,0.0,0.0]
nu=1e-3
Re=velocity[1]*0.1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Dirichlet(:wall_top, [0.0, 0.0, 0.0]),
    Dirichlet(:wall_1, [0.0, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall_bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:wall_2, [0.0, 0.0, 0.0])
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Neumann(:wall_top, 0.0),
    Neumann(:wall_1, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall_bottom, 0.0),
    Neumann(:wall_2, 0.0)
)

schemes = (
    U = set_schemes(),
    p = set_schemes()
)

solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=-1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rz, Rp = simple!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rz), Rz, yscale=:log10, label="Uz")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

using Profile, PProf

GC.gc()
initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rz, Rp = simple!(model, config)
end

PProf.Allocs.pprof()

#function boundary_index(
    #boundaries::Vector{Boundary}, name::Symbol,
    #) 
    bci = zero(Int)
    for i ∈ eachindex(mesh.boundaries)
        bci += one(Int)
        if mesh.boundaries[i].name == name
            return bci 
        end
    end
#end

name=Symbol("outlet")
boundary_index(mesh.boundaries,name)
mesh.boundaries