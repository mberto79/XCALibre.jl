using Plots
using FVM_1D
using Krylov
using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

# mesh.cell_faces[mesh.cells[800].faces_map[1]:mesh.cells[800].faces_map[end]]

facesID_gpu = []
nodesID_gpu = []
centres_gpu = []
volumes_gpu = []

faces_range_gpu = []
nodes_range_gpu = []

for i in 1:length(mesh.cells)

    cell_faces_cpu = mesh.cell_faces[mesh.cells[i].faces_range]
    cell_nodes_cpu = mesh.cell_nodes[mesh.cells[i].nodes_range]
    cell_centres_cpu = mesh.cells[i].centre
    cell_volumes_cpu = mesh.cells[i].volume
    # faces_nodes_cpu = mesh.faces[mesh.faces[i].nodes_range]

    push!(facesID_gpu, cell_faces_cpu...)
    push!(nodesID_gpu, cell_nodes_cpu...)
    push!(centres_gpu, cell_centres_cpu...)
    push!(volumes_gpu, cell_volumes_cpu...)
    push!(faces_range_gpu, mesh.cells[i].faces_range)
    push!(nodes_range_gpu, mesh.cells[i].nodes_range)    

end

function test_kernel(f, n, c, v, fr, nr)
    i = threadIdx().x
    
    @inbounds if i <= length(fr) && i > 0
        
        for j in fr[i]
           
            f[j] = f[j] + f[j]
            
        end

    end

    return nothing

end

@cuda threads = length(facesID_gpu) test_kernel!(facesID_gpu, nodesID_gpu, centres_gpu, volumes_gpu,
                                                faces_range_gpu, nodes_range_gpu)

# solver 
velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
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
    iterations=1000, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rp = simple!(model, config) # 9.39k allocs

plot(; xlims=(0,184))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

# using Profile, PProf

# GC.gc()
# initialise!(U, velocity)
# initialise!(p, 0.0)

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=1 begin Rx, Ry, Rp = isimple!(
#     mesh, nu, U, p,
#     # setup_U, setup_p, iterations, pref=0.0)
#     setup_U, setup_p, iterations)
# end

# PProf.Allocs.pprof()