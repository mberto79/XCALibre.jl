using Plots
using FVM_1D
using Krylov
using StaticArrays
using CUDA

abstract type AbstractMesh end

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

# CuArray(mesh.cells[1].centre)
# cu(mesh.cells[1].volume)
# cu(mesh.cell_nodes[mesh.cells[1].nodes_range])
# cu(mesh.cell_faces[mesh.cells[1].faces_range])
# cu(mesh.cell_neighbours[mesh.cells[1].faces_range])
# cu(mesh.cell_nsign[mesh.cells[1].faces_range])



#=
Cells kernel and function
    Three options exist for implementation:
        1. Write individual functions for each kernel to be executed where only variables to be used in the kernel are
        converted to isbits - too complicated
        2. Write one master function which converts all cell variables to isbits and use them as needed in program!!
        3. Convert these functions to structs and call "meshGPU" into kernels
            3.a. The structs will be identical to the current Mesh2 struct, but all vector types will be replaced
            with CuArray types
    Some possible solutions to potential problems:
        1. Indexing each array based on nodes_range or faces_range, as relevant, and passing each individual array into
        the kernel
            1.a. Con - Will cause more memory copies to and from GPU, likely to be less performant
            1.b. Pro - Will allow direct indexing of variables and prevents for loops therefore preventing potential
            thread divergence
            1.c. Con - More if statements needed within the kernel, may reduce performance
            1.d. Conclusion - this will only be needed if thread divergence becomes an issue - there may be ways around
            this by cleverly partitioning the thread execution 
=#
# Calling cells function and cehcking outputs
cell_test = test_cells(mesh)




#=
Faces kernel and function
    When declaring a CuArray of SVectors wrapped in a kernel, each index will correspond to the SVector of the same index
    in the CuArray - the individual elements of each SVector CANNOT be queried directly in the kernel as the program will
    crash
    Push used to force the variables into arrays - changes philosophy of current FVM algorithm from indexing individual
    array elements to passing entire arrays into calculation functions
    More setup prior to function will likely be required to save memory on GPU
=#
# Calling faces function and checking outputs
test_faces = test_faces(mesh)

test_faces[1] # nodes_range
test_faces[2] # ownerCells
test_faces[3] # centre
test_faces[4] # normal
test_faces[5] # e
test_faces[6] # area
test_faces[7] # delta
test_faces[8] # weight

test_faces[1] # nodes_range
test_faces[2] # ownerCells
test_faces[3] # centre
test_faces[4] # normal
test_faces[5] # e
test_faces[6] # area
test_faces[7] # delta
test_faces[8] # weight


#=
Boundaries kernel and function
    Boundaries are unique to their given name (inlet, wall, outlet, etc.) and so the vectors corresponding to each boundary
    name should be declared individually in the kernel
        The kernel call must be wrapped in a for loop - this will cause additional memory copies but seems necessary for
        functionality
    Potential solution 1 - change the type of boundaries to an SVector, but this will hard-code the number of elements in the
    boundary arrays which isn't very practical
    Potential solution 2 - uses the UnitRange technique already used in cells, but this may remove the segregation by
    boundary name
=#
#Calling boundaries function and checking outputs
test_boundaries = test_boundary(mesh)

# test_boundaries[1] # name
test_boundaries[1] # facesID
test_boundaries[2] # nodesID



## CELLS FUNCTION AND KERNEL
function test_cells(mesh::Mesh2)

    nodes_range = UnitRange{Int64}[]
    faces_range = UnitRange{Int64}[]
    volume = Float64[]
    centre = SVector{3,Float64}[]

    for i in 1:length(mesh.cells)
        nr = mesh.cells[i].nodes_range
        fr = mesh.cells[i].faces_range
        v = mesh.cells[i].volume
        c = mesh.cells[i].centre

        #=
            note - push! function used as arrays are necessary for indexing in kernels - the current algorithm method of
            directly calculating variables in for loops will not work in kernels
        =#

        push!(nodes_range, nr)
        push!(faces_range, fr)
        push!(volume, v)
        push!(centre, c)

    end

    nodes = cu(mesh.cell_nodes)
    faces = cu(mesh.cell_faces)
    neighbours = cu(mesh.cell_neighbours)
    nsign = cu(mesh.cell_nsign)
    volume = cu(volume)
    centre = CuArray(centre)
    nodes_range = cu(nodes_range)
    faces_range = cu(faces_range)

    @cuda threads = 1024 blocks = cld(length(nodes_range),1024) test_kernel_cells!(nodes, faces, neighbours, nsign,
                                                                             volume, centre, nodes_range, faces_range)
    

    return nodes, faces, neighbour, nsign, centre, volume
    
end

function test_kernel_cells!(nodes, faces, neighbours, nsign, volume, centre, nr, fr)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    
    @inbounds if i <= length(nr) && i > 0
        
        for j in nr[i]

        nodes[j] = nodes[j] + nodes[j]

        end

        for k in fr[i]

            faces[k] = faces[k]+faces[k]
            neighbours[k] = neighbours[k] + neighbours[k]
            nsign[k] = nsign[k] + nsign[k]

        end

        
        centre[i] = centre[i] + centre[i]

        volume[i] = volume[i] + volume[i]

    end

    return nothing
end  



## FACES FUNCTION AND KERNEL
function test_faces(mesh)

    nodes_range = UnitRange{Int64}[]
    ownerCells = SVector{2,Int64}[] #Need to check if this needs to be {3,I} for 3D
    centre = SVector{3,Float64}[]
    normal = SVector{3,Float64}[]
    e = SVector{3,Float64}[]
    area = Float64[]
    delta = Float64[]
    weight = Float64[]
    
    for i in 1:length(mesh.faces)
        nr = mesh.faces[i].nodes_range
        oc = mesh.faces[i].ownerCells
        c = mesh.faces[i].centre
        n = mesh.faces[i].normal
        e_var = mesh.faces[i].e
        a = mesh.faces[i].area
        d = mesh.faces[i].delta
        w = mesh.faces[i].weight

        #=
            note - push! function used as arrays are necessary for indexing in kernels - the current algorithm method of
            directly calculating variables in for loops will not work in kernels
        =#

        push!(nodes_range, nr)
        push!(ownerCells,oc)
        push!(centre, c)
        push!(normal, n)
        push!(e, e_var)
        push!(area, a)
        push!(delta, d)
        push!(weight, w)

    end

    nodes_range = cu(nodes_range)
    ownerCells = CuArray(ownerCells)
    centre = CuArray(centre)
    normal = CuArray(normal)
    e = CuArray(e)
    area = cu(area)
    delta = cu(delta)
    weight = cu(weight)

    @cuda threads = 1024 blocks = cld(length(nodes_range),1024) test_kernel_faces!(ownerCells, centre, normal, e, area, delta, weight)

    return nodes_range, ownerCells, centre, normal, e, area, delta, weight

end

function test_kernel_faces!(ownerCells, centre, normal, e, area, delta, weight)
    i = threadIdx().x

    @inbounds if i <= length(ownerCells) && i > 0
       
        ownerCells[i] = ownerCells[i] + ownerCells[i]
        centre[i] = centre[i] + centre[i]
        normal[i] = normal[i] + normal[i]
        e[i] = e[i] + e[i]
        area[i] = area[i] + area[i]
        delta[i] = delta[i] + delta[i]
        weight[i] = weight[i] + weight[i]

    end

    return nothing

end


## BOUNDARY CONDITIONS FUNCTION AND KERNEL
# function test_boundary(mesh::Mesh2)

    name = Symbol[]
    facesID = Vector{Int64}[]
    cellsID = Vector{Int64}[]

    for i in 1:length(mesh.boundaries)
        n = cu(mesh.boundaries[i].name)
        f = mesh.boundaries[i].facesID
        c = mesh.boundaries[i].cellsID

        push!(name, n)
        push!(facesID, f)
        push!(cellsID, c)
    end

    facesID[1]

    name = cu.(name)
    facesID = cu.(facesID)
    cellsID = cu.(cellsID)

    @cuda threads = 1024 test_kernel_boundary!(facesID, cellsID)

    return facesID, cellsID

# end

function test_kernel_boundary!(facesID, cellsID)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    @inbounds if i <= length(ownerCells) && i > 0
        
        # name[i] = "Thread $i executed"

        for j in 1:length(facesID[i])
            facesID[j] = faceID[j] + facesID[j]
        end

        for k in 1:length(cellsID[i])
            cellsID[k] = cellsID[k] + cellsID[k]
        end
    end

    return nothing

end



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