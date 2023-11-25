using Plots
using Krylov
using StaticArrays
using CUDA
using FVM_1D

function test_kernel_cells!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign) = mesh
    
    @inbounds if i <= length(cells) && i > 0

        (; centre, volume, nodes_range, faces_range) = cells[i]
        
        for j in nodes_range

        cell_nodes[j] = cell_nodes[j] + cell_nodes[j]

        end

        for k in faces_range

            cell_faces[k] = cell_faces[k]+cell_faces[k]
            cell_neighbours[k] = cell_neighbours[k] + cell_neighbours[k]
            cell_nsign[k] = cell_nsign[k] + cell_nsign[k]

        end

        centre = centre + centre
        volume = volume + volume

        cells[i] = Cell{Int64, Float64}(
            centre,
            volume,
            nodes_range,
            faces_range
        )

    end

    # mesh = Mesh2(
    #     cells,
    #     cell_nodes,
    #     cell_faces,
    #     cell_neighbours,
    #     cell_nsign,
    #     mesh.faces,
    #     mesh.face_nodes,
    #     mesh.boundaries,
    #     mesh.nodes,
    # )

    return nothing
end 

function test_kernel_faces!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; faces, face_nodes) = mesh

    @inbounds if i <= length(faces) && i > 0

        (; nodes_range, ownerCells, centre, normal, e, area, delta, weight) = faces[i]
       
        ownerCells = ownerCells + ownerCells
        centre = centre + centre
        normal = normal + normal
        e = e + e
        area = area + area
        delta = delta + delta
        weight = weight + weight

        for j in nodes_range
            face_nodes[j] = face_nodes[j] + face_nodes[j]
        end

        faces[i] = Face2D{Int64,Float64}(
        nodes_range,
        ownerCells,
        centre,
        normal,
        e,
        area,
        delta,
        weight
        ) 

    end

    # mesh = Mesh2(
    #     mesh.cells,
    #     mesh.cell_nodes,
    #     mesh.cell_faces,
    #     mesh.cell_neighbours,
    #     mesh.cell_nsign,
    #     faces,
    #     face_nodes,
    #     mesh.boundaries,
    #     mesh.nodes,
    # )

    return nothing
end

function test_kernel_boundaries!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; boundaries) = mesh

    for j in eachindex(boundaries)

        (; name, facesID, cellsID) = boundaries[j]

        # @cuprint(name)

        @inbounds if i <= length(facesID) && i > 0

            facesID[i] = facesID[i] + facesID[i]
            cellsID[i] = cellsID[i] + cellsID[i]

        end

        boundaries[j] = Boundary{Vector{Int64}}(
            name,
            facesID,
            cellsID
        )

    end

    # mesh = Mesh2(
    #     mesh.cells,
    #     mesh.cell_nodes,
    #     mesh.cell_faces,
    #     mesh.cell_neighbours,
    #     mesh.cell_nsign,
    #     mesh.faces,
    #     mesh.face_nodes,
    #     boundaries,
    #     mesh.nodes,
    # )

    return nothing

end

function test_kernel_nodes!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; nodes) = mesh

    @inbounds if i <= length(nodes) && i > 0
        (; coords, neighbourCells) = nodes[i]

        coords = coords + coords

        # for j in eachindex(neighbourCells)
        #     neighbourCells[j] = neighbourCells[j] + neighbourCells[j]
        # end

        nodes[i] = Node{Vector{Int64},Float64}(
            coords,
            neighbourCells
        )

    end

    # mesh = Mesh2(
    #     mesh.cells,
    #     mesh.cell_nodes,
    #     mesh.cell_faces,
    #     mesh.cell_neighbours,
    #     mesh.cell_nsign,
    #     mesh.faces,
    #     mesh.face_nodes,
    #     mesh.boundaries,
    #     nodes,
    # )

    return nothing

end

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

# CELLS
mesh.cell_nodes
mesh.cell_faces
mesh.cells[1].centre

@cuda threads = 1024 blocks = cld(length(mesh.cells),1024) test_kernel_cells!(mesh)

mesh.cell_nodes
mesh.cell_faces
mesh.cells[1].centre

# FACES
mesh.faces[1].ownerCells
mesh.face_nodes

@cuda threads = 1024 blocks = cld(length(mesh.faces),1024) test_kernel_faces!(mesh)

mesh.faces[1].ownerCells
mesh.face_nodes


# BOUNDARIES
mesh.boundaries

@cuda threads = 1024 test_kernel_boundaries!(mesh)

mesh.boundaries

# NODES
#=
    potential to use pointer/range method to index neighbour cells since the vectors are not of the same size
        possibly more performant than eachindex()
=#
mesh.nodes[1].coords
mesh.nodes[1].neighbourCells

@cuda threads = 1024 blocks = cld(length(mesh.nodes),1024) test_kernel_nodes!(mesh)

mesh.nodes[1].coords
mesh.nodes[1].neighbourCells

#=
    BUG REPORT:
        - Illegal memory accesses error thrown when executing kernels, but not consistently
        - Error can occur when executing any of these kernels
        - The 4 kernels in this function can't all be run before error occurs
        - Only began occuring after implementing boundaries kernels
        - Only got worse after implementing nodes kernels
        - Kernels execute correctly when memory is not accessed illegally
=#