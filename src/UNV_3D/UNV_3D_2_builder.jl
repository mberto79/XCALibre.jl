export build_mesh3D
#push! is BANNED! THOSE FOUND USING PUSH WILL BE EXILED!
# [] is also BANNED! No more kneecaps if found using []!
#Exceptions apply to where I have no idea what the size will be

function build_mesh3D(unv_mesh; integer=Int64, float=Float64)
    stats = @timed begin
        println("Loading UNV File...")
        points, edges, bfaces, volumes, boundaryElements = load_3D(
            unv_mesh; integer=integer, float=float)
        println("File Read Successfully")
        println("Generating Mesh...")

        node_cells, cells_range = generate_node_cells(points, volumes) #Rewritten, optimized
        nodes = generate_nodes(points, cells_range) #Rewritten, optimzied

        ifaces, faces,cell_face_nodes = generate_tet_internal_faces(volumes, bfaces) # Temp fix added, New method appraoch needed
        #faces=quad_internal_faces(volumes,faces)

        face_nodes = generate_face_nodes(faces) #Removed push
        cell_nodes = generate_cell_nodes(volumes) #Removed push

        all_cell_faces = generate_all_cell_faces(faces, cell_face_nodes) # New method needed

        cell_nodes_range = generate_cell_nodes_range(volumes) #Removed push
        face_nodes_range = generate_face_nodes_range(faces) #Removed Push
        all_cell_faces_range = generate_all_cell_faces_range(volumes) #Removed push

        cells_centre = calculate_centre_cell(volumes, nodes) #Removed push

        boundary_faces, boundary_face_range = generate_boundary_faces(boundaryElements,bfaces) #Rewritten
        boundary_cells = generate_boundary_cells(bfaces, all_cell_faces, all_cell_faces_range) #Rewritten, error found, using face index of boundary_faces instead of bfaces

        cell_faces, cell_faces_range = generate_cell_faces(bfaces, volumes, all_cell_faces) # Removed push

        boundaries = generate_boundaries(boundaryElements, boundary_face_range) #Removed push

        face_ownerCells = generate_face_ownerCells(faces, all_cell_faces, all_cell_faces_range) #New method approach needed

        faces_area = calculate_face_area(nodes, faces) #Rewrite needed, removed push
        faces_centre = calculate_face_centre(faces, nodes) # Removed push
        faces_normal = calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) # Rewrite needed
        faces_e, faces_delta, faces_weight = calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, faces_normal) #Removed push

        cells_volume = calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, faces_normal, cells_centre, faces_centre, face_ownerCells, faces_area) #Removed push

        cells = generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range) #Removed push
        cell_neighbours = generate_cell_neighbours(cells, cell_faces) # Removed push, new method needed
        faces = generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight) #Removed push

        cell_nsign = calculate_cell_nsign(cells, faces, cell_faces) #removed push

        get_float = SVector(0.0, 0.0, 0.0)
        get_int = UnitRange(0, 0)

        mesh = Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes, boundaries, nodes, node_cells, get_float, get_int, boundary_cells)

    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return mesh
    #For unit testing
    #return mesh,cell_face_nodes, node_cells, all_cell_faces,boundary_cells,boundary_faces,all_cell_faces_range
end

# Node connectivity

function generate_node_cells(points, volumes)
    temp_node_cells = [Int64[] for _ ∈ eachindex(points)] # array of vectors to hold cellIDs

    # Add cellID to each point that defines a "volume"
    for (cellID, volume) ∈ enumerate(volumes)
        for nodeID ∈ volume.volumes
            push!(temp_node_cells[nodeID], cellID)
        end
    end

    node_cells_size = sum(length.(temp_node_cells)) # number of cells in node_cells

    index = 0 # change to node cells index
    node_cells = zeros(Int64, node_cells_size)
    cells_range = [UnitRange{Int64}(1, 1) for _ ∈ eachindex(points)]
    for (nodeID, cellsID) ∈ enumerate(temp_node_cells)
        for cellID ∈ cellsID
            index += 1
            node_cells[index] = cellID
        end
        cells_range[nodeID] = UnitRange{Int64}(index - length(cellsID) + 1, index)
    end
    return node_cells, cells_range
end

function generate_nodes(points, cells_range)
    nodes = [Node(SVector{3, Float64}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
    @inbounds for i ∈ eachindex(points)
        nodes[i] =  Node(points[i].xyz, cells_range[i])
    end
    return nodes
end

# DEFINE FUNCTIONS
function calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, face_normal)
    faces_e = Vector{SVector{3,Float64}}(undef,length(faces))
    faces_delta = Vector{Float64}(undef,length(faces))
    faces_weight = Vector{Float64}(undef,length(faces))
    for i = eachindex(faces) #Boundary Face
        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            delta = norm(d_cf)
            faces_delta[i] = delta
            e = d_cf / delta
            faces_e[i] = e
            weight = one(Float64)
            faces_weight[i] = weight

        else #Internal Face
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_1f = cf - c1
            d_f2 = c2 - cf
            d_12 = c2 - c1

            delta = norm(d_12)
            faces_delta[i] = delta
            e = d_12 / delta
            faces_e[i] = e
            weight = abs((d_1f ⋅ face_normal[i]) / (d_1f ⋅ face_normal[i] + d_f2 ⋅ face_normal[i]))
            faces_weight[i] = weight

        end
    end
    return faces_e, faces_delta, faces_weight
end

function calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) #Rewrite needed
    face_normal = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        n1 = nodes[faces[i].faces[1]].coords
        n2 = nodes[faces[i].faces[2]].coords
        n3 = nodes[faces[i].faces[3]].coords

        t1x = n2[1] - n1[1]
        t1y = n2[2] - n1[2]
        t1z = n2[3] - n1[3]

        t2x = n3[1] - n1[1]
        t2y = n3[2] - n1[2]
        t2z = n3[3] - n1[3]

        nx = t1y * t2z - t1z * t2y
        ny = -(t1x * t2z - t1z * t2x)
        nz = t1x * t2y - t1y * t2x

        magn2 = (nx)^2 + (ny)^2 + (nz)^2

        snx = nx / sqrt(magn2)
        sny = ny / sqrt(magn2)
        snz = nz / sqrt(magn2)

        normal = SVector(snx, sny, snz)
        face_normal[i] = normal

        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            if d_cf ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        else
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_12 = c2 - c1

            if d_12 ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        end
    end
    return face_normal
end

function calculate_face_centre(faces, nodes)
    face_centres = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(faces[i].faces))
        for ic = 1:length(faces[i].faces)
            temp_coords[ic] = nodes[faces[i].faces[ic]].coords
        end
        face_centres[i] = sum(temp_coords) / length(faces[i].faces)
    end
    return face_centres
end

function calculate_face_area(nodes, faces) # Need to shorten
    face_area= Vector{Float64}(undef,length(faces))
    for i = eachindex(faces)
        if faces[i].faceCount == 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2
            face_area[i]= area
        end

        if faces[i].faceCount > 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2

            for ic = 4:faces[i].faceCount
                n1 = nodes[faces[i].faces[ic]].coords
                n2 = nodes[faces[i].faces[2]].coords
                n3 = nodes[faces[i].faces[3]].coords

                t1x = n2[1] - n1[1]
                t1y = n2[2] - n1[2]
                t1z = n2[3] - n1[3]

                t2x = n3[1] - n1[1]
                t2y = n3[2] - n1[2]
                t2z = n3[3] - n1[3]

                area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
                area = area + sqrt(area2) / 2

            end

            face_area[i]=area

        end
    end
    return face_area
end

function generate_cell_faces(bfaces, volumes, all_cell_faces)
    cell_faces = Vector{Int64}[] # May be a way to preallocate this. For now will leave as []
    cell_face_range = Vector{UnitRange{Int64}}(undef,length(volumes))
    counter_start = 0
    x = 0
    max = length(bfaces)

    for i = eachindex(volumes)
        push!(cell_faces, all_cell_faces[counter_start+1:counter_start+length(volumes[i].volumes)])
        counter_start = counter_start + length(volumes[i].volumes)
        cell_faces[i] = filter(x -> x > max, cell_faces[i])

        if length(cell_faces[i]) == 1
            cell_face_range[i] = UnitRange(x + 1, x + 1)
            x = x + 1
        else
            cell_face_range[i] = UnitRange(x + 1, x + length(cell_faces[i]))
            x = x + length(cell_faces[i])
        end
    end
    cell_faces = reduce(vcat, cell_faces)

    return cell_faces, cell_face_range
end

function calculate_cell_nsign(cells, faces, cell_faces)
    cell_nsign = Vector{Int}(undef,length(cell_faces))
    counter=0
    for i = 1:length(cells)
        centre = cells[i].centre
        for ic = 1:length(cells[i].faces_range)
            fcentre = faces[cell_faces[cells[i].faces_range][ic]].centre
            fnormal = faces[cell_faces[cells[i].faces_range][ic]].normal
            d_cf = fcentre - centre
            fnsign = zero(Int)

            if d_cf ⋅ fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            counter=counter+1
            cell_nsign[counter] = fnsign
        end

    end
    return cell_nsign
end

function calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, face_normal, cells_centre, faces_centre, face_ownerCells, faces_area)
    cells_volume = Vector{Float64}(undef,length(volumes))
    for i = eachindex(volumes)
        volume = zero(Float64) # to avoid type instability
        for f = all_cell_faces_range[i]
            findex = all_cell_faces[f]

            normal = face_normal[findex]
            cc = cells_centre[i]
            fc = faces_centre[findex]
            d_fc = fc - cc

            if face_ownerCells[findex, 1] ≠ face_ownerCells[findex, 2]
                if dot(d_fc, normal) < 0.0
                    normal = -1.0 * normal
                end
            end

            volume = volume + (normal[1] * faces_centre[findex][1] * faces_area[findex])

        end
        cells_volume[i] = volume
    end
    return cells_volume
end

function calculate_centre_cell(volumes,nodes)
    cell_centres = Vector{SVector{3,Float64}}(undef,length(volumes))
    for i = eachindex(volumes)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(volumes[i].volumes))
        for ic = eachindex(volumes[i].volumes)
            temp_coords[ic] = nodes[volumes[i].volumes[ic]].coords
        end
        cell_centres[i] = sum(temp_coords) / length(volumes[i].volumes)
    end
    return cell_centres
end

# function calculate_centre_cell(volumes, nodes)
#     centre_store = SVector{3,Float64}[]
#     for i = 1:length(volumes)
#         cell_store = typeof(nodes[volumes[1].volumes[1]].coords)[]
#         for ic = 1:length(volumes[i].volumes)
#             push!(cell_store, nodes[volumes[i].volumes[ic]].coords)
#         end
#         centre = (sum(cell_store) / length(cell_store))
#         push!(centre_store, centre)
#     end
#     return centre_store
# end

function generate_cell_neighbours(cells, cell_faces)
    cell_neighbours = Vector{Int64}(undef,length(cell_faces))
    counter=0
    for ID = 1:length(cells)
        for i = cells[ID].faces_range
            faces = cell_faces[i]
            for ic = 1:length(i)
                face = faces[ic]
                index = findall(x -> x == face, cell_faces)
                if length(index) == 2
                    if i[1] <= index[1] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[2] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                    if i[1] <= index[2] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[1] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                end
            end
        end
    end
    return cell_neighbours
end

function generate_tet_internal_faces(volumes, bfaces) #temp fix
    cell_face_nodes = Vector{Int}[]
    counter=0

    for i = eachindex(volumes)
        cell_faces = zeros(Int, 4, 3)
        cell = sort(volumes[i].volumes)

        cell_faces[1, 1:3] = cell[1:3]
        cell_faces[2, 1:2] = cell[1:2]
        cell_faces[2, 3] = cell[4]
        cell_faces[3, 1] = cell[1]
        cell_faces[3, 2:3] = cell[3:4]
        cell_faces[4, 1:3] = cell[2:4]

        for ic = 1:4
            push!(cell_face_nodes, cell_faces[ic, :])
        end
    end

    sorted_faces = Vector{Int}[]
    for i = eachindex(bfaces)
        push!(sorted_faces, sort(bfaces[i].faces))
    end

    internal_faces = setdiff(cell_face_nodes, sorted_faces)

    ifaces=Vector{UNV_3D.Face}(undef,length(internal_faces))
    faces=Vector{UNV_3D.Face}(undef,(length(internal_faces)+length(bfaces)))

    for i = eachindex(bfaces)
        faces[i]=UNV_3D.Face(bfaces[i].faceindex , bfaces[i].faceCount, bfaces[i].faces)
    end

    bface_index=length(bfaces)

    for i = eachindex(internal_faces)
        counter=counter+1
        ifaces[i]=UNV_3D.Face(bfaces[end].faceindex + counter, length(internal_faces[i]), internal_faces[i])
        bface_index=bface_index+1
        faces[bface_index]=ifaces[i]
    end
    return ifaces, faces, cell_face_nodes
end

function quad_internal_faces(volumes, faces)
    store_cell_faces1 = Int64[]

    for i = 1:length(volumes)
        cell_faces = zeros(Int, 6, 4)

        cell_faces[1, 1:4] = volumes[i].volumes[1:4]
        cell_faces[2, 1:4] = volumes[i].volumes[5:8]
        cell_faces[3, 1:2] = volumes[i].volumes[1:2]
        cell_faces[3, 3:4] = volumes[i].volumes[5:6]
        cell_faces[4, 1:2] = volumes[i].volumes[3:4]
        cell_faces[4, 3:4] = volumes[i].volumes[7:8]
        cell_faces[5, 1:2] = volumes[i].volumes[2:3]
        cell_faces[5, 3:4] = volumes[i].volumes[6:7]
        cell_faces[6, 1] = volumes[i].volumes[1]
        cell_faces[6, 2] = volumes[i].volumes[4]
        cell_faces[6, 3] = volumes[i].volumes[5]
        cell_faces[6, 4] = volumes[i].volumes[8]

        for ic = 1:6
            push!(store_cell_faces1, cell_faces[ic, :])
        end
    end

    sorted_cell_faces = Int64[]
    for i = 1:length(store_cell_faces1)

        push!(sorted_cell_faces, sort(store_cell_faces1[i]))
    end

    sorted_faces = Int64[]
    for i = 1:length(faces)
        push!(sorted_faces, sort(faces[i].faces))
    end

    internal_faces = setdiff(sorted_cell_faces, sorted_faces)

    for i = 1:length(internal_faces)
        push!(faces, UNV_3D.Face(faces[end].faceindex + 1, faces[end].faceCount, internal_faces[i]))
    end
    return faces
end

function generate_boundaries(boundaryElements, boundary_face_range)
    boundaries = Vector{Boundary{Symbol,UnitRange{Int64}}}(undef,length(boundaryElements))
    for i = eachindex(boundaryElements)
        boundaries[i] = Boundary(Symbol(boundaryElements[i].name), boundary_face_range[i])
    end
    return boundaries
end

function generate_boundary_cells(bfaces, all_cell_faces, all_cell_faces_range)
    boundary_cells = Vector{Int64}(undef,length(bfaces))
    index_all_cell_faces = Vector{Int64}(undef,length(bfaces))
    for ic = eachindex(bfaces) 
        for i in eachindex(all_cell_faces) 
                if all_cell_faces[i] == bfaces[ic].faceindex
                    index_all_cell_faces[ic]=i
                end
        end
        for i = eachindex(all_cell_faces_range) 
            if all_cell_faces_range[i][1] <= index_all_cell_faces[ic] <= all_cell_faces_range[i][end]
                boundary_cells[ic]=i
            end
        end
    end
    return boundary_cells
end

# function generate_boundary_cells(boundary_faces, cell_faces, cell_faces_range)
#     boundary_cells = Int64[]
#     store = Int64[]
#     for ic = 1:length(boundary_faces)
#         for i in eachindex(cell_faces)
#             if cell_faces[i] == boundary_faces[ic]
#                 push!(store, i)
#             end
#         end
#     end
#     store

#     for ic = 1:length(store)
#         for i = 1:length(cell_faces_range)
#             if cell_faces_range[i][1] <= store[ic] <= cell_faces_range[i][end]
#                 push!(boundary_cells, i)
#             end
#         end
#     end
#     return boundary_cells
# end

function generate_boundary_faces(boundaryElements,bfaces) #Only works if all bc have more than 1 face, which is very unlikely
    boundary_faces = Vector{Int64}(undef,length(bfaces)) #Same length as bfaces
    counter = 0
    boundary_face_range = Vector{UnitRange{Int64}}(undef,length(boundaryElements))
    for i = eachindex(boundaryElements)
        for n = eachindex(boundaryElements[i].elements)
            counter=counter+1
            boundary_faces[counter] = boundaryElements[i].elements[n]
        end
        boundary_face_range[i] = UnitRange(boundaryElements[i].elements[1], boundaryElements[i].elements[end])
    end
    return boundary_faces, boundary_face_range
end

# function generate_boundary_faces(boundaryElements)
#     boundary_faces = Int64[]
#     z = 0
#     wipe = Int64[]
#     boundary_face_range = UnitRange{Int64}[]
#     for i = 1:length(boundaryElements)
#         for n = 1:length(boundaryElements[i].elements)
#             push!(boundary_faces, boundaryElements[i].elements[n])
#             push!(wipe, boundaryElements[i].elements[n])
#         end
#         if length(wipe) == 1
#             push!(boundary_face_range, UnitRange(boundaryElements[i].elements[1], boundaryElements[i].elements[1]))
#             z = z + 1
#         elseif length(wipe) ≠ 1
#             push!(boundary_face_range, UnitRange(boundaryElements[i].elements[1], boundaryElements[i].elements[end]))
#             z = z + length(wipe)
#         end
#         wipe = Int64[]
#     end
#     return boundary_faces, boundary_face_range
# end

function generate_face_ownerCells(faces, all_cell_faces, all_cell_faces_range)
    cell_face_index = Vector{Vector{Int64}}(undef, length(faces))
    for i = 1:length(cell_face_index)
        cell_face_index[i] = findall(x -> x == i, all_cell_faces)
    end

    face_owners = zeros(Int, length(cell_face_index), 2)
    for ic = 1:length(all_cell_faces_range)
        for i = 1:length(cell_face_index)
            if all_cell_faces_range[ic][1] <= cell_face_index[i][1] <= all_cell_faces_range[ic][end]
                face_owners[i, 1] = ic
                face_owners[i, 2] = ic
            end

            if length(cell_face_index[i]) == 2
                if all_cell_faces_range[ic][1] <= cell_face_index[i][2] <= all_cell_faces_range[ic][end]
                    face_owners[i, 2] = ic
                end
            end

        end
    end
    return face_owners
end

# function generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)
#     x=Vector{Int64}[]
#     for i=1:length(faces)
#         push!(x,findall(x->x==i,all_cell_faces))
#     end
#     y=zeros(Int,length(x),2)
#     for ic=1:length(volumes)
#         for i=1:length(x)
#             #if length(x[i])==1
#                 if all_cell_faces_range[ic][1]<=x[i][1]<=all_cell_faces_range[ic][end]
#                     y[i,1]=ic
#                     y[i,2]=ic
#                 end
#             #end

#             if length(x[i])==2
#                 if all_cell_faces_range[ic][1]<=x[i][2]<=all_cell_faces_range[ic][end]
#                     #y[i]=ic
#                     y[i,2]=ic

#                 end
#             end

#         end
#     end
#     return y
# end



# function generate_nodes(points,cells_range)
#     # nodes=Node{SVector{3,Float64}, UnitRange{Int64}}[]
#     nnodes = length(points)
#     nodes = [Node(SVector{3,Float64}(0.0,0.0,0.0), 1:1) for i ∈ 1:nnodes]
#     tnode = Node(SVector{3,Float64}(0.0,0.0,0.0), 1:1) # temporary node object used to rewrite
#     @inbounds for i ∈ 1:length(points)
#         #point=points[i].xyz
#         # push!(nodes,Node(points[i].xyz,cells_range[i]))
#         tnode = @reset tnode.coords = points[i].xyz
#         tnode = @reset tnode.cells_range = cells_range[i]
#         nodes[i] = tnode # overwrite preallocated array with temporary node
#     end
#     return nodes
# end

#Generate Faces

function generate_face_nodes(faces)
    face_nodes = Vector{Int64}(undef, length(faces) * 3) # number of bc faces times number of nodes per face. Tet Only for now.
    counter = 0
    for n = eachindex(faces)
        for i = 1:faces[n].faceCount
            counter = counter + 1
            face_nodes[counter] = faces[n].faces[i]
        end
    end
    return face_nodes
end

# function generate_face_nodes(faces)
#     face_nodes=typeof(faces[1].faces[1])[] # Giving type to array constructor
#     for n=1:length(faces)
#         for i=1:faces[n].faceCount
#             push!(face_nodes,faces[n].faces[i])
#         end
#     end
#     return face_nodes
# end

#Generate cells
function generate_cell_nodes(volumes)
    cell_nodes = Vector{Int64}(undef, length(volumes) * 4) #length of cells times number of nodes per cell
    counter = 0
    for n = eachindex(volumes)
        for i = 1:volumes[n].volumeCount
            counter = counter + 1
            cell_nodes[counter] = volumes[n].volumes[i]
        end
    end
    return cell_nodes
end

# function generate_cell_nodes(volumes)
#     cell_nodes=typeof(volumes[1].volumes[1])[] # Giving type to array constructor
#     for n=1:length(volumes)
#         for i=1:volumes[n].volumeCount
#             push!(cell_nodes,volumes[n].volumes[i])
#         end
#     end
#     return cell_nodes
# end

# function generate_all_cell_faces(faces,cell_face_nodes)
#     all_cell_faces=Int[]
#     sorted_faces=Vector{Int}[]
#     for i=1:length(faces)
#         push!(sorted_faces,sort(faces[i].faces))
#     end

#     for i=1:length(cell_face_nodes)
#         push!(all_cell_faces,findfirst(x -> x==cell_face_nodes[i],sorted_faces))
#     end
#     return all_cell_faces
# end

function generate_all_cell_faces(faces, cell_face_nodes)
    sorted_faces = Vector{Vector{Int64}}(undef, length(faces))
    for i = 1:length(faces)
        sorted_faces[i] = sort(faces[i].faces)
    end

    all_cell_faces = zeros(Int, length(cell_face_nodes)) #May only work for Tet
    for i = 1:length(cell_face_nodes)
        all_cell_faces[i] = findfirst(x -> x == cell_face_nodes[i], sorted_faces)
    end
    return all_cell_faces
end

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(volumes))
    x = 0
    for i = eachindex(volumes)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(volumes[i].volumes))
        x = x + length(volumes[i].volumes)
    end
    return cell_nodes_range
end

# function generate_cell_nodes_range(volumes)
#     cell_nodes_range=UnitRange(0,0)
#     store=typeof(cell_nodes_range)[]
#     x=0
#     for i=1:length(volumes)
#         cell_nodes_range=UnitRange(x+1,x+length(volumes[i].volumes))
#         x=x+length(volumes[i].volumes)
#         push!(store,cell_nodes_range)
#     end
#     return store
# end


function generate_face_nodes_range(faces)
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(faces))
    x = 0
    for i = eachindex(faces)
        face_nodes_range[i] = UnitRange(x + 1, x + faces[i].faceCount)
        x = x + faces[i].faceCount
    end
    return face_nodes_range
end

# function generate_face_nodes_range(faces)
#     face_nodes_range=UnitRange(0,0)
#     store=typeof(face_nodes_range)[]
#     x=0
#     for i=1:length(faces)
#         face_nodes_range=UnitRange(x+1,x+faces[i].faceCount)
#         x=x+faces[i].faceCount
#         push!(store,face_nodes_range)
#     end
#     return store
# end

function generate_all_cell_faces_range(volumes)
    cell_faces_range = Vector{UnitRange{Int64}}(undef, length(volumes))
    x = 0
    @inbounds for i = eachindex(volumes)
        #Tetra
        if length(volumes[i].volumes) == 4
            cell_faces_range[i] = UnitRange(x + 1, x + 4)
            x = x + 4
        end

        #Hexa
        if length(volumes[i].volumes) == 8
            cell_faces_range[i] = UnitRange(x + 1, x + 6)
            x = x + 6
        end
    end
    return cell_faces_range
end

# function generate_all_faces_range(volumes)
#     cell_faces_range=UnitRange(0,0)
#     store=typeof(cell_faces_range)[]
#     x=0
#     @inbounds for i=1:length(volumes)
#         #Tetra
#         if length(volumes[i].volumes)==4
#             cell_faces_range=UnitRange(x+1,x+4)
#             x=x+4
#             push!(store,cell_faces_range)
#         end

#         #Hexa
#         if length(volumes[i].volumes)==8
#                 cell_faces_range=UnitRange(x+1,x+6)
#                 x=x+6
#                 push!(store,cell_faces_range)
#         end
#     end
#     return store
# end

#Generate cells
function generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range)
    cells = Vector{Cell{Float64,SVector{3,Float64},UnitRange{Int64}}}(undef,length(volumes))
    for i = eachindex(volumes)
        cells[i] = Cell(cells_centre[i], cells_volume[i], cell_nodes_range[i], cell_faces_range[i])
    end
    return cells
end

function generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight)
    faces3D = Vector{Face3D{Float64,SVector{2,Int64},SVector{3,Float64},UnitRange{Int64}}}(undef,length(faces))
    for i = eachindex(faces)
        faces3D[i] = Face3D(face_nodes_range[i], SVector(face_ownerCells[i, 1], face_ownerCells[i, 2]), faces_centre[i], faces_normal[i], faces_e[i], faces_area[i], faces_delta[i], faces_weight[i])
    end
    return faces3D
end