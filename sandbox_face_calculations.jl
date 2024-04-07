using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK/VTK_writer_3D.jl")


unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
unv_mesh="src/UNV_3D/TET_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
mesh.cells
mesh.boundaries


mesh.nodes
x=[]
y=[]
zp=[]
for i=1:length(mesh.nodes)
    push!(x,mesh.nodes[i].coords[1])
    push!(y,mesh.nodes[i].coords[2])
    push!(zp,mesh.nodes[i].coords[3])
end
x
maximum(x)
minimum(x)
rangex=
y
maximum(y)
minimum(y)
zp
maximum(y)
minimum(y)

vol=[]
for i=1:length(mesh.cells)
    push!(vol,mesh.cells[i].volume)
end

vol
sum(vol,init=0.0)

name="tet_prism"

write_vtk(name, mesh::Mesh3)

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa


iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.

iface_nodes_range .= [
    iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
    ]

# Concatenate boundary and internal faces
face_nodes = vcat(bface_nodes, iface_nodes)
face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

# Sort out cell to face connectivity
cell_faces, cell_nsign, cell_faces_range, cell_neighbours = FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.

# Build mesh (without calculation of geometry/properties)
cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range) # Hybrid compatible. Hexa tested.
faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells) # Hybrid compatible. Hexa tested.

mesh = Mesh3(
cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, 
faces, face_nodes, boundaries, 
nodes, node_cells,
SVector{3, Float64}(0.0, 0.0, 0.0), UnitRange{Int64}(0, 0), boundary_cellsID
) # Hexa tested.

# Update mesh to include all geometry calculations required
FVM_1D.UNV_3D.calculate_centres!(mesh) # Uses centroid instead of geometric. Will need changing, should work fine for regular cells and faces
FVM_1D.UNV_3D.calculate_face_properties!(mesh) # Touched up face properties, double check values.
FVM_1D.UNV_3D.calculate_area_and_volume!(mesh)



all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells)]
for fID ∈ eachindex(faces)
    owners = faces[fID].ownerCells
    owner1 = owners[1]
    owner2 = owners[2]
    #if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
        push!(all_cell_faces[owner1],fID)
        if owner1 !== owner2 #avoid duplication of cells for boundary faces
            push!(all_cell_faces[owner2],fID)
        end
    #end
end

calculate_area_and_volume!(mesh) = begin
    (; nodes, faces, face_nodes, cells, cell_nodes) = mesh

    n_faces=length(faces)
    n_cells=length(cells)

    #Using old method for now, know it works. Following method outlined by Sandip.

    for fID ∈ 1:n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        if length(face.nodes_range) == 3 # For Triangles
            #nIDs = nodeIDs(face_nodes, face.nodes_range)
            n1 = nodes[nIDs[1]].coords
            n2 = nodes[nIDs[2]].coords
            n3 = nodes[nIDs[3]].coords
            
            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            
            @reset face.area = area

            faces[fID] = face
        end

        if length(face.nodes_range) >= 4 # For any shape
            n1 = nodes[nIDs[1]].coords
            n2 = nodes[nIDs[2]].coords
            n3 = nodes[nIDs[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:length(face.nodes_range)
                n1 = nodes[nIDs[ic]].coords
                n2 = nodes[nIDs[2]].coords
                n3 = nodes[nIDs[3]].coords

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end
            @reset face.area = area
            faces[fID] = face
        end
    end

    all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells)]
    for fID ∈ eachindex(faces)
        owners = faces[fID].ownerCells
        owner1 = owners[1]
        owner2 = owners[2]
        #if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
            push!(all_cell_faces[owner1],fID)
            if owner1 !== owner2 #avoid duplication of cells for boundary faces
                push!(all_cell_faces[owner2],fID)
            end
        #end
    end

    for cID ∈ eachindex(cells)
        cell = cells[cID]
        nface = length(all_cell_faces[cID])
        volume = 0
        cc = cell[cID].centre

        for f=1:nface
            ifc=all_cell_faces[cID][f]
            face = faces[ifc]

            normal=face[ifc].normal
            fc=face[ifc].centre
            d_fc=fc-cc

            if  face[ifc,1].ownerCells ≠ face[ifc,2].ownerCells
                if dot(d_fc,normal)<0.0
                    normal=-1.0*normal
                end
            end

            volume=volume+(normal[1]*fc[1]*face[ifc].area)

        end
        @reset cell.volume = volume
        cells[cID] = cell
    end
end




function calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,face_normal,cell_centre,face_centre,face_ownerCells,face_area)
    volume_store=[]
    for i=1:length(volumes)
        volume=0
        for f=all_cell_faces_range[i]
            findex=all_cell_faces[f]

            normal=face_normal[findex]
            cc=cell_centre[i]
            fc=face_centre[findex]
            d_fc=fc-cc

            if  face_ownerCells[findex,1] ≠ face_ownerCells[findex,2]
                if dot(d_fc,normal)<0.0
                    normal=-1.0*normal
                end
            end


            volume=volume+(normal[1]*face_centre[findex][1]*face_area[findex])
            
        end
        push!(volume_store,volume)
    end
    return volume_store
end





