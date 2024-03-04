using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

mesh.nodes[end].cells_range
mesh.node_cells

mesh.boundaries[end].IDs_range
mesh.boundary_cellsID

mesh.cells[end].nodes_range
mesh.cell_nodes

mesh.cells[end].faces_range
mesh.cell_faces
mesh.cell_neighbours
mesh.cell_nsign

mesh.faces[end].nodes_range
mesh.face_nodes

mesh.nodes
mesh.faces
mesh.cells
mesh.boundaries

mesh.faces[16].ownerCells

function generate_internal_faces(volumes,faces)
    store_cell_faces=Int[]
    store_faces=Int[]
    
    for i=1:length(volumes)
    cell=sort(volumes[i].volumes)
    push!(store_cell_faces,cell[1],cell[2],cell[3])
    push!(store_cell_faces,cell[1],cell[2],cell[4])
    push!(store_cell_faces,cell[1],cell[3],cell[4])
    push!(store_cell_faces,cell[2],cell[3],cell[4])
    end
    
    for i=1:length(faces)
        face=sort(faces[i].faces)
        push!(store_faces,face[1],face[2],face[3])
    end

    range=[]

    x=0
    for i=1:length(store_cell_faces)/3
        store=UnitRange(x+1:x+3)
        x=x+3
        push!(range,store)
    end

    faces1=[]

    for i=1:length(range)
        face1=store_cell_faces[range[i]]
        for ic=1:length(range)
            face2=store_cell_faces[range[ic]]
            store=[]
    
            push!(store,face2[1] in face1)
            push!(store,face2[2] in face1)
            push!(store,face2[3] in face1)
    
            count1=count(store)
    
            if count1!=3
                push!(faces1,face1)
            end
        end
    end

    all_faces=unique(faces1)

    store1_faces=[]
    for i=1:length(faces)
        push!(store1_faces,sort(faces[i].faces))
    end

    all_faces=sort(all_faces)
    store1_faces=sort(store1_faces)
    
    internal_faces=setdiff(all_faces,store1_faces)
    
    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end

faces=generate_internal_faces(volumes,faces)

function generate_all_cell_faces(volumes,faces)
    cell_faces=[]
    for i=1:length(volumes)
        for ic=1:length(faces)
            bad=sort(volumes[i].volumes)
            good=sort(faces[ic].faces)
            store=[]

            push!(store,good[1] in bad)
            push!(store,good[2] in bad)
            push!(store,good[3] in bad)

            if store[1:3] == [true,true,true]
                push!(cell_faces,faces[ic].faceindex)
            end
            continue
        end
    end
    return cell_faces
end

all_cell_faces=generate_all_cell_faces(volumes,faces)

function generate_all_faces_range(volumes,faces)
    cell_faces_range=UnitRange(0,0)
    store=[]
    x=0
    @inbounds for i=1:length(volumes)
        #Tetra
        if length(volumes[i].volumes)==4
            #cell_faces_range=UnitRange(faces[(4*i)-3].faceindex,faces[4*i].faceindex)
            cell_faces_range=UnitRange(x+1,x+length(volumes[i].volumes))
            x=x+length(volumes[i].volumes)
            push!(store,cell_faces_range)
        end

        #Hexa
        if length(volumes[i].volumes)==8
                cell_faces_range=UnitRange(faces[6*i-5].faceindex,faces[6*i].faceindex)
                push!(store,cell_faces_range)
        end

        #wedge
        if length(volumes[i].volumes)==6
                cell_faces_range=UnitRange(faces[5*i-4].faceindex,faces[5*i].faceindex)
                push!(store,cell_faces_range)
        end
    end
    return store
end

all_cell_faces_range=generate_all_faces_range(volumes,faces)

function generate_boundary_faces(boundaryElements)
    boundary_faces=[]
    z=0
    wipe=Int64[]
    boundary_face_range=[]
    for i=1:length(boundaryElements)
        for n=1:length(boundaryElements[i].elements)
            push!(boundary_faces,boundaryElements[i].elements[n])
            push!(wipe,boundaryElements[i].elements[n])
        end
        if length(wipe)==1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[1]))
            z=z+1
        elseif length(wipe) ≠1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[end]))
            z=z+length(wipe)
        end
        wipe=Int64[]
    end
    return boundary_faces,boundary_face_range
end

boundary_faces,boundary_face_range=generate_boundary_faces(boundaryElements)

#function generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)
    boundary_cells=Int[]
    store=[]
    for ic=1:length(boundary_faces)
        for i in eachindex(all_cell_faces)
                if all_cell_faces[i]==boundary_faces[ic]
                    push!(store,i)
                end
        end
    end
    store
    
    for ic=1:length(store)
        for i=1:length(all_cell_faces_range)
            if all_cell_faces_range[i][1]<=store[ic]<=all_cell_faces_range[i][end]
                push!(boundary_cells,i)
            end
        end
    end
    return boundary_cells
#end

generate_boundary_cells(boundary_faces,all_cell_faces,all_cell_faces_range)