using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

mesh.cell_faces



struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
    coords::SV3
    cells_range::UR # to access neighbour cells (can be dummy entry for now)
end

function generate_nodes(points,volumes)
    nodes=Node[]
    cells_range=nodes_cells_range!(points,volumes)
    @inbounds for i ∈ 1:length(points)
        #point=points[i].xyz
        push!(nodes,Node(points[i].xyz,cells_range[i]))
    end
    return nodes
end

function nodes_cells_range!(points,volumes)
    neighbour=Int64[]
    wipe=Int64[]
    cells_range=UnitRange[]
    x=0
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(wipe,neighbour)
                    
                end
                continue
                
            end
        end
        if length(wipe)==1
            #cells_range[in]=UnitRange(x+1,x+1)
            push!(cells_range,UnitRange(x+1,x+1))
            x=x+1
        elseif length(wipe) ≠1
            #cells_range[in]=UnitRange(x+1,x+length(wipe))
            push!(cells_range,UnitRange(x+1,x+length(wipe)))
            x=x+length(wipe)
        end
        #push!(mesh.nodes[in].cells_range,cells_range)
        wipe=Int64[]
    end
    return cells_range
end

nodes=generate_nodes(points,volumes)

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

# function generate_all_cell_faces(volumes,faces)
#     cell_faces=[]
#     for i=1:length(volumes)
#         for ic=1:length(faces)
#             bad=sort(volumes[i].volumes)
#             good=sort(faces[ic].faces)
#             store=[]

#             push!(store,good[1] in bad)
#             push!(store,good[2] in bad)
#             push!(store,good[3] in bad)

#             if store[1:3] == [true,true,true]
#                 push!(cell_faces,faces[ic].faceindex)
#             end
#             continue
#         end
#     end
#     return cell_faces
# end

# generate_all_cell_faces(volumes,faces)

function generate_all_cell_faces(volumes,faces)
    cell_faces=[]
    for i=1:length(volumes)
        for ic=1:length(faces)
            bad=sort(volumes[i].volumes)
            good=sort(faces[ic].faces)
            store=[]
            true_store=[]

            for ip=1:length(good)
                push!(store,good[ip] in bad)
                push!(true_store,true)
            end

            if store[1:length(good)] == true_store
                push!(cell_faces,faces[ic].faceindex)
            end
            continue
        end
    end
    return cell_faces
end

generate_all_cell_faces(volumes,faces)

function generate_cell_faces(volumes,faces,boundaryElements)
    cell_faces=Int[]
    cell_faces_range=[]
    max_store=0
    max=0
    for ib=1:length(boundaryElements)
        max_store=maximum(boundaryElements[ib].elements)
        if max_store>=max
            max=max_store
        end
    end
    
    x=0
    for i=1:length(volumes)
    wipe=[]
        for ic=max+1:length(faces)
            bad=sort(volumes[i].volumes)
            good=sort(faces[ic].faces)
            store=[]
            true_store=[]

            for ip=1:length(good)
                push!(store,good[ip] in bad)
                push!(true_store,true)
            end

            if store[1:length(good)] == true_store
                push!(cell_faces,faces[ic].faceindex)
                push!(wipe,faces[ic].faceindex)
            end
            continue
        end

        if length(wipe)==1
            push!(cell_faces_range,UnitRange(x+1,x+1))
            x=x+1
        end

        if length(wipe) ≠ 1
            push!(cell_faces_range,UnitRange(x+1,x+length(wipe)))
            x=x+length(wipe)
        end
    end
    return cell_faces,cell_faces_range
end

cell_faces,cell_faces_range=generate_cell_faces(volumes,faces,boundaryElements)
cell_faces
cell_faces_range

function generate_all_faces_range(volumes)
    cell_faces_range=UnitRange(0,0)
    store=[]
    x=0
    @inbounds for i=1:length(volumes)
        #Tetra
        if length(volumes[i].volumes)==4
            cell_faces_range=UnitRange(x+1,x+4)
            x=x+4
            push!(store,cell_faces_range)
        end

        #Hexa
        if length(volumes[i].volumes)==8
                cell_faces_range=UnitRange(x+1,x+6)
                x=x+6
                push!(store,cell_faces_range)
        end
    end
    return store
end

generate_all_faces_range(volumes)