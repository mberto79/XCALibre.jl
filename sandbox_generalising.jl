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

mesh.faces[16].centre

#mesh=build_mesh3D(unv_mesh)

# area_store=[]
# for i=1:length(faces)
#     if faces[i].faceCount==3
#         n1=nodes[faces[i].faces[1]].coords
#         n2=nodes[faces[i].faces[2]].coords
#         n3=nodes[faces[i].faces[3]].coords

#         t1x=n2[1]-n1[1]
#         t1y=n2[2]-n1[2]
#         t1z=n2[3]-n1[3]

#         t2x=n3[1]-n1[1]
#         t2y=n3[2]-n1[2]
#         t2z=n3[3]-n1[3]

#         area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#         area=sqrt(area2)/2
#         push!(area_store,area)
#     end

#     if faces[i].faceCount>3
#         n1=nodes[faces[i].faces[1]].coords
#         n2=nodes[faces[i].faces[2]].coords
#         n3=nodes[faces[i].faces[3]].coords

#         t1x=n2[1]-n1[1]
#         t1y=n2[2]-n1[2]
#         t1z=n2[3]-n1[3]

#         t2x=n3[1]-n1[1]
#         t2y=n3[2]-n1[2]
#         t2z=n3[3]-n1[3]

#         area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#         area=sqrt(area2)/2

#         for ic=4:faces[i].faceCount
#             n1=nodes[faces[i].faces[ic]].coords
#             n2=nodes[faces[i].faces[2]].coords
#             n3=nodes[faces[i].faces[3]].coords

#             t1x=n2[1]-n1[1]
#             t1y=n2[2]-n1[2]
#             t1z=n2[3]-n1[3]

#             t2x=n3[1]-n1[1]
#             t2y=n3[2]-n1[2]
#             t2z=n3[3]-n1[3]

#             area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
#             area=area+sqrt(area2)/2

#         end

#         push!(area_store,area)

#     end
# end
# area_store

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

function calculate_face_area(nodes,faces)
    area_store=Float64[]
    for i=1:length(faces)
        if faces[i].faceCount==3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            push!(area_store,area)
        end

        if faces[i].faceCount>3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:faces[i].faceCount
                n1=nodes[faces[i].faces[ic]].coords
                n2=nodes[faces[i].faces[2]].coords
                n3=nodes[faces[i].faces[3]].coords

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end

            push!(area_store,area)

        end
    end
    return area_store
end

face_area=calculate_face_area(nodes,faces)


function calculate_face_centre(faces,nodes)
    centre_store=[]
    for i=1:length(faces)
        face_store=[]
        for ic=1:length(faces[i].faces)
            push!(face_store,nodes[faces[i].faces[ic]].coords)
        end
        centre=(sum(face_store)/length(face_store))
        push!(centre_store,centre)
    end
    return centre_store
end

face_centre=calculate_face_centre(faces,nodes)

# normal_store=[]
# for i=1:length(faces)
#     n1=nodes[faces[i].faces[1]].coords
#     n2=nodes[faces[i].faces[2]].coords
#     n3=nodes[faces[i].faces[3]].coords

#     t1x=n2[1]-n1[1]
#     t1y=n2[2]-n1[2]
#     t1z=n2[3]-n1[3]

#     t2x=n3[1]-n1[1]
#     t2y=n3[2]-n1[2]
#     t2z=n3[3]-n1[3]

#     nx=t1y*t2z-t1z*t2y
#     ny=-(t1x*t2z-t1z*t2x)
#     nz=t1x*t2y-t2y*t2x

#     magn2=(nx)^2+(ny)^2+(nz)^2

#     snx=nx/sqrt(magn2)
#     sny=ny/sqrt(magn2)
#     snz=nz/sqrt(magn2)

#     normal=SVector(snx,sny,snz)
#     push!(normal_store,normal)
# end
# normal_store


function calculate_face_normal(faces,nodes)
    normal_store=[]
    for i=1:length(faces)
        n1=nodes[faces[i].faces[1]].coords
        n2=nodes[faces[i].faces[2]].coords
        n3=nodes[faces[i].faces[3]].coords

        t1x=n2[1]-n1[1]
        t1y=n2[2]-n1[2]
        t1z=n2[3]-n1[3]

        t2x=n3[1]-n1[1]
        t2y=n3[2]-n1[2]
        t2z=n3[3]-n1[3]

        nx=t1y*t2z-t1z*t2y
        ny=-(t1x*t2z-t1z*t2x)
        nz=t1x*t2y-t2y*t2x

        magn2=(nx)^2+(ny)^2+(nz)^2

        snx=nx/sqrt(magn2)
        sny=ny/sqrt(magn2)
        snz=nz/sqrt(magn2)

        normal=SVector(snx,sny,snz)
        push!(normal_store,normal)
    end
    return normal_store
end

mesh.faces[16].normal
face_normal=calculate_face_normal(faces,nodes)

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

all_cell_faces=Vector{Int}(generate_all_cell_faces(volumes,faces))


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

volume_store=[]
volume=0
for f=all_cell_faces_range[5]
    findex=all_cell_faces[f]
    volume=volume+(face_normal[findex][1]*face_centre[findex][1]*face_area[findex])
end
volume