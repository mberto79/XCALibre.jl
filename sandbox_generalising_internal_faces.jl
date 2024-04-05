using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

include("src/VTK_3D/VTU.jl")

#unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

mesh.faces
mesh.cells
mesh.cell_neighbours
mesh.cell_faces
mesh.face_nodes
mesh.faces[1].nodes_range
mesh.node_cells
mesh.nodes[4].cells_range

mesh.faces[1].ownerCells
mesh.boundaries
mesh.boundary_cellsID[1]


name="test_vtu"
write_vtu(name,mesh)


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

#function generate_internal_faces(volumes,faces)
    # store_cell_faces=Int[]
    # store_faces=Int[]
    
    # for i=1:length(volumes)
    # cell=sort(volumes[i].volumes)
    # push!(store_cell_faces,cell[1],cell[2],cell[3])
    # push!(store_cell_faces,cell[1],cell[2],cell[4])
    # push!(store_cell_faces,cell[1],cell[3],cell[4])
    # push!(store_cell_faces,cell[2],cell[3],cell[4])
    # end
    # store_cell_faces
    
    # for i=1:length(faces)
    #     face=sort(faces[i].faces)
    #     push!(store_faces,face[1],face[2],face[3])
    # end
    # store_faces

    # range=[]

    # x=0
    # for i=1:length(store_cell_faces)/3
    #     store=UnitRange(x+1:x+3)
    #     x=x+3
    #     push!(range,store)
    # end

    # faces1=[]

    # for i=1:length(range)
    #     face1=store_cell_faces[range[i]]
    #     for ic=1:length(range)
    #         face2=store_cell_faces[range[ic]]
    #         store=[]
    
    #         push!(store,face2[1] in face1)
    #         push!(store,face2[2] in face1)
    #         push!(store,face2[3] in face1)
    
    #         count1=count(store)
    
    #         if count1!=3
    #             push!(faces1,face1)
    #         end
    #     end
    # end

    # all_faces=unique(faces1)

    # store1_faces=[]
    # for i=1:length(faces)
    #     push!(store1_faces,sort(faces[i].faces))
    # end

    # all_faces=sort(all_faces)
    # store1_faces=sort(store1_faces)
    
    # internal_faces=setdiff(all_faces,store1_faces)
    
    # for i=1:length(internal_faces)
    #     push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    # end
    # return faces
#end

# faces=generate_internal_faces(volumes,faces)

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

function calculate_centre_cell(volumes,nodes)
    centre_store=[]
    for i=1:length(volumes)
        cell_store=[]
        for ic=1:length(volumes[i].volumes)
            push!(cell_store,nodes[volumes[i].volumes[ic]].coords)
        end
        centre=(sum(cell_store)/length(cell_store))
        push!(centre_store,centre)
    end
    return centre_store
end

cell_centre=calculate_centre_cell(volumes,nodes)




volumes[1]
p1=nodes[18].coords
p2=nodes[23].coords
p3=nodes[27].coords
p4=nodes[26].coords

v1=p1-p2
v2=p3-p2

normal=abs.(cross(v1,v2))

v3=p4-p2

tol=1e-6
if dot(v3,normal)<tol
    println("On the same plane")
    cc=cell_centre[1]

    fc=(p1+p2+p3+p4)/4

    d_fc=fc-cc

    par=abs.(cross(d_fc,normal))

    par=norm(par)

    if par<tol
        println("Vectors are Parallel")

    end
end

cc=cell_centre[1]

fc=(p1+p2+p3+p4)/4

d_fc=fc-cc

par=abs.(cross(d_fc,normal))

par=norm(par)

if par<tol
    println("Vectors are Parallel")
end

rand((1,2,3))

v1=p1-fc
v2=p4-fc

normal = cross(v1, v2)


if normal[3] > 0
    println("point is counter-clockwise")
end




volumes[1]
p1=nodes[volumes[1].volumes[1]].coords
p2=nodes[volumes[1].volumes[2]].coords
p3=nodes[volumes[1].volumes[3]].coords
p4=nodes[volumes[1].volumes[4]].coords

p5=nodes[volumes[1].volumes[5]].coords
p6=nodes[volumes[1].volumes[6]].coords
p7=nodes[volumes[1].volumes[7]].coords
p8=nodes[volumes[1].volumes[8]].coords

cell_points=[]
cell_faces=zeros(Int,6,4)

cell_faces[1,1:4]=volumes[1].volumes[1:4]
cell_faces[2,1:4]=volumes[1].volumes[5:8]
cell_faces[3,1:2]=volumes[1].volumes[1:2]
cell_faces[3,3:4]=volumes[1].volumes[5:6]
cell_faces[4,1:2]=volumes[1].volumes[3:4]
cell_faces[4,3:4]=volumes[1].volumes[7:8]
cell_faces[5,1:2]=volumes[1].volumes[2:3]
cell_faces[5,3:4]=volumes[1].volumes[6:7]
cell_faces[6,1]=volumes[1].volumes[1]
cell_faces[6,2]=volumes[1].volumes[4]
cell_faces[6,3]=volumes[1].volumes[5]
cell_faces[6,4]=volumes[1].volumes[8]

cell_faces
cell_faces[3,1]

p1=nodes[cell_faces[3,1]].coords
p2=nodes[cell_faces[3,2]].coords
p3=nodes[cell_faces[3,3]].coords
p4=nodes[cell_faces[3,4]].coords


v1=p2-p1
v2=p3-p2

normal=abs.(cross(v1,v2))

tol=1e-6
if dot(v3,normal)<tol
    println("On the same plane")
    cc=cell_centre[1]

    fc=(p1+p2+p3+p4)/4

    d_fc=fc-cc

    par=abs.(cross(d_fc,normal))

    par=norm(par)

    if par<tol
        println("Vectors are Parallel")

    end
end

fc=(p1+p2+p3+p4)/4
cell_centre[1]
dir=fc-cell_centre[1]

winding=dot(cross(v1,v2),dir)
if winding < 0.0
    println("points are counter-clockwise from cell centre")
end

winding=dot(cross(p3-p2,p4-p3),dir)
if winding < 0.0
    println("points are counter-clockwise from cell centre")
end



cell_points=[]
store_cell_faces1=[]

for i=1:length(volumes)
    cell_faces=zeros(Int,6,4)

    cell_faces[1,1:4]=volumes[i].volumes[1:4]
    cell_faces[2,1:4]=volumes[i].volumes[5:8]
    cell_faces[3,1:2]=volumes[i].volumes[1:2]
    cell_faces[3,3:4]=volumes[i].volumes[5:6]
    cell_faces[4,1:2]=volumes[i].volumes[3:4]
    cell_faces[4,3:4]=volumes[i].volumes[7:8]
    cell_faces[5,1:2]=volumes[i].volumes[2:3]
    cell_faces[5,3:4]=volumes[i].volumes[6:7]
    cell_faces[6,1]=volumes[i].volumes[1]
    cell_faces[6,2]=volumes[i].volumes[4]
    cell_faces[6,3]=volumes[i].volumes[5]
    cell_faces[6,4]=volumes[i].volumes[8]

    for ic=1:6
        push!(store_cell_faces1,cell_faces[ic,:])
    end
end

store_cell_faces1

sorted_cell_faces=[]
for i=1:length(store_cell_faces1)

    push!(sorted_cell_faces,sort(store_cell_faces1[i]))
end

sorted_cell_faces

faces
sorted_faces=[]
for i=1:length(faces)
    push!(sorted_faces,sort(faces[i].faces))
end
sorted_faces

internal_faces=setdiff(sorted_cell_faces,sorted_faces)

for i=1:length(internal_faces)
    push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
end
faces

function quad_internal_faces(volumes,faces)
    store_cell_faces1=[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,6,4)

        cell_faces[1,1:4]=volumes[i].volumes[1:4]
        cell_faces[2,1:4]=volumes[i].volumes[5:8]
        cell_faces[3,1:2]=volumes[i].volumes[1:2]
        cell_faces[3,3:4]=volumes[i].volumes[5:6]
        cell_faces[4,1:2]=volumes[i].volumes[3:4]
        cell_faces[4,3:4]=volumes[i].volumes[7:8]
        cell_faces[5,1:2]=volumes[i].volumes[2:3]
        cell_faces[5,3:4]=volumes[i].volumes[6:7]
        cell_faces[6,1]=volumes[i].volumes[1]
        cell_faces[6,2]=volumes[i].volumes[4]
        cell_faces[6,3]=volumes[i].volumes[5]
        cell_faces[6,4]=volumes[i].volumes[8]

        for ic=1:6
            push!(store_cell_faces1,cell_faces[ic,:])
        end
    end

    store_cell_faces1

    sorted_cell_faces=[]
    for i=1:length(store_cell_faces1)

        push!(sorted_cell_faces,sort(store_cell_faces1[i]))
    end

    sorted_cell_faces

    faces
    sorted_faces=[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end
    sorted_faces

    internal_faces=setdiff(sorted_cell_faces,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end

quad_internal_faces(volumes,faces)