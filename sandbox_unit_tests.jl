using FVM_1D
using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.08m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.04m.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"


points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

#mesh,cell_face_nodes, node_cells, all_cell_faces,boundary_cells,boundary_faces,all_cell_faces_range=build_mesh3D(unv_mesh)
# mesh.boundaries
# mesh.faces

@time faces_checked, results = check_face_owners(mesh)
@time check_cell_face_nodes(mesh,cell_face_nodes)
@time boundary_faces(mesh)
@time check_node_cells(mesh,node_cells)
@time check_all_cell_faces(mesh,all_cell_faces)
@time check_boundary_faces(boundary_cells,boundary_faces,all_cell_faces,all_cell_faces_range)

@time nodes=generate_nodes(points,volumes)
@time faces=generate_tet_internal_faces(volumes,faces)

function nodes_cells_range!(points,volumes)
    neighbour=Int64[]
    wipe=Int64[]
    cells_range=UnitRange{Int64}[]
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

function generate_nodes(points,volumes)
    # nodes=Node{SVector{3,Float64}, UnitRange{Int64}}[]
    nnodes = length(points)
    nodes = [Node(SVector{3,Float64}(0.0,0.0,0.0), 1:1) for i ∈ 1:nnodes]
    tnode = Node(SVector{3,Float64}(0.0,0.0,0.0), 1:1) # temporary node object used to rewrite
    cells_range=nodes_cells_range!(points,volumes)
    @inbounds for i ∈ 1:length(points)
        #point=points[i].xyz
        # push!(nodes,Node(points[i].xyz,cells_range[i]))
        tnode = @reset tnode.coords = points[i].xyz
        tnode = @reset tnode.cells_range = cells_range[i]
        nodes[i] = tnode # overwrite preallocated array with temporary node
    end
    return nodes
end

function calculate_face_normal(faces,nodes)
    normal_store=SVector{3,Float64}[]
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
        nz=t1x*t2y-t1y*t2x

        magn2=(nx)^2+(ny)^2+(nz)^2

        snx=nx/sqrt(magn2)
        sny=ny/sqrt(magn2)
        snz=nz/sqrt(magn2)

        normal=SVector(snx,sny,snz)
        push!(normal_store,normal)
    end
    return normal_store
end

function generate_tet_internal_faces(volumes,faces)
    cell_face_nodes=Vector{Int}[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,4,3)
        cell=sort(volumes[i].volumes)

        cell_faces[1,1:3]=cell[1:3]
        cell_faces[2,1:2]=cell[1:2]
        cell_faces[2,3]=cell[4]
        cell_faces[3,1]=cell[1]
        cell_faces[3,2:3]=cell[3:4]
        cell_faces[4,1:3]=cell[2:4]

        for ic=1:4
            push!(cell_face_nodes,cell_faces[ic,:])
        end
    end

    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    internal_faces=setdiff(cell_face_nodes,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces, cell_face_nodes
end

function calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    store_e=SVector{3,Float64}[]
    store_delta=Float64[]
    store_weight=Float64[]
    for i=1:length(faces) #Boundary Face
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            delta=norm(d_cf)
            push!(store_delta,delta)
            e=d_cf/delta
            push!(store_e,e)
            weight=one(Float64)
            push!(store_weight,weight)

        else #Internal Face
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            cf=face_centre[i]
            d_1f=cf-c1
            d_f2=c2-cf
            d_12=c2-c1

            delta=norm(d_12)
            push!(store_delta,delta)
            e=d_12/delta
            push!(store_e,e)
            weight=abs((d_1f⋅face_normal[i])/(d_1f⋅face_normal[i] + d_f2⋅face_normal[i]))
            push!(store_weight,weight)

        end
    end
    return store_e,store_delta,store_weight
end

function flip_face_normal(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    for i=1:length(faces)
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            if d_cf⋅face_normal[i]<0
                face_normal[i]=-1.0*face_normal[i]
            end
        else
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            cf=face_centre[i]
            #d_1f=cf-c1
            #d_f2=c2-cf
            d_12=c2-c1

            if d_12⋅face_normal[i]<0
                face_normal[i]=-1.0*face_normal[i]
            end
        end
    end
    return face_normal
end