using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

#unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"

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

#function calculate_faces_properties(faces,face_nodes,nodes,cell_nodes,cells,face_ownerCells,face_nodes_range)
    store_normal=[]
    store_area=[]
    store_centre=[]
    store_e=[]
    store_delta=[]
    store_weight=[]
    for i=1:length(faces)
        p1=nodes[face_nodes[face_nodes_range[i]][1]].coords
        p2=nodes[face_nodes[face_nodes_range[i]][2]].coords
        p3=nodes[face_nodes[face_nodes_range[i]][3]].coords

        e1 = p2 - p1
        e2 = p3 - p1
        
        normal = cross(e1, e2)
        normal /= norm(normal)
        
        area = 0.5 * norm(cross(e1, e2))
        
        cf=(p1+p2+p3)/3
        
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            p1=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][1]].coords
            p2=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][2]].coords
            p3=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][3]].coords
            p4=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][4]].coords

            cc=(p1+p2+p3+p4)/4

            d_cf=cf-cc

            if d_cf⋅normal<0
                normal=-1.0*normal
            end
            normal

            delta=norm(d_cf)
            push!(store_delta,delta)
            e=d_cf/delta
            push!(store_e,e)
            weight=one(Float64)
            push!(store_weight,weight)

        else
            p1=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][1]].coords
            p2=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][2]].coords
            p3=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][3]].coords
            p4=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][4]].coords
            
            o1=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][1]].coords
            o2=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][2]].coords
            o3=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][3]].coords
            o4=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][4]].coords
            
            c1=(p1+p2+p3+p4)/4
            c2=(o1+o2+o3+o4)/4
            d_1f=cf-c1
            d_f2=c2-cf
            d_12=c2-c1
            
            if d_12⋅normal<0
                normal=-1.0*normal
            end
            
            delta=norm(d_12)
            push!(store_delta,delta)
            e=d_12/delta
            push!(store_e,e)
            weight=abs((d_1f⋅normal)/(d_1f⋅normal + d_f2⋅normal))
            push!(store_weight,weight)
        end
        push!(store_normal,normal)
        push!(store_area,area)
        push!(store_centre,cf)
    end
    return store_normal,store_area,store_centre,store_e,store_delta,store_weight
#end

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

function generate_cell_faces(volumes,faces)
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

            push!(store,good[1] in bad)
            push!(store,good[2] in bad)
            push!(store,good[3] in bad)

            if store[1:3] == [true,true,true]
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

cell_faces,cell_faces_range=generate_cell_faces(volumes,faces)

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

function generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)
    boundary_cells=Int[]
    store=[]
    for ic=1:length(boundary_faces)
        for i in eachindex(cell_faces)
                if cell_faces[i]==boundary_faces[ic]
                    push!(store,i)
                end
        end
    end
    store
    
    for ic=1:length(store)
        for i=1:length(cell_faces_range)
            if cell_faces_range[i][1]<=store[ic]<=cell_faces_range[i][end]
                push!(boundary_cells,i)
            end
        end
    end
    return boundary_cells
end

boundary_cells=generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)

#function generate_cell_neighbours(cells,cell_faces)
    cell_neighbours=Int[]
    for ID=1:length(cells) # 1:5
        for i=cells[ID].faces_range # 1:4, 5:8, etc
            faces=cell_faces[i]
            for ic=1:length(i) #1:4
                face=faces[ic]
                index=findall(x->x==face,cell_faces)
                if length(index)==2
                    if i[1]<=index[1]<=i[end]
                        for ip=1:length(cells)
                            if cells[ip].faces_range[1]<=index[2]<=cells[ip].faces_range[end]
                                push!(cell_neighbours,ip)
                            end
                        end
                    end
                    if i[1]<=index[2]<=i[end]
                        for ip=1:length(cells)
                            if cells[ip].faces_range[1]<=index[1]<=cells[ip].faces_range[end]
                                push!(cell_neighbours,ip)
                            end
                        end
                    end
                end
                if length(index)==1
                    x=0
                    push!(cell_neighbours,x)
                end
            end
        end
    end
    return cell_neighbours
#end


