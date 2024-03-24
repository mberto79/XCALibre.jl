export build_mesh3D

function build_mesh3D(unv_mesh; integer=Int64, float=Float64)
    stats= @timed begin
    println("Loading UNV File...")
    points,edges,faces,volumes,boundaryElements=load_3D(
        unv_mesh; integer=integer, float=float)
    println("File Read Successfully")
    println("Generating Mesh...")
    nodes=generate_nodes(points,volumes)

    node_cells=generate_node_cells(points,volumes)

    #faces=generate_internal_faces(volumes,faces)

    faces,cell_face_nodes=generate_tet_internal_faces(volumes,faces)
    #faces=quad_internal_faces(volumes,faces)

    face_nodes=Vector{Int}(generate_face_nodes(faces))
    cell_nodes=Vector{Int}(generate_cell_nodes(volumes))
    
    all_cell_faces=generate_all_cell_faces(faces,cell_face_nodes)

    cell_nodes_range=generate_cell_nodes_range(volumes)
    face_nodes_range=generate_face_nodes_range(faces)
    all_cell_faces_range=generate_all_faces_range(volumes)

    centre_of_cells=calculate_centre_cell(volumes,nodes)
    #volume_of_cells=calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,face_normal,cell_centre,face_centre,face_ownerCells,face_area)

    boundary_faces,boundary_face_range=generate_boundary_faces(boundaryElements)
    boundary_cells=generate_boundary_cells(boundary_faces,all_cell_faces,all_cell_faces_range)

    cell_faces,cell_faces_range=generate_cell_faces(boundaryElements,volumes,all_cell_faces)

    boundaries=generate_boundaries(boundaryElements,boundary_face_range)
    #cells=generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)

    #cell_neighbours=generate_cell_neighbours(cells,cell_faces)

    face_ownerCells=generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)

    #faces_normal,faces_area,faces_centre,faces_e,faces_delta,faces_weight=calculate_faces_properties(faces,face_nodes,nodes,cell_nodes,cells,face_ownerCells,face_nodes_range)

    faces_area=calculate_face_area(nodes,faces)
    faces_centre=calculate_face_centre(faces,nodes)
    faces_normal=calculate_face_normal(faces,nodes)
    faces_normal=flip_face_normal(faces,face_ownerCells,centre_of_cells,faces_centre,faces_normal)
    faces_e,faces_delta,faces_weight=calculate_face_properties(faces,face_ownerCells,centre_of_cells,faces_centre,faces_normal)
    
    volume_of_cells=calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,faces_normal,centre_of_cells,faces_centre,face_ownerCells,faces_area)

    cells=generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)
    cell_neighbours=generate_cell_neighbours(cells,cell_faces)
    faces=generate_faces(faces,face_nodes_range,faces_centre,faces_normal,faces_area,face_ownerCells,faces_e,faces_delta,faces_weight)
    
    cell_nsign=calculate_cell_nsign(cells,faces,cell_faces)

    get_float=SVector(0.0,0.0,0.0)
    get_int=UnitRange(0,0)

    mesh=Mesh3(cells,cell_nodes,cell_faces,cell_neighbours,cell_nsign,faces,face_nodes,boundaries,nodes,node_cells,get_float,get_int,boundary_cells)

    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    #return mesh

    #For unit testing
    return mesh,cell_face_nodes, node_cells, all_cell_faces
end

# DEFINE FUNCTIONS
function calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    store_e=SVector{3,Float64}[]
    store_delta=Float64[]
    store_weight=Float64[]
    for i=1:length(faces)
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

        else
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

function calculate_face_centre(faces,nodes)
    centre_store=SVector{3,Float64}[]
    for i=1:length(faces)
        face_store=SVector{3,Float64}[]
        for ic=1:length(faces[i].faces)
            push!(face_store,nodes[faces[i].faces[ic]].coords)
        end
        centre=(sum(face_store)/length(face_store))
        push!(centre_store,centre)
    end
    return centre_store
end

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

# function generate_cell_faces(volumes,faces,boundaryElements)
#     cell_faces=Int[]
#     cell_faces_range=[]
#     max_store=0
#     max=0
#     for ib=1:length(boundaryElements)
#         max_store=maximum(boundaryElements[ib].elements)
#         if max_store>=max
#             max=max_store
#         end
#     end
    
#     x=0
#     for i=1:length(volumes)
#     wipe=[]
#         for ic=max+1:length(faces)
#             bad=sort(volumes[i].volumes)
#             good=sort(faces[ic].faces)
#             store=[]

#             push!(store,good[1] in bad)
#             push!(store,good[2] in bad)
#             push!(store,good[3] in bad)

#             if store[1:3] == [true,true,true]
#                 push!(cell_faces,faces[ic].faceindex)
#                 push!(wipe,faces[ic].faceindex)
#             end
#             continue
#         end

#         if length(wipe)==1
#             push!(cell_faces_range,UnitRange(x+1,x+1))
#             x=x+1
#         end

#         if length(wipe) ≠ 1
#             push!(cell_faces_range,UnitRange(x+1,x+length(wipe)))
#             x=x+length(wipe)
#         end
#     end
#     return cell_faces,cell_faces_range
# end

function generate_cell_faces(boundaryElements,volumes,all_cell_faces)
    cell_faces=Vector{Int}[]
    cell_face_range=UnitRange{Int64}[]
    counter_start=0
    x=0
    max=0

    for ib=1:length(boundaryElements)
        max_store=maximum(boundaryElements[ib].elements)
        if max_store>=max
            max=max_store
        end
    end

    for i=1:length(volumes)
        push!(cell_faces,all_cell_faces[counter_start+1:counter_start+length(volumes[i].volumes)])
        counter_start=counter_start+length(volumes[i].volumes)
        cell_faces[i]=filter(x-> x>max,cell_faces[i])

        if length(cell_faces[i])==1
            push!(cell_face_range,UnitRange(x+1,x+1))
            x=x+1
        else
            push!(cell_face_range,UnitRange(x+1,x+length(cell_faces[i])))
            x=x+length(cell_faces[i])
        end
    end
    cell_faces=reduce(vcat,cell_faces)

    return cell_faces,cell_face_range
end

function calculate_cell_nsign(cells,faces1,cell_faces)
    cell_nsign=Int[]
    for i=1:length(cells)
        centre=cells[i].centre 
        for ic=1:length(cells[i].faces_range)
            fcentre=faces1[cell_faces[cells[i].faces_range][ic]].centre
            fnormal=faces1[cell_faces[cells[i].faces_range][ic]].normal
            d_cf=fcentre-centre
            fnsign=zero(Int)

            if d_cf⋅fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            push!(cell_nsign,fnsign)
        end

    end
    return cell_nsign
end

# function calculate_cell_volume(volumes,nodes)
#     volume_store=Float64[]
#     for i=1:length(volumes)
#         A=nodes[volumes[i].volumes[1]].coords
#         B=nodes[volumes[i].volumes[2]].coords
#         C=nodes[volumes[i].volumes[3]].coords
#         D=nodes[volumes[i].volumes[4]].coords

#         AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
#         AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]
#         AD=[D[1]-A[1],D[2]-A[2],D[3]-A[3]]
#         volume=abs(dot(AB, cross(AC, AD)))/6
#         push!(volume_store,volume)
#     end
#     return volume_store
# end

function calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,face_normal,cell_centre,face_centre,face_ownerCells,face_area)
    volume_store = Float64[]
    for i=1:length(volumes)
        volume = zero(Float64) # to avoid type instability
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

# function calculate_centre_cell(volumes,nodes)
#     centre_store=[]
#     for i=1:length(volumes)
#         A=nodes[volumes[i].volumes[1]].coords
#         B=nodes[volumes[i].volumes[2]].coords
#         C=nodes[volumes[i].volumes[3]].coords
#         D=nodes[volumes[i].volumes[4]].coords
#         centre=((A+B+C+D)/4)
#         push!(centre_store,centre)
#     end
#     return centre_store
# end

function calculate_centre_cell(volumes,nodes)
    centre_store=SVector{3,Float64}[]
    for i=1:length(volumes)
        cell_store=typeof(nodes[volumes[1].volumes[1]].coords)[]
        for ic=1:length(volumes[i].volumes)
            push!(cell_store,nodes[volumes[i].volumes[ic]].coords)
        end
        centre=(sum(cell_store)/length(cell_store))
        push!(centre_store,centre)
    end
    return centre_store
end

# function calculate_faces_properties(faces,face_nodes,nodes,cell_nodes,cells,face_ownerCells,face_nodes_range)
#     store_normal=[]
#     store_area=[]
#     store_centre=[]
#     store_e=[]
#     store_delta=[]
#     store_weight=[]
#     for i=1:length(faces)
#         p1=nodes[face_nodes[face_nodes_range[i]][1]].coords
#         p2=nodes[face_nodes[face_nodes_range[i]][2]].coords
#         p3=nodes[face_nodes[face_nodes_range[i]][3]].coords

#         e1 = p2 - p1
#         e2 = p3 - p1
        
#         normal = cross(e1, e2)
#         normal /= norm(normal)
        
#         area = 0.5 * norm(cross(e1, e2))
        
#         cf=(p1+p2+p3)/3
        
#         if face_ownerCells[i,2]==face_ownerCells[i,1]
#             p1=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][1]].coords
#             p2=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][2]].coords
#             p3=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][3]].coords
#             p4=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][4]].coords

#             cc=(p1+p2+p3+p4)/4

#             d_cf=cf-cc

#             if d_cf⋅normal<0
#                 normal=-1.0*normal
#             end
#             normal

#             delta=norm(d_cf)
#             push!(store_delta,delta)
#             e=d_cf/delta
#             push!(store_e,e)
#             weight=one(Float64)
#             push!(store_weight,weight)

#         else
#             p1=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][1]].coords
#             p2=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][2]].coords
#             p3=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][3]].coords
#             p4=nodes[cell_nodes[cells[face_ownerCells[i,1]].nodes_range][4]].coords
            
#             o1=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][1]].coords
#             o2=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][2]].coords
#             o3=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][3]].coords
#             o4=nodes[cell_nodes[cells[face_ownerCells[i,2]].nodes_range][4]].coords
            
#             c1=(p1+p2+p3+p4)/4
#             c2=(o1+o2+o3+o4)/4
#             d_1f=cf-c1
#             d_f2=c2-cf
#             d_12=c2-c1
            
#             if d_12⋅normal<0
#                 normal=-1.0*normal
#             end
            
#             delta=norm(d_12)
#             push!(store_delta,delta)
#             e=d_12/delta
#             push!(store_e,e)
#             weight=abs((d_1f⋅normal)/(d_1f⋅normal + d_f2⋅normal))
#             push!(store_weight,weight)
#         end
#         push!(store_normal,normal)
#         push!(store_area,area)
#         push!(store_centre,cf)
#     end
#     return store_normal,store_area,store_centre,store_e,store_delta,store_weight
# end

function generate_cell_neighbours(cells,cell_faces)
    cell_neighbours=Int64[]
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
end

# function generate_internal_faces(volumes,faces)
#     store_cell_faces=Int64[]
#     store_faces=Int64[]
    
#     for i=1:length(volumes)
#     cell=sort(volumes[i].volumes)
#     push!(store_cell_faces,cell[1],cell[2],cell[3])
#     push!(store_cell_faces,cell[1],cell[2],cell[4])
#     push!(store_cell_faces,cell[1],cell[3],cell[4])
#     push!(store_cell_faces,cell[2],cell[3],cell[4])
#     end
    
#     for i=1:length(faces)
#         face=sort(faces[i].faces)
#         push!(store_faces,face[1],face[2],face[3])
#     end

#     range=UnitRange{Int64}[]

#     x=0
#     for i=1:length(store_cell_faces)/3 # Very dodgy! could give float!!!
#         store=UnitRange(x+1:x+3)
#         x=x+3
#         push!(range,store)
#     end

#     faces1=Vector{Int64}[]

#     for i=1:length(range)
#         face1=store_cell_faces[range[i]]
#         for ic=1:length(range)
#             face2=store_cell_faces[range[ic]]
#             store=[] # Why are you allocating inside the loop?
    
#             push!(store,face2[1] in face1)
#             push!(store,face2[2] in face1)
#             push!(store,face2[3] in face1)

#             count1=count(store)
#             # println(count1)
    
#             if count1!=3
#             # if length(store) !=3 # is this what you want to do?
#                 push!(faces1,face1)
#             end
#         end
#     end

#     all_faces=unique(faces1)

#     store1_faces=Vector{Int64}[]
#     for i=1:length(faces)
#         push!(store1_faces,sort(faces[i].faces))
#     end

#     all_faces=sort(all_faces)
#     store1_faces=sort(store1_faces)
    
#     internal_faces=setdiff(all_faces,store1_faces)
    
#     for i=1:length(internal_faces)
#         push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
#     end
#     return faces
# end

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

function quad_internal_faces(volumes,faces)
    store_cell_faces1=Int64[]

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

    

    sorted_cell_faces=Int64[]
    for i=1:length(store_cell_faces1)

        push!(sorted_cell_faces,sort(store_cell_faces1[i]))
    end


    sorted_faces=Int64[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    internal_faces=setdiff(sorted_cell_faces,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end


function generate_boundaries(boundaryElements,boundary_face_range)
    boundaries=Boundary{Symbol, UnitRange{Int64}}[]
    for i=1:length(boundaryElements)
        push!(boundaries,Boundary(Symbol(boundaryElements[i].name),boundary_face_range[i]))
    end
    return boundaries
end

function generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)
    boundary_cells = Int64[]
    store = Int64[]
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

function generate_boundary_faces(boundaryElements)
    boundary_faces=Int64[]
    z=0
    wipe=Int64[]
    boundary_face_range=UnitRange{Int64}[]
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

# function generate_boundary_faces(boundaryElements)
#     boundary_faces=[]
#     z=0
#     wipe=Int64[]
#     boundary_face_range=[]
#     for i=1:length(boundaryElements)
#         for n=1:length(boundaryElements[i].elements)
#             push!(boundary_faces,boundaryElements[i].elements[n])
#             push!(wipe,boundaryElements[i].elements[n])
#         end
#         if length(wipe)==1
#             push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[1]))
#             z=z+1
#         elseif length(wipe) ≠1
#             push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[end]))
#             z=z+length(wipe)
#         end
#         wipe=Int64[]
#     end

#     store=[]

#     boundary_face_range=sort(boundary_face_range)
#     for i=1:length(boundary_face_range)
#         for ic=boundary_face_range[i]
#         push!(store,ic)
#         end
#     end

#     store

#     return store,boundary_face_range
# end


function generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)
    x=Vector{Int64}[]
    for i=1:length(faces)
        push!(x,findall(x->x==i,all_cell_faces))
    end
    y=zeros(Int,length(x),2)
    for ic=1:length(volumes)
        for i=1:length(x)
            #if length(x[i])==1
                if all_cell_faces_range[ic][1]<=x[i][1]<=all_cell_faces_range[ic][end]
                    y[i,1]=ic
                    y[i,2]=ic
                end
            #end

            if length(x[i])==2
                if all_cell_faces_range[ic][1]<=x[i][2]<=all_cell_faces_range[ic][end]
                    #y[i]=ic
                    y[i,2]=ic

                end
            end

        end
    end
    return y
end




#Generate nodes
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

#Generate Faces
function generate_face_nodes(faces)
    face_nodes=typeof(faces[1].faces[1])[] # Giving type to array constructor
    for n=1:length(faces)
        for i=1:faces[n].faceCount
            push!(face_nodes,faces[n].faces[i])
        end
    end
    return face_nodes
end

#Generate cells
function generate_cell_nodes(volumes)
    cell_nodes=typeof(volumes[1].volumes[1])[] # Giving type to array constructor
    for n=1:length(volumes)
        for i=1:volumes[n].volumeCount
            push!(cell_nodes,volumes[n].volumes[i])
        end
    end
    return cell_nodes
end

# function generate_cell_faces(volumes,faces)
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

# function generate_all_cell_faces(volumes,faces)
#     cell_faces=typeof(faces[1].faceindex)[]
#     for i=1:length(volumes)
#         for ic=1:length(faces)
#             bad=sort(volumes[i].volumes)
#             good=sort(faces[ic].faces)
#             store=typeof(good[1])[]
#             true_store=typeof(true)[]

#             for ip=1:length(good)
#                 push!(store,good[ip] in bad)
#                 push!(true_store,true)
#             end

#             if store[1:length(good)] == true_store
#                 push!(cell_faces,faces[ic].faceindex)
#             end
#             continue
#         end
#     end
#     return cell_faces
# end

function generate_all_cell_faces(faces,cell_face_nodes)
    all_cell_faces=Int[]
    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    for i=1:length(cell_face_nodes)
        push!(all_cell_faces,findfirst(x -> x==cell_face_nodes[i],sorted_faces))
    end
    return all_cell_faces
end

#function generate_cell_faces(faces)
    #cell_faces=[]
    #for n=1:length(faces)
            #push!(cell_faces,faces[n].faceindex)
    #end
    #return cell_faces
#end

#function generate_cell_faces(faces)
    #cell_faces=[]
    #for n=1:length(faces)
        #for i=1:faces[n].faceCount
            #push!(cell_faces,faces[n].faces[i])
        #end
    #end
    #return cell_faces
#end

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range=UnitRange(0,0)
    store=typeof(cell_nodes_range)[]
    x=0
    for i=1:length(volumes)
        #cell_nodes_range=UnitRange(volumes[i].volumes[1],volumes[i].volumes[end])
        cell_nodes_range=UnitRange(x+1,x+length(volumes[i].volumes))
        x=x+length(volumes[i].volumes)
        push!(store,cell_nodes_range)
    end
    return store
end

function generate_face_nodes_range(faces)
    face_nodes_range=UnitRange(0,0)
    store=typeof(face_nodes_range)[]
    x=0
    for i=1:length(faces)
        #face_nodes_range=UnitRange(((faces[i].faceCount)*i-(faces[i].faceCount-1)),(faces[i].faceCount)*i)
        #face_nodes_range=UnitRange((length(faces[i].faces)-length(faces[i].faces)+1)*i,(length(faces[i].faces))*i)
        face_nodes_range=UnitRange(x+1,x+faces[i].faceCount)
        x=x+faces[i].faceCount
        push!(store,face_nodes_range)
    end
    return store
end

#Faces Range
# function generate_all_faces_range(volumes,faces)
#     cell_faces_range=UnitRange(0,0)
#     store=[]
#     x=0
#     @inbounds for i=1:length(volumes)
#         #Tetra
#         if length(volumes[i].volumes)==4
#             #cell_faces_range=UnitRange(faces[(4*i)-3].faceindex,faces[4*i].faceindex)
#             cell_faces_range=UnitRange(x+1,x+length(volumes[i].volumes))
#             x=x+length(volumes[i].volumes)
#             push!(store,cell_faces_range)
#         end

#         #Hexa
#         if length(volumes[i].volumes)==8
#                 cell_faces_range=UnitRange(faces[6*i-5].faceindex,faces[6*i].faceindex)
#                 push!(store,cell_faces_range)
#         end

#         #wedge
#         if length(volumes[i].volumes)==6
#                 cell_faces_range=UnitRange(faces[5*i-4].faceindex,faces[5*i].faceindex)
#                 push!(store,cell_faces_range)
#         end
#     end
#     return store
# end

function generate_all_faces_range(volumes)
    cell_faces_range=UnitRange(0,0)
    store=typeof(cell_faces_range)[]
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

#Generate cells
function generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)
    cells=Cell{Float64,SVector{3,Float64},UnitRange{Int64}}[]
    for i=1:length(volumes)
        push!(cells,Cell(centre_of_cells[i],volume_of_cells[i],cell_nodes_range[i],cell_faces_range[i]))
    end
    return cells
end

function generate_faces(faces,face_nodes_range,faces_centre,faces_normal,faces_area,face_ownerCells,faces_e,faces_delta,faces_weight)
    faces3D=Face3D{Float64,SVector{2,Int64},SVector{3,Float64},UnitRange{Int64}}[]
    for i=1:length(faces)
        push!(faces3D,Face3D(face_nodes_range[i],SVector(face_ownerCells[i,1],face_ownerCells[i,2]),faces_centre[i],faces_normal[i],faces_e[i],faces_area[i],faces_delta[i],faces_weight[i]))
    end
    return faces3D
end

#Node connectivity

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

function generate_node_cells(points,volumes)
    neighbour=Int64[]
    store=Int64[]
    #cells_range=UnitRange(0,0)
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(store,neighbour)
                end
                continue
            end
        end
    end
    return store
end