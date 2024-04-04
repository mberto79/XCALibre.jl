# function write_vtu(name,mesh)
# export write_vtk, model2vtk
# export copy_to_cpu

# function model2vtk(model::RANS{Laminar,F1,F2,V,T,E,D}, name) where {F1,F2,V,T,E,D}
#     args = (
#         ("U", model.U), 
#         ("p", model.p)
#     )
#     write_vtk(name, model.mesh, args...)
# end

# function model2vtk(model::RANS{KOmega,F1,F2,V,T,E,D}, name) where {F1,F2,V,T,E,D}
#     args = (
#         ("U", model.U), 
#         ("p", model.p),
#         ("k", model.turbulence.k),
#         ("omega", model.turbulence.omega),
#         ("nut", model.turbulence.nut)
#     )
#     write_vtk(name, model.mesh, args...)
# end

get_data(arr, backend::CUDABackend) = begin
    arr_cpu = Array{eltype(arr)}(undef, length(arr))
    copyto!(arr_cpu, arr)
    a_cpu
end

get_data(arr, backend::CPU) = begin
    arr
end

function write_vtk(name, mesh::Mesh3, args...)
    filename=name*".vtu"

    # Deactivate copies below for serial version of the codebase
    backend = _get_backend(mesh)
    nodes_cpu = get_data(mesh.nodes, backend)
    faces_cpu = get_data(mesh.faces, backend)
    cells_cpu = get_data(mesh.cells, backend)
    cell_nodes_cpu = get_data(mesh.cell_nodes, backend)
    face_nodes_cpu = get_data(mesh.face_nodes, backend)

    # Serial version
    # nodes_cpu = mesh.nodes
    # faces_cpu = mesh.faces
    # cells_cpu = mesh.cells
    # cell_nodes_cpu = mesh.cell_nodes
    # face_nodes_cpu = mesh.face_nodes

    open(filename,"w") do io

        #Variables
        nPoints=length(nodes_cpu)
        nCells=length(cells_cpu)
        type="UnstructuredGrid"
        one="1.0"
        format="ascii"
        byte_order="LittleEndian"
        F32="Float32"
        I64="Int64"
        con="connectivity"
        three="3"
        offsets="offsets"
        faces="faces"
        face_offsets="faceoffsets"
        types="types"
        #temp="temperature"
        scalar="scalar"
        vector="vector"
        #pressure="pressure"
        poly=42
        x=0
        y=0

        # Not in use!
        # #Modifying Data
        # store_cells=zeros(Int32,length(cells_cpu))
        # for i=1:length(cells_cpu)
        #     store_cells[i]=length(cells_cpu[1].nodes_range)*i
        # end

        #temp
        #temp_cells=LinRange(0,500,length(cells_cpu))

        #pressure
        #pressure_cells=LinRange(0,10000,length(cells_cpu))

        #Writing
        write(io,"<?xml version=\"$(one)\"?>\n")
        write(io," <VTKFile type=\"$(type)\" version=\"$(one)\" byte_order=\"$(byte_order)\">\n")
        write(io,"  <UnstructuredGrid>\n")
        write(io,"   <Piece NumberOfPoints=\"$(nPoints)\" NumberOfCells=\"$(nCells)\">\n")
        write(io,"    <Points>\n")
        write(io,"     <DataArray type=\"$(F32)\" NumberOfComponents=\"$(three)\" format=\"$(format)\">\n")
        for i=1:nPoints
            write(io,"      $(nodes_cpu[i].coords[1]) $(nodes_cpu[i].coords[2]) $(nodes_cpu[i].coords[3])\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"    </Points>\n")
        write(io,"    <Cells>\n")
        #write(io,"     <DataArray type=\"$(I64)\" Name=\"$(con)\" format=\"$(format)\">\n")
        #write(io,"      $(join((cell_nodes_cpu.-1)," "))\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(con)\" format=\"$(format)\">\n")
        #write(io,"      $(length(cells_cpu))\n")
        #for i=1:length(cells_cpu)
            #write(io,"      $(length(cells_cpu[i].nodes_range)) $(join(cell_nodes_cpu[cells_cpu[i].nodes_range[1]:cells_cpu[i].nodes_range[end]].-1," "))\n")
        #end
        for i=1:length(cells_cpu)
            write(io,"      $(join(cell_nodes_cpu[cells_cpu[i].nodes_range[1]:cells_cpu[i].nodes_range[end]].-1," "))\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(offsets)\" format=\"$(format)\">\n")
        #write(io,"      $(join(store_cells," "))\n")
        #write(io,"      $(length(cell_nodes_cpu)+length(cells_cpu)+1)\n")
        for i=1:length(cells_cpu)
            write(io,"      $(length(cells_cpu[i].nodes_range)*i)\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(faces)\" format=\"$(format)\">\n")
        
        # for i=1:length(cells_cpu)
        #     write(io,"      $(length(cells_cpu[i].faces_range))\n")
        #     for ic=cells_cpu[i].faces_range
        #     write(io,"      $(length(faces_cpu[i].nodes_range)) $(join(face_nodes_cpu[faces_cpu[mesh.cell_faces[ic]].nodes_range].-1," "))\n")
        #     end
        # end

        # This is needed because boundary faces are missing from cell level connectivity
        
        # # Version 1
        # for i=1:length(cells_cpu)
        #     store_faces=[]
        #     for id=1:length(faces_cpu)
        #         if faces_cpu[id].ownerCells[1]==i || faces_cpu[id].ownerCells[2]==i
        #             push!(store_faces,id)
        #         end
        #     end
        #     write(io,"      $(length(store_faces))\n")
        #     for ic=1:length(store_faces)
        #     write(io,"      $(length(faces_cpu[store_faces[ic]].nodes_range)) $(join(face_nodes_cpu[faces_cpu[store_faces[ic]].nodes_range].-1," "))\n")
        #     end
        # end

        # # Version 2 (x2.8 faster)
        all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_cpu)]
        for fID ∈ eachindex(faces_cpu)
            owners = faces_cpu[fID].ownerCells
            owner1 = owners[1]
            owner2 = owners[2]
            # if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
            push!(all_cell_faces[owner1],fID)
            if owner1 !== owner2 #avoid duplication of cells for boundary faces
                push!(all_cell_faces[owner2],fID)
            end
            # end
        end
        for (cID, fIDs) ∈ enumerate(all_cell_faces)
            write(io,"\t$(length(all_cell_faces[cID]))\n")
            for fID ∈ fIDs
                write(
                    io,"\t$(length(faces_cpu[fID].nodes_range)) $(join(face_nodes_cpu[faces_cpu[fID].nodes_range] .- 1," "))\n"
                    )
            end
        end

        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(face_offsets)\" format=\"$(format)\">\n")
        #write(io,"      $(length(faces_cpu)*(length(faces_cpu[1].nodes_range)+1)+1)\n")
        #write(io,"      $(length(face_nodes_cpu)+length(faces_cpu)+1)\n")
        # for i=1:length(cells_cpu)
        #     x=1+(length(cells_cpu[i].faces_range)+length(mesh.cell_faces[cells_cpu[i].faces_range])*3)
        #     y=x+y
        #     write(io,"     $(y)\n")
        # end

        for i=1:length(cells_cpu)
            if length(cells_cpu[i].nodes_range)==4
                x=17*i
                write(io,"     $(x)\n")
            end
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(types)\" format=\"$(format)\">\n")
        #write(io,"      $(poly)\n")
        for i=1:length(cells_cpu)
            write(io,"      $(poly)\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"    </Cells>\n")
        write(io,"    <CellData>\n")


        # write(io,"     <DataArray type=\"$(F32)\" Name=\"$(temp)\" format=\"$(format)\">\n")
        # for i=1:length(temp_cells)
        #     write(io,"      $(join(temp_cells[i]," "))\n")
        # end
        # write(io,"     </DataArray>\n")


        # write(io,"     <DataArray type=\"$(F32)\" Name=\"$(pressure)\" format=\"$(format)\">\n")
        # for i=1:length(pressure_cells)
        #     write(io,"      $(join(pressure_cells[i]," "))\n")
        # end
        # write(io,"     </DataArray>\n")

        for arg ∈ args
            label = arg[1]
            field = arg[2]
            field_type=typeof(field)
            if field_type <: ScalarField
                write(io,"     <DataArray type=\"$(F32)\" Name=\"$(label)\" format=\"$(format)\">\n")
                # values_cpu= copy_scalarfield_to_cpu(field.values, backend)
                values_cpu= field.values
                for value ∈ values_cpu
                    println(io,value)
                end
                write(io,"     </DataArray>\n")
            elseif field_type <: VectorField
                write(io,"     <DataArray type=\"$(F32)\" Name=\"$(label)\" format=\"$(format)\" NumberOfComponents=\"3\">\n")
                # x_cpu, y_cpu, z_cpu = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
                x_cpu, y_cpu, z_cpu = field.x.values, field.y.values, field.z.values
                for i ∈ eachindex(x_cpu)
                    println(io, x_cpu[i]," ",y_cpu[i] ," ",z_cpu[i] )
                end
                write(io,"     </DataArray>\n")

                # write out single component
                # println(io,"     <DataArray type=\"$(F32)\" Name=\"Ux\" format=\"$(format)\">")
                # # x_cpu, y_cpu, z_cpu = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
                # x_cpu, y_cpu, z_cpu = field.x.values, field.y.values, field.z.values
                # for i ∈ eachindex(x_cpu)
                #     println(io, x_cpu[i])
                # end
                # println(io,"     </DataArray>")
            else
                throw("""
                Input data should be a ScalarField or VectorField e.g. ("U", U)
                """)
            end
        end


        write(io,"    </CellData>\n")
        write(io,"   </Piece>\n")
        write(io,"  </UnstructuredGrid>\n")
        write(io," </VTKFile>\n")
    end
end
