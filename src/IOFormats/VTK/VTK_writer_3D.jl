export initialise_writer

get_data(arr, backend::KernelAbstractions.GPU) = begin
    arr_cpu = Array{eltype(arr)}(undef, length(arr))
    copyto!(arr_cpu, arr)
    arr_cpu
end

get_data(arr, backend::KernelAbstractions.CPU) = begin
    arr
end

segment(p1, p2) = p2 - p1
unit_vector(vec) = vec/norm(vec)

function initialise_writer(format::VTK, mesh::Mesh3)
    @info "Initialise writer (Store mesh in host memory)"
    # Extract mesh information (copy to CPU if mesh in GPU format)
    backend = _get_backend(mesh)
    nodes_cpu = get_data(mesh.nodes, backend)
    faces_cpu = get_data(mesh.faces, backend)
    cells_cpu = get_data(mesh.cells, backend)
    cell_nodes_cpu = get_data(mesh.cell_nodes, backend)
    face_nodes_cpu = get_data(mesh.face_nodes, backend)

    # Variables
    nPoints=length(nodes_cpu)
    nCells=length(cells_cpu)
    type="UnstructuredGrid"
    version="1.0"
    format="ascii"
    byte_order="LittleEndian"
    F32="Float64"
    I64="Int64"
    poly=42
    x=0
    y=0

    # Writing constant header information

    header = IOBuffer()
    write(header, """
    <?xml version="$(version)"?>
     <VTKFile type="$(type)" version="$(version)" byte_order="$(byte_order)">
      <UnstructuredGrid>
       <Piece NumberOfPoints="$(nPoints)" NumberOfCells="$(nCells)">
        <Points>
         <DataArray type="$(F32)" NumberOfComponents="3" format="$(format)">
    """)

    for i=1:nPoints
        write(header,"      $(nodes_cpu[i].coords[1]) $(nodes_cpu[i].coords[2]) $(nodes_cpu[i].coords[3])\n")
    end
    write(header,"     </DataArray>\n")
    write(header,"    </Points>\n")
    write(header,"    <Cells>\n")
    write(header,"     <DataArray type=\"$(I64)\" Name=\"connectivity\" format=\"$(format)\">\n")
    for i=1:length(cells_cpu)
        write(header,"      $(join(cell_nodes_cpu[cells_cpu[i].nodes_range[1]:cells_cpu[i].nodes_range[end]].-1," "))\n")
    end
    write(header,"     </DataArray>\n")
    write(header,"     <DataArray type=\"$(I64)\" Name=\"offsets\" format=\"$(format)\">\n")
    node_counter=0
    for i=1:length(cells_cpu)
        node_counter=node_counter+length(cells_cpu[i].nodes_range)
        write(header,"      $(node_counter)\n")
    end
    write(header,"     </DataArray>\n")
    write(header,"     <DataArray type=\"$(I64)\" Name=\"faces\" format=\"$(format)\">\n")

    # This is needed because boundary faces are missing from cell level connectivity

    # Version 2 (x2.8 faster)
    # Calculating all the faces that belong to each cell
    all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_cpu)] 
    for fID ∈ eachindex(faces_cpu)
        owners = faces_cpu[fID].ownerCells
        owner1 = owners[1]
        owner2 = owners[2]
        push!(all_cell_faces[owner1],fID)
        if owner1 !== owner2 #avoid duplication of cells for boundary faces
            push!(all_cell_faces[owner2],fID)
        end
    end

    for (cID, fIDs) ∈ enumerate(all_cell_faces)
        write(header,"\t$(length(all_cell_faces[cID]))\n") # No. of Faces for each cell
        for fID ∈ fIDs
            #Ordering of face nodes so that they are ordered anti-clockwise when looking at the cell from the outside
            nIDs=face_nodes_cpu[faces_cpu[fID].nodes_range] # Get ids of nodes of face

            n1=nodes_cpu[nIDs[1]].coords # Coordinates of 3 nodes only.
            n2=nodes_cpu[nIDs[2]].coords
            n3=nodes_cpu[nIDs[3]].coords

            points = [n1, n2, n3]

            _x(n) = n[1]
            _y(n) = n[2]
            _z(n) = n[3]

            # surface vectors (segments connecting nodes to reference node)
            l = segment.(Ref(points[1]), points) 
            fn = unit_vector(l[2] × l[3]) # Calculating face normal 
            cc=cells_cpu[cID].centre
            fc=faces_cpu[fID].centre
            d_fc=fc-cc

            if dot(d_fc,fn)<0.0
                nIDs=reverse(nIDs)
            end
            write(
                header,"\t$(length(faces_cpu[fID].nodes_range)) $(join(nIDs .- 1," "))\n"
                )
        end
    end

    write(header,"     </DataArray>\n")
    write(header,"     <DataArray type=\"$(I64)\" Name=\"faceoffsets\" format=\"$(format)\">\n")

    for fIDs ∈ all_cell_faces
        totalNodes = 0
        for fID ∈ fIDs
            node_count = length(faces_cpu[fID].nodes_range)
            totalNodes += node_count
        end
        x += 1 + length(fIDs) + totalNodes
        write(header,"     $(x)\n")
    end

    write(header,"     </DataArray>\n")
    write(header,"     <DataArray type=\"$(I64)\" Name=\"types\" format=\"$(format)\">\n")

    for i=1:length(cells_cpu)
        write(header,"      $(poly)\n")
    end
    write(header,"     </DataArray>\n")
    write(header,"    </Cells>\n")
    write(header,"    <CellData>\n")

    # Write footer information

    footer = IOBuffer()
    write(footer,"    </CellData>\n")
    write(footer,"   </Piece>\n")
    write(footer,"  </UnstructuredGrid>\n")
    write(footer," </VTKFile>\n")

    return VTKWriter3D(
        String(take!(header)), 
        String(take!(footer))
        )
end


function write_results(iteration::TI, mesh, meshData::VTKWriter3D, args...) where TI
    name = ""
    if TI <: Integer
        name = @sprintf "iteration_%i" iteration
    else
        name = @sprintf "time_%.8f" iteration
    end
    filename=name*".vtu"

    # Define backend and variables
    backend = _get_backend(mesh)
    format = "ascii"
    F32 = "Float32"

    open(filename,"w") do io

        write(io, meshData.header)

        for arg ∈ args
            label = arg[1]
            field = arg[2]
            field_type=typeof(field)
            if field_type <: ScalarField
                write(io,"     <DataArray type=\"$(F32)\" Name=\"$(label)\" format=\"$(format)\">\n")
                values_cpu = get_data(field.values, backend)
                for value ∈ values_cpu
                    println(io,value)
                end
                write(io,"     </DataArray>\n")
            elseif field_type <: VectorField
                write(io,"     <DataArray type=\"$(F32)\" Name=\"$(label)\" format=\"$(format)\" NumberOfComponents=\"3\">\n")
                x_cpu = get_data(field.x.values, backend)
                y_cpu = get_data(field.y.values, backend)
                z_cpu = get_data(field.z.values, backend)
                for i ∈ eachindex(x_cpu)
                    println(io, x_cpu[i]," ",y_cpu[i] ," ",z_cpu[i] )
                end
                write(io,"     </DataArray>\n")
            else
                throw("""
                Input data should be a ScalarField or VectorField e.g. ("U", U)
                """)
            end
        end

        write(io, meshData.footer)

    end
    nothing
end
