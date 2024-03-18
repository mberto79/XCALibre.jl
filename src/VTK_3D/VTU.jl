function write_vtu(name,mesh)
    filename=name*".vtu"
    open(filename,"w") do io

        #Variables
        nPoints=length(mesh.nodes)
        nCells=length(mesh.cells)
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
        temp="temperature"
        pressure="pressure"
        poly=42
        x=0
        y=0

        #Modifying Data
        store_cells=zeros(Int32,length(mesh.cells))
        for i=1:length(mesh.cells)
            store_cells[i]=length(mesh.cells[1].nodes_range)*i
        end

        #temp
        temp_cells=LinRange(0,500,length(mesh.cells))

        #pressure
        pressure_cells=LinRange(0,10000,length(mesh.cells))

        #Writing
        write(io,"<?xml version=\"$(one)\"?>\n")
        write(io," <VTKFile type=\"$(type)\" version=\"$(one)\" byte_order=\"$(byte_order)\">\n")
        write(io,"  <UnstructuredGrid>\n")
        write(io,"   <Piece NumberOfPoints=\"$(nPoints)\" NumberOfCells=\"$(nCells)\">\n")
        write(io,"    <Points>\n")
        write(io,"     <DataArray type=\"$(F32)\" NumberOfComponents=\"$(three)\" format=\"$(format)\">\n")
        for i=1:nPoints
            write(io,"      $(mesh.nodes[i].coords[1]) $(mesh.nodes[i].coords[2]) $(mesh.nodes[i].coords[3])\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"    </Points>\n")
        write(io,"    <Cells>\n")
        #write(io,"     <DataArray type=\"$(I64)\" Name=\"$(con)\" format=\"$(format)\">\n")
        #write(io,"      $(join((mesh.cell_nodes.-1)," "))\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(con)\" format=\"$(format)\">\n")
        #write(io,"      $(length(mesh.cells))\n")
        #for i=1:length(mesh.cells)
            #write(io,"      $(length(mesh.cells[i].nodes_range)) $(join(mesh.cell_nodes[mesh.cells[i].nodes_range[1]:mesh.cells[i].nodes_range[end]].-1," "))\n")
        #end
        for i=1:length(mesh.cells)
            write(io,"      $(join(mesh.cell_nodes[mesh.cells[i].nodes_range[1]:mesh.cells[i].nodes_range[end]].-1," "))\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(offsets)\" format=\"$(format)\">\n")
        #write(io,"      $(join(store_cells," "))\n")
        #write(io,"      $(length(mesh.cell_nodes)+length(mesh.cells)+1)\n")
        for i=1:length(mesh.cells)
            write(io,"      $(length(mesh.cells[i].nodes_range)*i)\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(faces)\" format=\"$(format)\">\n")
        
        # for i=1:length(mesh.cells)
        #     write(io,"      $(length(mesh.cells[i].faces_range))\n")
        #     for ic=mesh.cells[i].faces_range
        #     write(io,"      $(length(mesh.faces[i].nodes_range)) $(join(mesh.face_nodes[mesh.faces[mesh.cell_faces[ic]].nodes_range].-1," "))\n")
        #     end
        # end

        
        for i=1:length(mesh.cells)
            store_faces=[]
            for id=1:length(mesh.faces)
                if mesh.faces[id].ownerCells[1]==i || mesh.faces[id].ownerCells[2]==i
                    push!(store_faces,id)
                end
            end
            write(io,"      $(length(store_faces))\n")
            for ic=1:length(store_faces)
            write(io,"      $(length(mesh.faces[store_faces[ic]].nodes_range)) $(join(mesh.face_nodes[mesh.faces[store_faces[ic]].nodes_range].-1," "))\n")
            end
        end

        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(face_offsets)\" format=\"$(format)\">\n")
        #write(io,"      $(length(mesh.faces)*(length(mesh.faces[1].nodes_range)+1)+1)\n")
        #write(io,"      $(length(mesh.face_nodes)+length(mesh.faces)+1)\n")
        # for i=1:length(mesh.cells)
        #     x=1+(length(mesh.cells[i].faces_range)+length(mesh.cell_faces[mesh.cells[i].faces_range])*3)
        #     y=x+y
        #     write(io,"     $(y)\n")
        # end

        for i=1:length(mesh.cells)
            if length(mesh.cells[i].nodes_range)==4
                x=17*i
                write(io,"     $(x)\n")
            end
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(I64)\" Name=\"$(types)\" format=\"$(format)\">\n")
        #write(io,"      $(poly)\n")
        for i=1:length(mesh.cells)
            write(io,"      $(poly)\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"    </Cells>\n")
        write(io,"    <CellData>\n")
        write(io,"     <DataArray type=\"$(F32)\" Name=\"$(temp)\" format=\"$(format)\">\n")
        for i=1:length(temp_cells)
            write(io,"      $(join(temp_cells[i]," "))\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"     <DataArray type=\"$(F32)\" Name=\"$(pressure)\" format=\"$(format)\">\n")
        for i=1:length(pressure_cells)
            write(io,"      $(join(pressure_cells[i]," "))\n")
        end
        write(io,"     </DataArray>\n")
        write(io,"    </CellData>\n")
        write(io,"   </Piece>\n")
        write(io,"  </UnstructuredGrid>\n")
        write(io," </VTKFile>\n")
    end
end