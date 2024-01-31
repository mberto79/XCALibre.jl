#VTP file writer
function write_vtp_3D(name,mesh,edges)
    filename=name*".vtp"
    open(filename,"w") do io

        #Variables
        nPoints=length(mesh[1].nodes)
        nVerts=0
        nLines=length(edges)
        nStrips=0
        nPolys=length(mesh[1].faces)
        type="""PolyData"""
        version="""0.1"""
        byte_order="""LittleEndian"""
        F32="""Float32"""
        I32="""Int32"""
        UInt8="""UInt8"""
        threeD="""3"""
        format="""ascii"""
        con="""connectivity"""
        offsets="""offsets"""
        version1="""1.0"""
        types="""types"""


        #Modifying Data
        store_nodes=zeros(length(mesh[1].nodes)*3)

        for i=1:length(mesh[1].nodes)
            store_nodes[(3*i-2):(3*i)]=mesh[1].nodes[i].coords
        end

        #nodesID=LinRange(1,length(mesh[1].nodes),length(mesh[1].nodes))

        store_faces=zeros(Int32,length(mesh[1].faces))
        for i=1:length(mesh[1].faces)
            store_faces[i]=length(mesh[1].faces[1].nodes_range)*i
        end

        store_cells=zeros(Int32,length(mesh[1].cells))
        for i=1:length(mesh[1].cells)
            store_cells[i]=length(mesh[1].cells[1].nodes_range)*i
        end
        
        cell_types=zeros(Int32,length(mesh[1].cells))
        if mesh[1].cells[1].nodes_range[end]==4
            for i=1:length(mesh[1].cells)
                cell_types[i]=10
            end
        end

        if mesh[1].cells[1].nodes_range[end]==8
            for i=1:length(mesh[1].cells)
                cell_types[i]=12
            end
        end

        edge_store=zeros(Int64,length(edges)*2)
        for i=1:length(edges)
            edge_store[(2*i-1):(2*i)]=edges[i].edges
        end

        edge_range=zeros(Int64,length(edges))
        for i=1:length(edges)
            edge_range[i]=2*i
        end

        #Writing to VTP

        write(io,"?xml version=\"$(version1)\"?\n")
        write(io," <VTKFile type=\"$(type)\" version=\"$(version)\" byte_order=\"$(byte_order)\">\n")
        write(io,"  <PolyData>\n")

        #Overall
        write(io,"   <Piece NumberOfPoints=\"$(nPoints)\" NumberofVerts=\"$(nVerts)\" NumberofLines=\"$(nLines)\" NumberofStrips=\"$(nStrips)\" NumberofPolys=\"$(nPolys)\">\n")

        #Cells
        write(io,"   <Cells>\n")

        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(con)\">\n")
        write(io,"     $(join(mesh[1].cell_nodes," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(offsets)\">\n")
        write(io,"     $(join(store_cells," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"    <DataArray type=\"$(UInt8)\" Name=\"$(types)\">\n")
        write(io,"     $(join(cell_types," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"   </Cells>\n")

        #Nodes/Points
        write(io,"   <Points>\n")

        write(io,"    <DataArray type=\"$(F32)\" NumberOfComponents=\"$(threeD)\" format=\"$(format)\">\n")
        write(io,"     $(join(store_nodes," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"   </Points>\n")

        #Verts

        #Lines
        write(io,"   <Lines>\n")

        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(con)\">\n")
        write(io,"     $(join(edge_store," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(offsets)\">\n")
        write(io,"     $(join(edge_range," "))\n")
        write(io,"    </DataArray>\n")

        write(io,"   </Lines>\n")

        #Strips

        #Face Data
        write(io,"   <Polys>\n")
        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(con)\" format=\"$(format)\">\n")
        write(io,"     $(join(mesh[1].face_nodes," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"    <DataArray type=\"$(I32)\" Name=\"$(offsets)\" format=\"$(format)\">\n")
        write(io,"     $(join(store_faces," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"   </Polys>\n")

        #End
        write(io,"   </Piece>\n")
        write(io,"  </PolyData>\n")
        write(io," </VTKFile>\n")
    end
end

name="test_vtp"
write_vtp_3D(name,mesh,edges)