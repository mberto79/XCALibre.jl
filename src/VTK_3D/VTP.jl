#VTP file writer
function write_vtp_3D(name,mesh)
    filename=name*".vtp"
    open(filename,"w") do io
        nPoints=length(mesh[1].nodes)
        nVerts=0
        #nCells=length(mesh[1].cells)
        #nLines=length(edges)
        nLines=0
        nStrips=0
        nPolys=length(mesh[1].faces)
        type="""PolyData"""
        version="""0.1"""
        byte_order="""LittleEndian"""
        F32="""Float32"""
        I32="""Int32"""
        threeD="""3"""
        format="""ascii"""
        my_scalars="""my_scalars"""
        #cell_scalars="cell_scalars"
        #cell_normals="cell_normals"
        pname="""connectivity"""
        offsets="""offsets"""
        version1="""1.0"""

        store_nodes=zeros(length(mesh[1].nodes)*3)
        for i=1:length(mesh[1].nodes)
            store_nodes[(3*i-2):(3*i)]=mesh[1].nodes[i].coords
        end

        nodesID=LinRange(1,length(mesh[1].nodes),length(mesh[1].nodes))

        store_faces=zeros(Int32,length(mesh[1].faces))
        for i=1:length(mesh[1].faces)
            store_faces[i]=length(mesh[1].faces[1].nodes_range)*i
        end

        write(io,"?xml version=$(version1)?\n")
        write(io," <VTKFile type=$(type) version=$(version) byte_order=$(byte_order)>\n")
        write(io,"  <PolyData>")
        write(io,"   <Piece NumberOfPoints=$(nPoints) NumberofVerts=$(nVerts) NumberofLines=$(nLines) NumberofStrips=$(nStrips) NumberofPolys=$(nPolys)>\n")
        write(io,"   <Points>\n")
        write(io,"    <DataArray type=$(F32) NumberOfComponents=$(threeD) format=$(format)>\n")
        write(io,"     $(join(store_nodes," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"   </Points>\n")
        write(io,"   <PointData Scalars=$(my_scalars)>\n")
        write(io,"    <DataArray type=$(F32) Name=$(my_scalars) format=$(format)>\n")
        write(io,"     $(join(nodesID," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"   </PointData>\n")
        #write(io,"   <CellData Scalars=$(cell_scalars) Normals=$(cell_normals)>")
        #write(io,"    <DataArray type=$(I32) Name=$(cell_scalars) format=$(format)>")
        #write(io,"     $(mesh.cellsID)")
        #write(io,"    </DataArray>")
        #write(io,"    <DataArray type=$(F32) Name=$(cell_normals) NumberOfComponents=$(threeD) format=$(ascii)>")
        #write(io,"     $(mesh.cell_nsign)")
        #write(io,"    </DataArray>")
        #write(io,"   </CellData>")
        write(io,"   <Polys>\n")
        write(io,"    <DataArray type=$(I32) Name=$(pname) format=$(format)>\n")
        write(io,"     $(join(mesh[1].face_nodes," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"    <DataArray type=$(I32) Name=$(offsets) format=$(format)>\n")
        write(io,"     $(join(store_faces," "))\n")
        write(io,"    </DataArray>\n")
        write(io,"   </Polys>\n")
        write(io,"   </Piece>\n")
        write(io,"  </PolyData>\n")
        write(io," </VTKFile>\n")
    end
end

name="test_vtp"
write_vtp_3D(name,mesh)