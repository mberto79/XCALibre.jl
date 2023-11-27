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

        write(io,"?xml version=$(version1)?")
        write(io," <VTKFile type=$(type) version=$(version) byte_order=$(byte_order)>")
        write(io,"  <PolyData>")
        write(io,"   <Piece NumberOfPoints=$(nPoints) NumberofVerts=$(nVerts) NumberofLines=$(nLines) NumberofStrips=$(nStrips) NumberofPolys=$(nPolys)>")
        write(io,"   <Points>")
        write(io,"    <DataArray type=$(F32) NumberOfComponents=$(threeD) format=$(format)>")
        write(io,"     $(mesh.nodes)")
        write(io,"    </DataArray>")
        write(io,"   </Points>")
        write(io,"   <PointData Scalars=$(my_scalars)>")
        write(io,"    <DataArray type=$(F32) Name=$(my_scalars) format=$(format)>")
        write(io,"     $(mesh.nodesID)")
        write(io,"    </DataArray>")
        write(io,"   </PointData>")
        #write(io,"   <CellData Scalars=$(cell_scalars) Normals=$(cell_normals)>")
        #write(io,"    <DataArray type=$(I32) Name=$(cell_scalars) format=$(format)>")
        #write(io,"     $(mesh.cellsID)")
        #write(io,"    </DataArray>")
        #write(io,"    <DataArray type=$(F32) Name=$(cell_normals) NumberOfComponents=$(threeD) format=$(ascii)>")
        #write(io,"     $(mesh.cell_nsign)")
        #write(io,"    </DataArray>")
        #write(io,"   </CellData>")
        write(io,"   <Polys>")
        write(io,"    <DataArray type=$(I32) Name=$(pname) format=$(format)>")
        write(io,"     $(mesh.face_nodes)")
        write(io,"    </DataArray>")
        write(io,"    <DataArray type=$(I32) Name=$(offsets) format=$(format)>")
        write(io,"     $(mesh.faces_offset)")
        write(io,"    </DataArray>")
        write(io,"   </Polys>")
        write(io,"   </Piece>")
        write(io,"  </PolyData>")
        write(io," </VTKFile>")
    end
end

name="test_vtp"
write_vtp_3D(name,mesh[1])