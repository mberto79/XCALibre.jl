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

get_data(a, backend::CUDABackend) = begin
    a_cpu = Array{eltype(a)}(undef, length(a))
    copyto!(a_cpu, a)
    a_cpu
end

function write_vtk(name, mesh::Mesh3, args...)
filename=name*".vtu"
backend = _get_backend(mesh)

nodes_cpu = get_data(mesh.nodes, backend)
faces_cpu = get_data(mesh.faces, backend)
cells_cpu = get_data(mesh.cells, backend)
cell_nodes_cpu = get_data(mesh.cell_nodes, backend)
face_nodes_cpu = get_data(mesh.face_nodes, backend)

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

    #Modifying Data
    store_cells=zeros(Int32,length(cells_cpu))
    for i=1:length(cells_cpu)
        store_cells[i]=length(cells_cpu[1].nodes_range)*i
    end

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

    
    for i=1:length(cells_cpu)
        store_faces=[]
        for id=1:length(faces_cpu)
            if faces_cpu[id].ownerCells[1]==i || faces_cpu[id].ownerCells[2]==i
                push!(store_faces,id)
            end
        end
        write(io,"      $(length(store_faces))\n")
        for ic=1:length(store_faces)
        write(io,"      $(length(faces_cpu[store_faces[ic]].nodes_range)) $(join(face_nodes_cpu[faces_cpu[store_faces[ic]].nodes_range].-1," "))\n")
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
            write(io,"     <DataArray type=\"$(F32)\" Name=\"$(scalar)\" format=\"$(format)\">\n")
            values_cpu= copy_scalarfield_to_cpu(field.values, backend)
            for value ∈ values_cpu
                println(io,value)
            end
        elseif field_type <: VectorField
            write(io,"     <DataArray type=\"$(F32)\" Name=\"$(vector)\" format=\"$(format)\">\n")
            x_cpu, y_cpu, z_cpu = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
            for i ∈ eachindex(x_cpu)
                println(io, x_cpu[i]," ",y_cpu[i] ," ",z_cpu[i] )
            end
        else
            throw("""
            Input data should be a ScalarField or VectorField e.g. ("U", U)
            """)
        end


    write(io,"    </CellData>\n")
    write(io,"   </Piece>\n")
    write(io,"  </UnstructuredGrid>\n")
    write(io," </VTKFile>\n")
end
