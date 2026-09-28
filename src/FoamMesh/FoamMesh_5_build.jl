export FOAM3D_mesh

"""
    FOAM3D_mesh(mesh_file; scale=1, integer_type=Int32, float_type=Float64)

Read and convert 3D OpenFOAM mesh file into XCALibre.jl. Note that, at present, it is not recommended to run 2D cases using meshes imported using this function.

### Input

- `mesh_file` -- path to mesh file.

### Optional arguments

- `scale` -- used to scale mesh file e.g. scale=0.001 will convert mesh from mm to metres defaults to 1 i.e. no scaling

- `integer_type` - integer type of the mesh indices; `Int64` is needed only when a mesh has more than 2^31 faces, face-node entries or matrix entries, and reading such a mesh as `Int32` stops with an error saying so

- `float_type` - select interger type to use in the mesh (Float32 may be useful on GPU runs) 

"""
function FOAM3D_mesh(mesh_file; scale=1, integer_type=Int32, float_type=Float64)
    mesh = _with_index_capacity(integer_type) do
        foamdata = read_FOAM3D(mesh_file, scale, integer_type, Float64)
        connectivity = connect_mesh(foamdata, integer_type, Float64)
        generate_mesh(foamdata, connectivity, integer_type, Float64)
    end
    mesh = compute_geometry!(mesh)
    return Mesh.convert_mesh_float(mesh, float_type)
end
