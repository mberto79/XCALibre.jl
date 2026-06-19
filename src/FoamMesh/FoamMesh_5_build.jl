export FOAM3D_mesh

"""
    FOAM3D_mesh(mesh_file; scale=1, integer_type=Int64, float_type=Float64)

Read and convert 3D OpenFOAM mesh file into XCALibre.jl. Note that, at present, it is not recommended to run 2D cases using meshes imported using this function.

### Input

- `mesh_file` -- path to mesh file.

### Optional arguments

- `scale` -- used to scale mesh file e.g. scale=0.001 will convert mesh from mm to metres defaults to 1 i.e. no scaling

- `integer_type` - select interger type to use in the mesh (Int32 may be useful on GPU runs) 

- `float_type` - select interger type to use in the mesh (Float32 may be useful on GPU runs) 

"""
function FOAM3D_mesh(mesh_file; scale=1, integer_type=Int64, float_type=Float64)
    foamdata = read_FOAM3D(mesh_file, scale, integer_type, Float64)
    connectivity = connect_mesh(foamdata, integer_type, Float64)
    mesh = generate_mesh(foamdata, connectivity, integer_type, Float64)
    mesh = compute_geometry!(mesh)
    return Mesh.convert_mesh_float(mesh, float_type)
end
