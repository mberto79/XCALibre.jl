export FOAM3D_mesh

function FOAM3D_mesh(mesh_file; scale=1, integer_type=Int64, float_type=Float64)

    foamdata = read_FOAM3D(mesh_file, scale, integer_type, float_type)
    connectivity = connect_mesh(foamdata, integer_type, float_type)
    mesh = generate_mesh(foamdata, connectivity, integer_type, float_type)
    mesh = compute_geometry!(mesh)

    return mesh
    # return foamdata
end