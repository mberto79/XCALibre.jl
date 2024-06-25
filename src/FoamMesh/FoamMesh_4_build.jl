export load_foamMesh

function load_foamMesh(mesh_file; scale=1, integer_type=Int64, float_type=Float64)

    foamdata = read_foamMesh(mesh_file, integer_type, float_type)
    connectivity = connect_mesh(foamdata, integer_type, float_type)
    mesh = generate_mesh(foamdata, connectivity, integer_type, float_type)
    mesh = compute_geometry!(mesh)

    return mesh
end