export build_foamMesh

function build_foamMesh(file_path; integer=Int64, float=Float64)

    foamdata = read_foamMesh(mesh_file, Int64, Float64)

    connectivity = connect_mesh(foamdata, Int64, Float64)

    mesh = generate_mesh(foamdata, connectivity, integer, float)
    
    return m
end