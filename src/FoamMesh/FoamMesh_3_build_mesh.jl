export build_foamMesh

function build_foamMesh(file_path; integer=Int64, float=Float64)

    points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace = read_foamMesh(file_path, integer, float)

    out = connect_mesh(
        points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace, integer, float
        )
    return out
end