function compute_geometry!(mesh)
    return Mesh.compute_3d_geometry!(mesh; orient_faces=false) # OpenFOAM defines the orientation
end
