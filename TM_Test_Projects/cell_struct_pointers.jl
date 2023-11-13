struct Cell{I,F}
    nodes::SVector{2,I} # 2 integers - 1 for the number of nodes and the 2nd give a pointer to a "cellNodes" array
    faces::SVector{2,I}  # 2 integers - 1 for the number of face and the 2nd give a pointer to a "cellFaces" array
    neighbours::SVector{2,I} # 2 integers - 1 for the number of nodes and the 2nd give a pointer to a "cellNeighbours" array
    nsign::SVector{2,I} # 2 integers - 1 for the number of nodes and the 2nd give a pointer to a "cellFaceSign" array
    centre::SVector{3, F}
    volume::F
end

# Simultaneously, the top level "Mesh" struct would be modified to include the required arrays e.g.

struct Mesh{I,F} <: AbstractMesh
    # cells
    cells::Vector{Cell{I,F}}
    cellNodes::Vector{I} # this array has all the nodesID stored in order
    cellFaces::Vector{I} # this array has all the facedID stored in order
    cellNeighbours::Vector{I} # this array has all the neighbours stored in order
    cellFaceSign::Vector{I} # ... and so on!

    # faces
    faces::Vector{Face2D{I,F}}
    faceNodesID::Vector{I}
    faceOwnerCells::Vector{I}
    faceCentre::Vector{I}
    faceNormal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F2D{I,F}

    # boundaries
    boundaries::Vector{Boundary{I}}
    

    # nodes
    nodes::Vector{Node{I,F}}
    ...
    ...
end