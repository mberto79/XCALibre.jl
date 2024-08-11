using FVM_1D
using LinearAlgebra

mesh_file = "unv_sample_meshes/BFS_UNV_3D_hex_5mm.unv"

mesh = UNV3D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Incompressible(nu = ConstantScalar(nu)),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev
    )

function construct_periodic(
    # model, patch1::Symbol, patch2::Symbol; translation::Number, direction::Vector{<:Number}
    model, patch1::Symbol, patch2::Symbol
    )

    (; faces, boundaries) = model.domain

    boundary_information = boundary_map(model.domain)
    idx1 = boundary_index(boundary_information, patch1)
    idx2 = boundary_index(boundary_information, patch2)

    face1 = faces[boundaries[idx1].IDs_range[1]]
    face2 = faces[boundaries[idx2].IDs_range[1]]
    distance = abs((face1.centre - face2.centre)⋅face1.normal)

    nfaces = length(boundaries[idx1].IDs_range)
    faceAddress1 = zeros(Int64, nfaces)
    faceAddress2 = zeros(Int64, nfaces)
    testData = zeros(Float64, nfaces)

    faceCentres1 = getproperty.(faces[boundaries[idx1].IDs_range], :centre)
    faceCentres2 = getproperty.(faces[boundaries[idx2].IDs_range], :centre)

    patchTranslated1 = faceCentres1 .- [distance*face1.normal]
    for (i, face) ∈ enumerate(faces[boundaries[idx2].IDs_range])
        testData .= norm.(patchTranslated1 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress1[i] = boundaries[idx2].IDs_range[idx]
    end

    patchTranslated2 = faceCentres2 .- [distance*face2.normal]
    for (i, face) ∈ enumerate(faces[boundaries[idx1].IDs_range])
        testData .= norm.(patchTranslated2 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress2[i] = boundaries[idx1].IDs_range[idx]
    end

    values1 = (distance=distance, face_map=faceAddress1)
    values2 = (distance=distance, face_map=faceAddress2)
    periodic1 = Periodic(patch1, values1)
    periodic2 = Periodic(patch2, values2)
    return periodic1, periodic2
end

(; faces, boundaries) = model.domain

fcs1 = getproperty.(faces[boundaries[5].IDs_range], :centre)
fcs2 = getproperty.(faces[boundaries[6].IDs_range], :centre)

face1 = faces[boundaries[5].IDs_range[1]]
face2 = faces[boundaries[6].IDs_range[1]]

distance = abs((face1.centre - face2.centre)⋅face1.normal)

normal1 = face1.normal
normal2 = face2.normal

faceTranslated1 = face1.centre - distance*face1.normal
face2.centre - faceTranslated1
face2.centre - distance*face2.normal
face1.centre
face2.centre

patchTranslated = fcs1 .- [distance*face1.normal]
testData = norm.(patchTranslated .- [face2.centre])
val, idx = findmin(testData)

(face1.centre - face2.centre)
norm(fc1 - fc2)

# periodic1, periodic2 = construct_periodic(model, :side1, :side2, translation=0.2, direction=[0,0,1])
periodic1, periodic2 = construct_periodic(model, :side1, :side2)

boundaries[5].IDs_range[120]
periodic1.value.face_map[120]

face1 = faces[boundaries[5].IDs_range[120]].centre
face2 = faces[periodic1.value.face_map[120]].centre

boundaries[5].IDs_range[120]
periodic1.value.face_map[120]

boundaries[6].IDs_range
boundaries[6].IDs_range[700]
periodic2.value.face_map[700]

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0]),
    periodic1, periodic2
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    periodic1, periodic2
)