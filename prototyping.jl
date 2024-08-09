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
    model, patch1::Symbol, patch2::Symbol; translation::Number, direction::Vector{<:Number}
    )

    boundary_information = boundary_map(model.domain)
    idx1 = boundary_index(boundary_information, patch1)
    idx2 = boundary_index(boundary_information, patch2)

    faceCentres1 = getproperty.(mesh.faces[mesh.boundaries[idx1].IDs_range], :centre)
    faceCentres2 = getproperty.(mesh.faces[mesh.boundaries[idx2].IDs_range], :centre)

    values1 = 1
    values2 = 2
    periodic1 = Periodic(primary, values1)
    periodic2 = Periodic(secondary, values2)
    return periodic1, periodic2
end

fcs1 = getproperty.(mesh.faces[mesh.boundaries[5].IDs_range], :centre)
fcs2 = getproperty.(mesh.faces[mesh.boundaries[6].IDs_range], :centre)

fc1 = mesh.faces[mesh.boundaries[5].IDs_range[1]].centre
fc2 = mesh.faces[mesh.boundaries[6].IDs_range[1]].centre

fc1 - fc2
fc2 - fc1
norm(fc1 - fc2)

periodic1, periodic2 = construct_periodic(model, :side1, :side2, translation=0.2, direction=[0,0,1])


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
    Neumann(:top, 0.0)
)