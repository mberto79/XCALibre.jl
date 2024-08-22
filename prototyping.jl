using FVM_1D
using FVM_1D.ModelFramework
using SparseArrays
using LinearAlgebra

mesh_file = "unv_sample_meshes/cascade_3D_periodic_2p5mm.unv"
mesh = UNV3D_mesh(mesh_file, scale=0.001)

periodic, connectivity = construct_periodic(mesh, CPU(), :top, :bottom)

connectivity = Discretise.periodic_matrix_connectivity(mesh, periodic...)
Ac = sparse(connectivity.i, connectivity.j, zeros(Float64, length(connectivity.i)))

eqn = ScalarEquation(mesh, connectivity)

(; colptr, rowval) = eqn.A
# faces: 6321 9521
nzcellID = spindex(colptr, rowval, 3161, 7881)
nzcellID = spindex(colptr, rowval, 7881, 3161)

# face i = 1: 6480 3280
# cell i = 1: 3161 7881

fID1 = periodic[1].value.face_map[3200]
fID2 = periodic[2].value.face_map[3200]

face1 = mesh.faces[fID1]
face2 = mesh.faces[fID2]
cID1 = face1.ownerCells[1]
cID2 = face2.ownerCells[1]
cell1 = mesh.cells[cID1]
cell2 = mesh.cells[cID2]

distance = norm((face1.centre - face2.centre)⋅face1.normal)
distance = ((face1.centre - face2.centre))
distance = face1.centre - face2.centre⋅face1.normal

cell1.centre
cell2.centre
cell1.centre - cell2.centre

distance = norm((cell1.centre - cell2.centre)⋅face1.normal)


nzcellID = spindex(colptr, rowval, cID1, cID2)

ip, jp = periodic_matrix_connectivity(mesh, top, bottom)


ic = [i; ip]
jc = [j; jp]
vc = zeros(Float64, length(ic))

Ac = sparse(ic, jc, vc)

I, J, V = findnz(Ac)

Ac1 = sparse(I,J,V)




using Accessors
IDs_range = mesh.boundaries[4].IDs_range
mesh.faces[IDs_range[1]].normal
# flip normals
faces = mesh.faces
face.normal
face = faces[IDs_range[1]]
@reset face.normal .+= 2
face.normal

faces = mesh.faces

boundary_information = boundary_map(mesh)
idx1 = boundary_index(boundary_information, periodic[1].ID)
idx2 = boundary_index(boundary_information, periodic[2].ID)
IDs_range = mesh.boundaries[idx2].IDs_range

for fID ∈ IDs_range
    face = faces[fID]  
    @reset face.normal *= -1 
    faces[fID] = face
end