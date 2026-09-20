# Does faces[fID].ownerCells agree with the owner/neighbour implied by cell_nsign?
using XCALibre, JLD2, Printf
mesh = load_object("/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/XCALibre/mesh.jld2")
(; cells, cell_faces, cell_neighbours, cell_nsign, faces, boundary_cellsID) = mesh
nb = length(boundary_cellsID)
owner_from_nsign = zeros(Int, length(faces))
neig_from_nsign  = zeros(Int, length(faces))
for cID ∈ eachindex(cells), fi ∈ cells[cID].faces_range
    fID = cell_faces[fi]
    if cell_nsign[fi] > 0; owner_from_nsign[fID] = cID else neig_from_nsign[fID] = cID end
end
bad_o = 0; bad_n = 0; swapped = 0
for fID ∈ (nb+1):length(faces)
    oc = faces[fID].ownerCells
    owner_from_nsign[fID] == oc[1] || (bad_o += 1)
    neig_from_nsign[fID]  == oc[2] || (bad_n += 1)
    (owner_from_nsign[fID] == oc[2] && neig_from_nsign[fID] == oc[1]) && (swapped += 1)
end
n = length(faces) - nb
@printf("internal faces          = %d\n", n)
@printf("ownerCells[1] != owner  = %d\n", bad_o)
@printf("ownerCells[2] != neigh  = %d\n", bad_n)
@printf("exactly swapped         = %d\n", swapped)
# also: are cell_neighbours consistent with ownerCells?
bad_nb = 0
for cID ∈ eachindex(cells), fi ∈ cells[cID].faces_range
    fID = cell_faces[fi]; oc = faces[fID].ownerCells
    other = cell_neighbours[fi]
    (other == oc[1] || other == oc[2]) || (bad_nb += 1)
end
@printf("cell_neighbours not in ownerCells = %d\n", bad_nb)
