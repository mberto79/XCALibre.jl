using XCALibre
using LinearAlgebra

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "3d_box_1000x1000x1000mm_5.unv"
# grid = "cascade_3D_periodic_4mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)

edge_length = 1/5 
face_area = edge_length^2
cell_volume = edge_length^3

getproperty.(mesh.cells, :volume) 
getproperty.(mesh.faces, :area) 

patch_area = 1.0*1.0

area_patches = typeof(patch_area)[]
for boundary ∈ mesh.boundaries 
    faces = mesh.faces 
    areai = zero(patch_area)
    for fID ∈ boundary.IDs_range 
        face = faces[fID]
        areai += face.area 
    end
    push!(area_patches, areai)
end

area_patches

volume = 0.0
for cell ∈ mesh.cells
    volume += cell.volume 
end
volume

norms = zeros(length(mesh.faces))
for (fi, face) ∈ enumerate(mesh.faces)
    mag = norm(face.normal)
    norms[fi] = mag
end


    
    