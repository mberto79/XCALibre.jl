# Does the Laplacian face geometry reduce to one per-face constant?
using XCALibre, JLD2, LinearAlgebra, Printf, Statistics
mesh = load_object("/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/XCALibre/mesh.jld2")
faces = mesh.faces
nb = length(mesh.boundary_cellsID)

orig(face, ns) = begin
    (; area, normal, delta, e) = face
    Sf = ns*area*normal
    ee = ns*e
    Ef = ((Sf⋅Sf)/(Sf⋅ee))*ee
    norm(Ef)/delta
end
new(face) = face.area/(abs(face.normal ⋅ face.e)*face.delta)

dn = Float64[]; de = Float64[]; rel = Float64[]
for fID in (nb+1):length(faces)          # internal faces only
    f = faces[fID]
    push!(dn, abs(norm(f.normal) - 1)); push!(de, abs(norm(f.e) - 1))
    for ns in (1, -1)
        a = orig(f, ns); b = new(f)
        push!(rel, abs(a - b)/max(abs(a), eps()))
    end
end
@printf("internal faces      = %d\n", length(dn))
@printf("max |‖normal‖-1|    = %.3e\n", maximum(dn))
@printf("max |‖e‖-1|         = %.3e\n", maximum(de))
@printf("max rel diff gDiff  = %.3e\n", maximum(rel))
@printf("mean rel diff gDiff = %.3e\n", mean(rel))
