# Int32 copy of the benchmark mesh, written into dev/komega. The polyMesh is read only;
# nothing in the benchmark directory is touched.
using XCALibre, JLD2
poly = "/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/OpenFOAM/constant/polyMesh"
out = joinpath(@__DIR__, "mesh_i32.jld2")
m = FOAM3D_mesh(poly, scale=1, integer_type=Int32)
save_object(out, m)
println("wrote $out  cells=", length(m.cells), " int=", eltype(m.cell_faces))
