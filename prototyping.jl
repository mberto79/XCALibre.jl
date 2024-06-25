using FVM_1D
using FVM_1D.FoamMesh
using StaticArrays

mesh_file = "unv_sample_meshes/OF_cavity_hex/constant/polyMesh"

mesh = load_foamMesh(mesh_file, integer_type=Int64, float_type=Float64)

field = ScalarField(mesh)
field.values .= 1:length(field.values)
field.values
@time write_vtk("foamMeshTest", mesh, ("F", field))