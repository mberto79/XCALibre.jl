using StaticArrays
using CUDA
using FVM_1D

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(unv_mesh)