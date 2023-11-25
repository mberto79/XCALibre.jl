using Plots
using Krylov
using StaticArrays
using CUDA
using FVM_1D

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

## SCALAR FIELD
scalarField = ScalarField(mesh)

(; values, mesh, BCs) = scalarField

values = cu(values)
mesh = cu(mesh)
BCs = cu(BCs)

scalarField = ScalarField(values, mesh, BCs)

function test_kernel_SF!(scalarField)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; values) = scalarField
    
    @inbounds if i <= length(values) && i > 0

        values[i] = values[i] + values[i]

    end

    # ScalarField(values, scalarField.mesh, scalarField.BCs)

    return nothing

end

scalarField.values

@device_code_warntype @cuda threads = 1024 blocks = cld(length(scalarField.values),1024) test_kernel_SF!(scalarField)

scalarField.values