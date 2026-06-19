using XCALibre
using CUDA
import AcceleratedKernels as AK

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig.unv"
grid = "trig40.unv"
grid = "trig100.unv"
# grid = "quad100.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU()

f1 = FaceScalarField(mesh)

f1GPU = adapt(CUDABackend(), f1)

function f!(f)
    AK.foreachindex(f.values, min_elems=7962, block_size=32) do i
        @inbounds f[i] = 1 + 10*f[i] + f[i] + test_func(f, i)
    end
end

@time f!(f1)
@time f!(f1GPU)

@time @. f1.values = 1 + 10*f1.values + f1.values + test_func2(f1)

test_func(f, i) = begin
    mesh = f.mesh 
    faces = mesh.faces 
    ownerCells = faces[i].ownerCells
    cID = ownerCells[1] + ownerCells[2]
end

test_func2(f) = begin
    output = zeros(length(f))
    mesh = f.mesh 
    faces = mesh.faces 
    for i ∈ eachindex(f.values)
        ownerCells = faces[i].ownerCells
        output[i] = ownerCells[1] + ownerCells[2]
    end
    output
end


s = similar(ZGPU)