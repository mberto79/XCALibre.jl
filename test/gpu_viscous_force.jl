# CPU/GPU equality of viscous_force and wall_shear_stress on Wall, DirichletFunction and
# RotatingWall patches (not in runtests.jl: needs a GPU). Run with an environment that has
# XCALibre and CUDA: julia --project=<env> test/gpu_viscous_force.jl
using XCALibre, Test, CUDA, StaticArrays

CUDA.functional() || (println("SKIP gpu_viscous_force: CUDA not functional"); exit(0))

grids = pkgdir(XCALibre, "examples/0_GRIDS")
wallfun(x, t, i) = SVector{3}(t*0.3, -0.05*x[1], 0.0)

function shear(backend, TF, bottom)
    host = UNV2D_mesh(joinpath(grids, "quad40.unv"), scale=0.001, float_type=TF)
    mesh = adapt(backend, host)
    model = Physics(
        time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    BCs = assign(region=mesh, (U=[Zerogradient(:inlet), Zerogradient(:outlet), bottom, Wall(:top, [0.1, 0.0, 0.0])],))
    config = Configuration(
        solvers=(;), schemes=(;), runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=32), boundaries=BCs)
    centres = [c.centre for c ∈ host.cells]
    copyto!(model.momentum.U.x.values, TF[0.4 + 30x + 200y for (x, y, _) ∈ centres])
    copyto!(model.momentum.U.y.values, TF[-0.1 + 5e4x*y for (x, y, _) ∈ centres])
    t = TF(2)
    F = viscous_force(:bottom, model, config; time=t)
    Ftop = viscous_force(:top, model, config; time=t)
    tauw, pos = wall_shear_stress(:bottom, model, config; time=t)
    F, Ftop, Array(tauw.x.values), Array(tauw.y.values)
end

@testset "viscous_force and wall_shear_stress, CPU vs GPU" begin
    for TF ∈ (Float64, Float32), bottom ∈ (
            Wall(:bottom, [0.3, -0.05, 0.0]), DirichletFunction(:bottom, wallfun),
            RotatingWall(:bottom; centre=[0.0, -0.01, 0.0], axis=[0.0, 0.0, 1.0], rpm=300.0))
        cpu = shear(CPU(), TF, bottom)
        gpu = shear(CUDABackend(), TF, bottom)
        rtol = TF === Float64 ? 1e-10 : 1e-4
        for (c, g) ∈ zip(cpu, gpu)
            @test eltype(g) === TF
            @test maximum(abs.(c .- g)) <= rtol*maximum(abs, c)
        end
    end
end
