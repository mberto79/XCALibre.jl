# CPU/GPU equality of the transpose stress and viscous work with DirichletFunction, RotatingWall,
# Outlet and periodic faces (not in runtests.jl: needs a GPU). Run with an environment that has
# XCALibre and CUDA: julia --project=<env> test/gpu_boundary_sngrad.jl
using XCALibre, Test, CUDA, StaticArrays

CUDA.functional() || (println("SKIP gpu_boundary_sngrad: CUDA not functional"); exit(0))

const TS = XCALibre.Solvers
const MP = XCALibre.ModelPhysics
grids = pkgdir(XCALibre, "examples/0_GRIDS")

inflow(x, t, i) = SVector{3}((1 + t)*4x[2]*(1 - x[2]), 0.1*t*x[2], 0.0)

function sngrad_terms(backend, TF; periodic=false)
    host = UNV2D_mesh(joinpath(grids, "quad40.unv"), scale=0.001, float_type=TF)
    n = [b.name for b ∈ host.boundaries] # inlet, outlet, bottom, top
    mesh = adapt(backend, host)
    sides = periodic ? construct_periodic(host, backend, n[3], n[4]) :
        (DirichletFunction(n[3], (x, t, i) -> SVector{3}(0.0, 0.0, 0.0)),
         RotatingWall(n[4]; centre=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], rpm=30.0))
    BCs = assign(region=mesh, (U=[DirichletFunction(n[1], inflow), Outlet(n[2]), sides...],))
    config = Configuration(
        solvers=(U=SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
        schemes=(U=Schemes(),), runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=32), boundaries=BCs)
    centres = [c.centre for c ∈ host.cells]
    hU = VectorField(host)
    hU.x.values .= [sin(4x)*cos(6y) + 3y for (x, y, _) ∈ centres]
    hU.y.values .= [x*y for (x, y, _) ∈ centres]
    hnuc = ScalarField(host); hnuc.values .= [1 + 3x + 5y^2 for (x, y, _) ∈ centres]
    hnuf = FaceScalarField(host); hnuf.values .= [1 + 3f.centre[1] + 5f.centre[2]^2 for f ∈ host.faces]
    hmdot = FaceScalarField(host); hmdot.values .= [f.normal[1] + f.normal[2] for f ∈ host.faces]
    U, nuc, nuf, mdotf = adapt(backend, hU), adapt(backend, hnuc), adapt(backend, hnuf), adapt(backend, hmdot)
    t = TF(0.5)
    gradU = Grad{Gauss}(U); Uf = FaceVectorField(mesh)
    grad!(gradU, Uf, U, BCs.U, t, config)
    source = VectorField(mesh)
    TS.transpose_stress!(source, nuf, gradU, BCs.U, t, config; cell_mueff=TS.cell_nueff(nuc, nothing))
    work = ScalarField(mesh)
    MP.viscous_work!(work, nuf, nuc, U, Uf, gradU, mdotf, BCs.U, t, config)
    Array(source.x.values), Array(source.y.values), Array(work.values)
end

@testset "Boundary-face gradient, CPU vs GPU" begin
    for TF ∈ (Float64, Float32), periodic ∈ (false, true)
        cpu = sngrad_terms(CPU(), TF; periodic)
        gpu = sngrad_terms(CUDABackend(), TF; periodic)
        rtol = TF === Float64 ? 1e-10 : 1e-4
        for (c, g) ∈ zip(cpu, gpu)
            @test eltype(g) === TF
            @test maximum(abs.(c .- g)) <= rtol*maximum(abs, c)
        end
    end
end
