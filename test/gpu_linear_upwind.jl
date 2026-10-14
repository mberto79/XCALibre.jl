# CPU/GPU equality of the LinearUpwind / LimitedLinearUpwind / LUSTDC assembly (not part of runtests.jl:
# needs a GPU). Run with an environment that has XCALibre and CUDA (or AMDGPU, oneAPI):
#   julia --project=<env> test/gpu_linear_upwind.jl
using XCALibre, Test, CUDA, Printf
using LinearAlgebra
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
host = UNV2D_mesh(joinpath(grids_dir, "backwardFacingStep_5mm.unv"), scale=0.001)

function assemble(backend, scheme, vector, limiter=CellBased())
    mesh = adapt(backend, host)
    config = Configuration(solvers=nothing, schemes=nothing, boundaries=nothing,
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1))
    n = length(mesh.cells)
    cx = [c.centre for c in host.cells]
    mdotf = FaceScalarField(mesh)
    fc = [f.centre for f in host.faces]
    copyto!(mdotf.values, [1e-3*(0.4 + sin(300x)*cos(500y)) for (x, y, _) in fc])
    if vector
        U = VectorField(mesh)
        copyto!(U.x.values, [sin(40x)*cos(60y) for (x, y, _) in cx])
        copyto!(U.y.values, [x*y*1e3 for (x, y, _) in cx])
        grad = Grad{Gauss}(U)
        BCs = assign(region=mesh, (U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],)).U
        grad!(grad, FaceVectorField(mesh), U, BCs, 0.0, config)
        limit_gradient!(limiter, grad, U, config)
        eqn = (Divergence{scheme}(mdotf, U, grad.result) == Source(VectorField(mesh))) → VectorEquation(U, BCs)
        discretise!(eqn, U, config)
        return Array(eqn.equation.bx), Array(eqn.equation.by)
    else
        phi = ScalarField(mesh)
        copyto!(phi.values, [cos(35x)*sin(55y) for (x, y, _) in cx])
        grad = Grad{Gauss}(phi)
        BCs = assign(region=mesh, (phi = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet),
            Zerogradient(:wall), Zerogradient(:top)],)).phi
        grad!(grad, FaceScalarField(mesh), phi, BCs, 0.0, config)
        limit_gradient!(limiter, grad, phi, config)
        eqn = (Divergence{scheme}(mdotf, phi, grad.result) == Source(ScalarField(mesh))) → ScalarEquation(phi, BCs)
        discretise!(eqn, phi, config)
        return Array(eqn.equation.b), nothing
    end
end

@testset "CPU/GPU assembly equality" begin
    for scheme in (LinearUpwind, LimitedLinearUpwind, LUSTDC), vector in (false, true),
        limiter in (CellBased(), Venkatakrishnan(5.0))
        scheme === LUSTDC && !(limiter isa CellBased) && continue # LUSTDC uses no gradient
        c = assemble(CPU(), scheme, vector, limiter)
        g = assemble(CUDABackend(), scheme, vector, limiter)
        for (a, b) in zip(c, g)
            a === nothing && continue
            err = norm(a .- b)/norm(a)
            @printf("%s %s %s rel. diff %.2e\n", scheme, vector ? "vector" : "scalar", nameof(typeof(limiter)), err)
            @test err < 1e-12
        end
    end
end
