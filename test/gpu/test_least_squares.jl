# CPU/GPU equality of the LeastSquares gradient (not in a runtests.jl tier: needs a GPU).
#   julia --project=<env with XCALibre and CUDA> test/gpu/test_least_squares.jl
using XCALibre, Test, CUDA, Printf
using LinearAlgebra
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
meshes = (
    "2D BFS" => UNV2D_mesh(joinpath(grids_dir, "backwardFacingStep_5mm.unv"), scale=0.001),
    "3D tetrahedra" => UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_15mm.unv"), scale=0.001),
)

function gradient(backend, host, scheme, vector)
    mesh = adapt(backend, host)
    config = Configuration(solvers=nothing, schemes=nothing, boundaries=nothing,
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1))
    cx = [c.centre for c in host.cells]
    names = [b.name for b in host.boundaries]
    if vector
        U = VectorField(mesh)
        copyto!(U.x.values, [sin(40x)*cos(60y) + z for (x, y, z) in cx])
        copyto!(U.y.values, [x*y*1e3 for (x, y, z) in cx])
        copyto!(U.z.values, [cos(30z)*x for (x, y, z) in cx])
        BCs = assign(region=mesh, (U = [name == :inlet ? Dirichlet(:inlet, [1.0, 0.5, 0.0]) :
            name == :wall ? Wall(:wall, [0.0, 0.0, 0.0]) : Zerogradient(name) for name in names],)).U
        g = Grad{scheme}(U)
        grad!(g, FaceVectorField(mesh), U, BCs, 0.0, config)
        return [Array(getproperty(g.result, f).values) for f in (:xx, :xy, :xz, :yx, :yy, :yz, :zx, :zy, :zz)]
    else
        phi = ScalarField(mesh)
        copyto!(phi.values, [cos(35x)*sin(55y) + z*x for (x, y, z) in cx])
        BCs = assign(region=mesh, (phi = [name == :inlet ? Dirichlet(:inlet, 1.0) :
            Zerogradient(name) for name in names],)).phi
        g = Grad{scheme}(phi)
        grad!(g, FaceScalarField(mesh), phi, BCs, 0.0, config)
        return [Array(getproperty(g.result, f).values) for f in (:x, :y, :z)]
    end
end

@testset "CPU/GPU least-squares gradient equality" begin
    for (mname, host) in meshes, vector in (false, true)
        c = gradient(CPU(), host, LeastSquares, vector)
        g = gradient(CUDABackend(), host, LeastSquares, vector)
        err = maximum(norm(a .- b)/max(norm(a), eps()) for (a, b) in zip(c, g))
        @printf("%s %s rel. diff %.2e\n", mname, vector ? "vector" : "scalar", err)
        @test err < 1e-12
    end
end
