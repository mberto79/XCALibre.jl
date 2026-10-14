using XCALibre
using Test
using StaticArrays
using LinearAlgebra

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

# fields of an analytic function: cell values at the centres, boundary faces at the face centres,
# internal faces linearly interpolated (as `Gauss` sees them)
function analytic_fields(mesh, f)
    nb = length(mesh.boundary_cellsID)
    phi = ScalarField(mesh)
    phi.values .= [f(c) for c ∈ mesh.cells.centre]
    phif = FaceScalarField(mesh)
    interpolate!(phif, phi, config)
    for i ∈ 1:nb
        phif.values[i] = f(mesh.faces.centre[i])
    end
    phi, phif
end

function gradient_error(scheme, mesh, f, gradf)
    phi, phif = analytic_fields(mesh, f)
    g = Grad{scheme}(phi)
    grad!(g, phif, phi, 0.0, config)
    maximum(norm(g.result[i] - gradf(mesh.cells.centre[i])) for i ∈ eachindex(mesh.cells))
end

lin(c) = 1.3 + 2.0c[1] - 0.7c[2] + 0.4c[3]
lin_grad(c) = SVector(2.0, -0.7, 0.4)
lin2d(c) = 1.3 + 2.0c[1] - 0.7c[2]            # 2D meshes have no z-variation to resolve
lin2d_grad(c) = SVector(2.0, -0.7, 0.0)

@testset "Least-squares gradient" begin
    meshes = (
        "skewed triangles (2D)" => UNV2D_mesh(joinpath(grids_dir, "trig40.unv")),
        "stretched hex (pitzDaily)" =>
            FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily", "polyMesh"), scale=1),
        "tetrahedra (3D)" => UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_15mm.unv"), scale=0.001),
    )

    @testset "exact on linear fields: $name" for (name, mesh) ∈ meshes
        f, df = name == "tetrahedra (3D)" ? (lin, lin_grad) : (lin2d, lin2d_grad)
        scale = norm(df(nothing))
        for S ∈ (LeastSquares, WeightedLeastSquares)
            @test gradient_error(S, mesh, f, df) < 1e-9*scale
        end
        # Green-Gauss with linearly interpolated face values is not exact on skewed meshes
        if name != "stretched hex (pitzDaily)"
            @test gradient_error(Gauss, mesh, f, df) > 1e-3*scale
        end
    end

    @testset "vector field -> tensor, BC-less and with BCs" begin
        mesh = meshes[1].second
        A = SMatrix{3,3}(0.5, -1.0, 0.0, 2.0, 0.3, 0.0, 0.0, 0.0, 0.0) # A[a,b] = d u_a / d x_b
        u(c) = SVector(1.0, -2.0, 0.0) + A*c
        U = VectorField(mesh); Uf = FaceVectorField(mesh)
        for i ∈ eachindex(mesh.cells)
            v = u(mesh.cells.centre[i]); U.x[i] = v[1]; U.y[i] = v[2]; U.z[i] = v[3]
        end
        interpolate!(Uf, U, config)
        for i ∈ 1:length(mesh.boundary_cellsID)
            v = u(mesh.faces.centre[i]); Uf.x[i] = v[1]; Uf.y[i] = v[2]; Uf.z[i] = v[3]
        end
        for S ∈ (LeastSquares, WeightedLeastSquares)
            g = Grad{S}(U)
            grad!(g, Uf, U, 0.0, config)
            @test maximum(norm(g.result[i] - A) for i ∈ eachindex(mesh.cells)) < 1e-9
        end
        # same result as the scalar gradient of each component
        g = Grad{LeastSquares}(U); grad!(g, Uf, U, 0.0, config)
        s = ScalarField(mesh); sf = FaceScalarField(mesh)
        s.values .= U.x.values; sf.values .= Uf.x.values
        gs = Grad{LeastSquares}(s); grad!(gs, sf, s, 0.0, config)
        @test all(g.result[i][1,:] ≈ gs.result[i] for i ∈ eachindex(mesh.cells))
    end

    @testset "boundary conditions are used" begin
        mesh = meshes[1].second
        phi, phif = analytic_fields(mesh, lin2d)
        bcs = assign(region=mesh, (phi = [
            Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0),
            Dirichlet(:bottom, 0.0), Dirichlet(:top, 0.0)],)).phi
        g = Grad{LeastSquares}(phi)
        grad!(g, phif, phi, bcs, 0.0, config)   # boundary value 0 from the BCs
        phif0 = FaceScalarField(mesh); phif0.values .= phif.values
        phif0.values[1:length(mesh.boundary_cellsID)] .= 0.0
        g0 = Grad{LeastSquares}(phi)
        grad!(g0, phif0, phi, 0.0, config)       # same through the face field
        @test all(g.result[i] == g0.result[i] for i ∈ eachindex(mesh.cells))
        @test maximum(norm(g.result[i] - lin2d_grad(nothing)) for i ∈ eachindex(mesh.cells)) > 1e-3
    end

    @testset "empty patches add no z-variation" begin
        mesh = meshes[2].second
        phi, phif = analytic_fields(mesh, lin2d)
        bcs = assign(region=mesh, (phi = [
            Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:upperWall),
            Zerogradient(:lowerWall), Empty(:frontAndBack)],)).phi
        g = Grad{LeastSquares}(phi)
        grad!(g, phif, phi, bcs, 0.0, config)
        @test all(isfinite, g.result.x.values)
        @test maximum(abs, g.result.z.values) < 1e-9
    end

    @testset "agrees with Gauss on a smooth field, second-order convergence" begin
        fs(c) = sin(2c[1]/1000)*cos(1.5c[2]/1000)
        gs(c) = SVector(2/1000*cos(2c[1]/1000)*cos(1.5c[2]/1000),
                       -1.5/1000*sin(2c[1]/1000)*sin(1.5c[2]/1000), 0.0)
        errs = Float64[]
        for name ∈ ("trig40.unv", "trig100.unv")
            mesh = UNV2D_mesh(joinpath(grids_dir, name))
            push!(errs, gradient_error(LeastSquares, mesh, fs, gs))
            @test gradient_error(WeightedLeastSquares, mesh, fs, gs) < 3*errs[end]
            @test errs[end] < 2*gradient_error(Gauss, mesh, fs, gs)
        end
        @test errs[2] < errs[1]/1.5  # refined mesh (40 -> 100 cells per side) is more accurate
    end

    @testset "Float32 mesh keeps Float32 results" begin
        mesh = UNV2D_mesh(joinpath(grids_dir, "trig40.unv"), float_type=Float32)
        phi = ScalarField(mesh); phif = FaceScalarField(mesh)
        phi.values .= Float32[lin(c) for c ∈ mesh.cells.centre]
        interpolate!(phif, phi, config)
        g = Grad{LeastSquares}(phi)
        grad!(g, phif, phi, 0.0f0, config)
        @test eltype(g.result.x.values) == Float32
        @test all(isfinite, g.result.x.values)
    end
end
