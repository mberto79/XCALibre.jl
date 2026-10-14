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

# both grids span a 1000 x 1000 square with cells of about 25 units
trig = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv")))
quad = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "quad40.unv")))

interior_cells(mesh) = setdiff(1:length(mesh.cells), unique(mesh.boundary_cellsID))

# limiter factor per cell: |limited gradient| / |raw gradient| (cells with zero gradient report 1)
function limiter_factors(method, mesh, values; TF=Float64)
    phi = ScalarField(mesh)
    phi.values .= values
    g = Grad{Gauss}(phi)
    grad!(g, FaceScalarField(mesh), phi, assign(region=mesh, (phi = [
        Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:bottom), Zerogradient(:top)],)).phi,
        zero(TF), config)
    raw = [g.result[i] for i ∈ eachindex(mesh.cells)]
    limit_gradient!(method, g, phi, config)
    lim = [g.result[i] for i ∈ eachindex(mesh.cells)]
    [norm(r) > 0 ? norm(l)/norm(r) : one(TF) for (r, l) ∈ zip(raw, lim)], lim
end

# a linear field with its exact gradient set by hand, so only the limiter acts
function linear_factors(method, mesh)
    a = SVector(0.3e-3, -0.7e-3, 0.0)
    phi = ScalarField(mesh)
    phi.values .= [a⋅c + 2.0 for c ∈ mesh.cells.centre]
    g = Grad{Gauss}(phi)
    for i ∈ eachindex(mesh.cells)
        g.result.x[i], g.result.y[i], g.result.z[i] = a
    end
    limit_gradient!(method, g, phi, config)
    [g.result[i]⋅a/(a⋅a) for i ∈ eachindex(mesh.cells)]
end

step_values(mesh) = [c[1] < 500 ? 1.0 : 0.0 for c ∈ mesh.cells.centre]
ramp_values(mesh) = [clamp((c[1] - 400)/200, 0.0, 1.0) for c ∈ mesh.cells.centre] # step over ~8 cells
wave_values(mesh) = [sin(c[1]/90)*cos(c[2]/130) for c ∈ mesh.cells.centre]

@testset "Venkatakrishnan limiter" begin
    @testset "constructor" begin
        @test Venkatakrishnan().K == 5.0
        @test Venkatakrishnan(1).K === 1.0
        @test Venkatakrishnan(0.5f0).K === 0.5f0
        @test_throws ErrorException Venkatakrishnan(-1.0)
    end

    @testset "factor in [0, 1], gradient direction kept" begin
        for mesh ∈ (trig, quad), values ∈ (step_values(mesh), wave_values(mesh)), K ∈ (0.0, 5.0)
            psi, _ = limiter_factors(Venkatakrishnan(K), mesh, values)
            @test all(p -> 0 ≤ p ≤ 1 + 1e-12, psi)
        end
    end

    @testset "linear field" begin
        # uniform quads: every active face has an allowed/extrapolated ratio of at least 2, so the
        # gradient is kept for any K
        for K ∈ (0.0, 5.0)
            @test all(isapprox.(linear_factors(Venkatakrishnan(K), quad)[interior_cells(quad)], 1; atol=1e-12))
        end
        # triangles: face centres are not halfway to the neighbour, so K = 0 limits a linear field
        # (unlike Barth-Jespersen); ε removes the limiting, monotonically in K
        interior = interior_cells(trig)
        @test all(isapprox.(linear_factors(CellBased(), trig)[interior], 1; atol=1e-12))
        mins = [minimum(linear_factors(Venkatakrishnan(K), trig)[interior]) for K ∈ (0.0, 0.01, 0.1, 1.0)]
        @test mins[1] < 0.95
        @test issorted(mins)
        @test mins[end] > 1 - 1e-6
    end

    @testset "smooth vs Barth-Jespersen on a step" begin
        for mesh ∈ (trig, quad)
            # with ε = 0 the smooth function lies below min(1, ratio): never less limiting than BJ,
            # and strictly below it where the ratio is under 2 (ends of the smeared step)
            for values ∈ (step_values(mesh), ramp_values(mesh))
                bj, _ = limiter_factors(CellBased(), mesh, values)
                v0, _ = limiter_factors(Venkatakrishnan(0.0), mesh, values)
                @test all(v0 .≤ bj .+ 1e-12)
            end
            bj, _ = limiter_factors(CellBased(), mesh, ramp_values(mesh))
            v0, _ = limiter_factors(Venkatakrishnan(0.0), mesh, ramp_values(mesh))
            @test count(v0 .< bj .- 1e-3) > 0
            values = step_values(mesh)
            bj, _ = limiter_factors(CellBased(), mesh, values)
            # cells at the step are local extrema: BJ zeroes their gradient, ε > 0 keeps part of it
            # (ε² = (K h)³ with h ≈ 25 here, so K = 0.1 gives ε² ≈ 16 against a jump of 1)
            vK, _ = limiter_factors(Venkatakrishnan(0.1), mesh, values)
            extrema = findall(p -> p < 1e-12, bj)
            @test !isempty(extrema)
            @test all(vK[extrema] .> 0.5)
        end
    end

    @testset "vector field limits each component like the scalar" begin
        mesh = trig
        U = VectorField(mesh)
        U.x.values .= step_values(mesh)
        U.y.values .= wave_values(mesh)
        gU = Grad{Gauss}(U)
        grad!(gU, FaceVectorField(mesh), U, assign(region=mesh, (U = [
            Extrapolated(:inlet), Extrapolated(:outlet), Extrapolated(:bottom), Extrapolated(:top)],)).U,
            0.0, config)
        limit_gradient!(Venkatakrishnan(), gU, U, config)
        _, limx = limiter_factors(Venkatakrishnan(), mesh, U.x.values)
        _, limy = limiter_factors(Venkatakrishnan(), mesh, U.y.values)
        @test all(i -> gU.result[i][1, :] ≈ limx[i], eachindex(mesh.cells))
        @test all(i -> gU.result[i][2, :] ≈ limy[i], eachindex(mesh.cells))
    end

    @testset "mesh float type preserved ($TF, $TI)" for (TF, TI) ∈ ((Float32, Int32), (Float64, Int64))
        m = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv"), float_type=TF, integer_type=TI))
        psi, lim = limiter_factors(Venkatakrishnan(5.0), m, TF.(step_values(m)); TF=TF)
        @test eltype(eltype(lim)) == TF
        @test all(p -> 0 ≤ p ≤ 1 + 10eps(TF), psi)
        @test all(v -> all(isfinite, v), lim)
    end
end
