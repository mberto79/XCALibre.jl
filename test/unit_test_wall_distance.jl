using XCALibre
using KernelAbstractions
using Test
using LinearAlgebra

function test_normal_distance_clamp(backend)
    mesh_cpu = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "laplace_unit_3by3.unv"))
    mesh = adapt(backend, mesh_cpu)
    phi = ScalarField(mesh)
    y = ScalarField(mesh)
    grad = Grad{Gauss}(phi)
    phi.values .= -1.0
    config = (; hardware = Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32))

    XCALibre.Calculate.normal_distance!(y, phi, grad, config)
    KernelAbstractions.synchronize(backend)

    @test all(Array(y.values) .== 0.0)
end

function test_wall_distance_channel(backend)
    mesh_cpu = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "laplace_unit_3by3.unv"))
    mesh = adapt(backend, mesh_cpu)
    hardware = Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32)

    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1.0),
        turbulence = RANS{KOmegaSST}(walls=(:bottom_wall, :upper_wall)),
        energy = Energy{Isothermal}(),
        domain = mesh,
    )

    BCs = assign(
        region = mesh,
        (
            U = [
                Extrapolated(:left_wall),
                Extrapolated(:right_wall),
                Wall(:bottom_wall, [0.0, 0.0, 0.0]),
                Wall(:upper_wall, [0.0, 0.0, 0.0]),
            ],
        ),
    )

    config = Configuration(
        schemes = (y = Schemes(),),
        solvers = (
            y = SolverSetup(
                solver = Cg(),
                preconditioner = Jacobi(),
                convergence = 1e-9,
                relax = 1.0,
                rtol = 1e-6,
                itmax = 2000,
            ),
        ),
        runtime = Runtime(iterations=1, write_interval=1, time_step=1),
        hardware = hardware,
        boundaries = BCs,
    )

    wall_distance!(model, (:bottom_wall, :upper_wall), config)
    y = Array(model.turbulence.y.values)

    @test all(isfinite, y)
    @test minimum(y) >= 0.0
    @test maximum(y) <= 0.51
    @test maximum(y) > 0.1
end

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh_dev = adapt(backend, mesh)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=1.0),
    turbulence = RANS{KOmegaSST}(walls=(:bottom_wall, :upper_wall)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev,
)

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Extrapolated(:left_wall),
            Extrapolated(:right_wall),
            Wall(:bottom_wall, [0.0, 0.0, 0.0]),
            Wall(:upper_wall, [0.0, 0.0, 0.0]),
        ],
    ),
)

config = Configuration(
    schemes = (y = Schemes(),),
    solvers = (
        y = SolverSetup(
            solver = Cg(),
            preconditioner = Jacobi(),
            convergence = 1e-9,
            relax = 1.0,
            rtol = 1e-6,
            itmax = 2000,
        ),
    ),
    runtime = Runtime(iterations=1, write_interval=1, time_step=1),
    hardware = hardware,
    boundaries = BCs,
)

@test_throws ErrorException XCALibre.Calculate.wall_distance_BCs(mesh_dev, (:missing_wall,), config)

wall_distance!(model, (:bottom_wall, :upper_wall), config)
y = Array(model.turbulence.y.values)

@test all(isfinite, y)
@test minimum(y) >= 0.0
@test maximum(y) <= 0.51
@test maximum(y) > 0.1

# exact distance by brute force over the wall faces (reference for MeshWave)
function exact_wall_distance(mesh, walls)
    boundaries = get_boundaries(mesh.boundaries)
    fIDs = reduce(vcat, [collect(b.IDs_range) for b ∈ boundaries if b.name ∈ walls])
    [minimum(norm(c - XCALibre.Calculate.closest_point_face(c, mesh.face_centre[f],
        view(mesh.face_nodes, mesh.face_nodes_range[f]), mesh.node_coords)) for f ∈ fIDs)
        for c ∈ mesh.cell_centre]
end

function wall_distance_y(mesh, walls; method=MeshWave(), config=nothing)
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1.0),
        turbulence = RANS{KOmegaSST}(walls=walls, wall_distance=method),
        energy = Energy{Isothermal}(),
        domain = mesh,
    )
    config = something(config, Configuration(schemes=(;), solvers=(;),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=(;)))
    wall_distance!(model, walls, config; method=model.wall_info.method)
    model
end

@testset "wall distance method selection" begin
    @test RANS{KOmegaSST}(walls=(:wall,)).args.wall_distance isa MeshWave
    @test RANS{KOmegaLKE}(Tu=0.01, walls=(:wall,), wall_distance=Poisson()).args.wall_distance isa Poisson
    @test Poisson().iterations == 1000
    @test Poisson(iterations=50).iterations == 50
    @test model.wall_info.method isa MeshWave
    @test model.wall_info.walls == (:bottom_wall, :upper_wall)
end

@testset "MeshWave wall distance" begin
    # channel between two plane walls: y = min(y_c, 1 - y_c)
    channel = wall_distance_y(mesh, (:bottom_wall, :upper_wall))
    yc = [c[2] for c ∈ mesh.cell_centre]
    @test Array(channel.turbulence.y.values) ≈ min.(yc, 1 .- yc) atol=1e-14

    # step with a convex corner: exact in every cell
    bfs = UNV2D_mesh(joinpath(grids_dir, "backwardFacingStep_10mm.unv"), scale=0.001)
    ybfs = Array(wall_distance_y(bfs, (:wall,)).turbulence.y.values)
    @test ybfs ≈ exact_wall_distance(bfs, (:wall,)) rtol=1e-12

    # unstructured triangles: close to exact, never below it by more than round-off
    tri = UNV2D_mesh(joinpath(grids_dir, "trig40.unv"), scale=0.001)
    ytri = Array(wall_distance_y(tri, (:bottom,)).turbulence.y.values)
    yexact = exact_wall_distance(tri, (:bottom,))
    @test all(ytri .>= yexact .* (1 - 1e-12))
    @test sum(ytri ./ yexact .- 1)/length(yexact) < 0.01

    # Float32 mesh
    bfs32 = UNV2D_mesh(joinpath(grids_dir, "backwardFacingStep_10mm.unv"), scale=0.001, float_type=Float32)
    y32 = Array(wall_distance_y(bfs32, (:wall,)).turbulence.y.values)
    @test eltype(y32) == Float32
    @test y32 ≈ ybfs rtol=1e-5

end

@testset "Poisson wall distance (method)" begin
    poisson = wall_distance_y(mesh_dev, (:bottom_wall, :upper_wall); method=Poisson(), config=config)
    yp = Array(poisson.turbulence.y.values)
    @test all(isfinite, yp)
    @test maximum(yp) <= 0.51
end

@testset "normal_distance clamp CPU" begin
    test_normal_distance_clamp(CPU())
end

cuda_available = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

if cuda_available
    @testset "normal_distance clamp CUDA" begin
        test_normal_distance_clamp(CUDABackend())
    end

    @testset "wall_distance CUDA channel" begin
        test_wall_distance_channel(CUDABackend())
    end
else
    @info "CUDA unavailable; skipping wall-distance CUDA tests"
end
