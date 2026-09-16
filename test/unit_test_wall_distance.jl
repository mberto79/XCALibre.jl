using XCALibre
using KernelAbstractions
using Test

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
