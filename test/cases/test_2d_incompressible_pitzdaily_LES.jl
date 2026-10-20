using XCALibre


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_pitzDaily/polyMesh"

mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file)
@test typeof(mesh) <: Mesh3
@test length(mesh.cells) == 12225
@test length(mesh.faces) == 49180
@test length(mesh.nodes) == 25012

backend = CPU(); workgroup = workgroupsize(mesh)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(hardware.backend, mesh) 



U_inlet = 10.0
velocity = [U_inlet, 0.0, 0.0]
k_inlet = 2e-05
nu = 1e-5
δt = 1e-05
wall_patches = [:upperWall, :lowerWall]

function run_les(turb; k_eq)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu = nu),
        turbulence = turb,
        energy = Energy{Isothermal}(),
        domain = mesh_dev
    )
    bcs = (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Empty(:frontAndBack),
            Wall.(wall_patches, Ref([0.0, 0.0, 0.0]))...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Empty(:frontAndBack),
            Wall.(wall_patches, Ref(0.0))...
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet, 0.0),
            Empty(:frontAndBack),
            Dirichlet.(wall_patches, Ref(0.0))...
        ]
    )
    sch = (U = Schemes(time=Euler, divergence=Upwind, gradient=Gauss),
           p = Schemes(time=Euler, divergence=Upwind, gradient=Gauss))
    ss(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-7, atol=1e-5, rtol=0.0, relax=1.0)
    sol = (U = ss(Bicgstab()), p = ss(Cg()))
    if k_eq
        bcs = merge(bcs, (k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Empty(:frontAndBack),
            Dirichlet.(wall_patches, Ref(0.0))...
        ],))
        sch = merge(sch, (k = Schemes(time=Euler, divergence=Upwind, gradient=Gauss),))
        sol = merge(sol, (k = ss(Bicgstab()),))
    end
    BCs = assign(region=mesh_dev, bcs)
    runtime = Runtime(iterations=30, time_step=δt, write_interval=30)
    config = Configuration(
        solvers=sol, schemes=sch, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    k_eq && initialise!(model.turbulence.k, k_inlet)
    initialise!(model.turbulence.nut, 0.0)

    run!(model, config, inner_loops=2, ncorrectors=0, output=OpenFOAM(), progress=false)

    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
    @test inlet[1] ≈ 10
    @test outlet[1] ≈ 7.60 atol=0.1
    @test all(>=(0), model.turbulence.nut.values)
    k_eq && @test all(>=(0), model.turbulence.k.values)
end

@testset "KEquation" begin
    run_les(LES{KEquation}(); k_eq=true)
end
@testset "Smagorinsky" begin
    run_les(LES{Smagorinsky}(); k_eq=false)
end
