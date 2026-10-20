# CPU/GPU equality of the LES models (not in a runtests.jl tier: needs a GPU). Run with an
# environment that has XCALibre and CUDA:
#   julia --project=<env> test/gpu/test_les_models.jl
using XCALibre, Test, CUDA, Printf

CUDA.functional() || (println("SKIP gpu_les_models: CUDA not functional"); exit(0))

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
host = FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily/polyMesh"))

function les_run(backend, Model; float_type=Float64, nsteps=5)
    hostT = float_type == Float64 ? host :
        FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily/polyMesh"), float_type=float_type)
    mesh = adapt(backend, hostT)
    walls = [:upperWall, :lowerWall]
    velocity = [10.0, 0.0, 0.0]
    k_inlet = 2e-5
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-5),
        turbulence=LES{Model}(), energy=Energy{Isothermal}(), domain=mesh)
    fields = (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Empty(:frontAndBack),
            Wall.(walls, Ref([0.0, 0.0, 0.0]))...],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Empty(:frontAndBack),
            Wall.(walls, Ref(0.0))...],
        nut = [Extrapolated(:inlet), Extrapolated(:outlet), Empty(:frontAndBack),
            Dirichlet.(walls, Ref(0.0))...])
    if Model == KEquation
        fields = (; fields..., k = [Dirichlet(:inlet, k_inlet), Zerogradient(:outlet),
            Empty(:frontAndBack), Dirichlet.(walls, Ref(0.0))...])
    end
    BCs = assign(region=mesh, fields)
    sch = Schemes(time=Euler, divergence=Upwind, gradient=Gauss)
    T = float_type
    tol = T == Float64 ? 1e-10 : 1f-6
    setup(solver) = SolverSetup(float_type=T, solver=solver, preconditioner=Jacobi(),
        convergence=tol, atol=tol, rtol=0, relax=1, itmax=1000)
    schemes = (U=sch, p=sch, k=sch)
    solvers = (U=setup(Bicgstab()), p=setup(Cg()), k=setup(Bicgstab()))
    config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs,
        runtime=Runtime(iterations=nsteps, time_step=T(1e-5), write_interval=-1),
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32))
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.nut, 0.0)
    Model == KEquation && initialise!(model.turbulence.k, k_inlet)
    run!(model, config, inner_loops=2, ncorrectors=0, progress=false)
    (; U, p) = model.momentum
    out = (nut=Array(model.turbulence.nut.values), Ux=Array(U.x.values),
        Uy=Array(U.y.values), p=Array(p.values))
    Model == KEquation ? (; out..., k=Array(model.turbulence.k.values)) : out
end

reldiff(a, b) = maximum(abs.(a .- b))/max(maximum(abs.(a)), eps(eltype(a)))

@testset "LES models: CPU/GPU equality" begin
    for Model in (Smagorinsky, WALE, KEquation)
        c = les_run(CPU(), Model)
        g = les_run(CUDABackend(), Model)
        @test maximum(c.nut) > 0
        for name in keys(c)
            err = reldiff(c[name], g[name])
            @printf("%-11s %-3s rel. diff %.2e\n", Model, name, err)
            @test err < 1e-8
        end
    end
end

@testset "LES models: Float32 on the GPU" begin
    for Model in (Smagorinsky, WALE, KEquation)
        g = les_run(CUDABackend(), Model; float_type=Float32)
        @test eltype(g.nut) === Float32
        @test all(isfinite, g.nut) && maximum(g.nut) > 0
    end
end
