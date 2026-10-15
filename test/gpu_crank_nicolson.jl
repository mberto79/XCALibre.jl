# CPU/GPU equality of Crank-Nicolson runs: piso! with the LES models and cpiso! (Time-term old
# levels, extrapolated explicit terms; not part of runtests.jl: needs a GPU). Run with an
# environment that has XCALibre and CUDA:
#   julia --project=<env> test/gpu_crank_nicolson.jl
using XCALibre, Test, CUDA, Printf

CUDA.functional() || (println("SKIP gpu_crank_nicolson: CUDA not functional"); exit(0))

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
host = FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily/polyMesh"))

function les_run(backend, Model; float_type=Float64, nsteps=8)
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
    sch = Schemes(time=CrankNicolson, divergence=Upwind, gradient=Gauss)
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

@testset "Crank-Nicolson LES: CPU/GPU equality" begin
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

@testset "Crank-Nicolson LES: Float32 on the GPU" begin
    for Model in (Smagorinsky, WALE, KEquation)
        g = les_run(CUDABackend(), Model; float_type=Float32)
        @test eltype(g.nut) === Float32
        @test all(isfinite, g.nut) && maximum(g.nut) > 0
    end
end

# cpiso!: acoustic standing wave in a periodic strip (outer loops, energy rates)
na = 64
pts = [Point(0.0,0.0,0.0), Point(1.0,0.0,0.0), Point(0.0,2/na,0.0), Point(1.0,2/na,0.0)]
edges = [line!(pts,1,2,na), line!(pts,3,4,na), line!(pts,1,3,2), line!(pts,2,4,2)]
shost = XCALibre.UNV2.update_mesh_format(generate!(MeshBuilder2D(pts,edges,
    [Patch(:left,[3]), Patch(:right,[4]), Patch(:bottom,[1]), Patch(:top,[2])], [quad(edges,[1,2,3,4])])), Int64, Float64)
sxc = [c.centre[1] for c in shost.cells]
cs() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, rtol=1e-12, atol=1e-15)
function cpiso_run(backend)
    mesh = adapt(backend, shost)
    PX = construct_periodic(shost, backend, :left, :right); PY = construct_periodic(shost, backend, :bottom, :top)
    model = Physics(time=Transient(), fluid=Fluid{Compressible}(nu=1e-3, cp=1005.0, gamma=1.4, Pr=0.7),
        turbulence=RANS{Laminar}(), energy=Energy{SensibleEnthalpy}(Tref=0.0), domain=mesh)
    BCs = assign(region=mesh, (U=[PX..., PY...], p=[PX..., PY...], he=[PX..., PY...]))
    config = Configuration(solvers=(U=cs(), p=cs(), he=cs()),
        schemes=(U=Schemes(time=CrankNicolson), p=Schemes(time=CrankNicolson), he=Schemes(time=CrankNicolson)),
        hardware=Hardware(backend=backend, workgroup=backend isa CPU ? 1024 : 32), boundaries=BCs,
        runtime=Runtime(iterations=30, write_interval=-1, time_step=2e-5))
    initialise!(model.momentum.U, [0.0,0.0,0.0])
    p0 = 101325.0 .* (1 .+ 0.01 .* cos.(2pi .* sxc))
    copyto!(model.momentum.p.values, p0)
    initialise!(model.energy.T, 300.0)
    run!(model, config; output=VTK(), progress=false, outer_loops=8)
    Array(model.momentum.p.values), Array(model.momentum.U.x.values), Array(model.energy.T.values)
end

@testset "Crank-Nicolson cpiso!: CPU/GPU equality" begin
    c, g = cpiso_run(CPU()), cpiso_run(CUDABackend())
    for (a, b) ∈ zip(c, g) # p, Ux, T
        @test reldiff(a, b) < 1e-8
    end
end
