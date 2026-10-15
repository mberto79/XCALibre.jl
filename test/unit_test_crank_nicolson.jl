using XCALibre
using StaticArrays
using Test

# Temporal accuracy of the Crank-Nicolson scheme (trapezoidal form, every other term at tⁿ⁺¹).
# Guards the Time operator's old levels (ddtⁿ, variable time steps, backward-Euler first step)
# and the extrapolation of the lagged explicit terms to tⁿ⁺¹.

function cn_strip_mesh(nx, ny, Lx, Ly)
    pts = [Point(0.0,0.0,0.0), Point(Lx,0.0,0.0), Point(0.0,Ly,0.0), Point(Lx,Ly,0.0)]
    edges = [line!(pts,1,2,nx), line!(pts,3,4,nx), line!(pts,1,3,ny), line!(pts,2,4,ny)]
    patches = [Patch(:left,[3]), Patch(:right,[4]), Patch(:bottom,[1]), Patch(:top,[2])]
    mesh = generate!(MeshBuilder2D(pts, edges, patches, [quad(edges,[1,2,3,4])]))
    XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)
end

cn_tight(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-12,
    relax=1.0, rtol=1e-12, atol=1e-15)
cn_order(e) = log2(e[1]/e[2])

cn_hardware = Hardware(backend=CPU(), workgroup=1024)

# prescribed time steps: the postprocess hook sets the time step of the next iteration
struct CNTimeSteps{V}
    dts::V
end
XCALibre.Postprocess.convert_time_to_iterations(s::CNTimeSteps, model, dt, iterations) = s
XCALibre.Postprocess.runtime_postprocessing!(s::CNTimeSteps, iter::Integer, n::Integer, S, time,
    config) = (iter < length(s.dts) && fill!(config.runtime.dt, s.dts[iter + 1]); nothing)

@testset "Crank-Nicolson temporal accuracy" begin

    # 1-D conduction, zero-gradient ends: cos(πx) is an exact eigenvector of the cell-centred
    # Laplacian, so T = 1 + cos(πx)exp(λₕt) is the exact semi-discrete solution
    nx = 20
    hmesh = cn_strip_mesh(nx, 1, 1.0, 0.1)
    xc = [c.centre[1] for c ∈ hmesh.cells]
    λ = -(2 - 2cos(pi/nx))*nx^2

    function heat(scheme, dt, tend; T_left=nothing, solver=cn_tight(Cg()))
        model = Physics(time=Transient(), solid=Solid{Uniform}(k=1.0, rho=1.0, cp=1.0),
            energy=Energy{Conduction}(), domain=hmesh)
        left = isnothing(T_left) ? Zerogradient(:left) : DirichletFunction(:left, T_left)
        BCs = assign(region=hmesh, (T=[left, Zerogradient(:right), Zerogradient(:bottom),
            Zerogradient(:top)],))
        config = Configuration(solvers=solver, schemes=Schemes(time=scheme),
            hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        model.energy.T.values .= isnothing(T_left) ? 1 .+ cos.(pi .* xc) : 0.0
        run!(model, config; output=VTK(), progress=false)
        copy(model.energy.T.values)
    end

    @testset "scalar decay rate and order" begin
        tend = 0.05
        exact = 1 .+ cos.(pi .* xc) .* exp(λ*tend)
        for (scheme, p) ∈ ((Euler, 1), (CrankNicolson, 2))
            sols = [heat(scheme, dt, tend) for dt ∈ (0.005, 0.0025)]
            e = [maximum(abs.(T .- exact)) for T ∈ sols]
            @test cn_order(e) ≈ p atol=0.1
            # amplitude decays as exp(λₕ t_eff): t_eff/t is 2 if a step advances by 2Δt
            amp = sum((sols[2] .- 1) .* cos.(pi .* xc))/sum(cos.(pi .* xc).^2)
            @test log(amp)/(λ*tend) ≈ 1 atol=(p == 2 ? 1e-3 : 0.02)
        end
    end

    @testset "smoother before the solve" begin
        # a smoother changes the initial guess only, in every solve path
        tend = 0.05
        plain = heat(CrankNicolson, 0.005, tend)
        smoother = JacobiSmoother(domain=hmesh, loops=5)
        for setup ∈ (
                SolverSetup(solver=Cg(), preconditioner=Jacobi(), smoother=smoother,
                    convergence=1e-12, relax=1.0, rtol=1e-12, atol=1e-15),
                SolverSetup(solver=AMG(), preconditioner=Jacobi(), smoother=smoother,
                    convergence=1e-12, relax=1.0, rtol=1e-12, atol=1e-15),
                SolverSetup(solver=Cg(), preconditioner=Jacobi(), smoother=smoother,
                    precision=MixedF32(), convergence=1e-12, relax=1.0, rtol=1e-10, atol=1e-15))
            @test maximum(abs.(heat(CrankNicolson, 0.005, tend; solver=setup) .- plain)) < 1e-6
        end
    end

    @testset "scalar time-dependent Dirichlet" begin
        # boundary values must be taken at the stage time t + Δt/2
        bc(x, t, i) = sin(10pi*t)
        tend = 0.1
        ref = heat(CrankNicolson, 0.005/16, tend; T_left=bc)
        e = [maximum(abs.(heat(CrankNicolson, dt, tend; T_left=bc) .- ref)) for dt ∈ (0.01, 0.005)]
        @test cn_order(e) > 1.9
    end

    # PISO on a shear wave u = sin(2πy)exp(λₕνt) in a doubly periodic box (pressure stays zero)
    ny = 20
    smesh = cn_strip_mesh(2, ny, 0.1, 1.0)
    px = construct_periodic(smesh, CPU(), :left, :right)
    py = construct_periodic(smesh, CPU(), :bottom, :top)
    yc = [c.centre[2] for c ∈ smesh.cells]
    ν = 1.0
    λs = -(2 - 2cos(2pi/ny))*ny^2

    function shear(scheme, dt, tend, inner_loops)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=ν),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=smesh)
        BCs = assign(region=smesh, (U=[px..., py...], p=[px..., py...]))
        config = Configuration(solvers=(U=cn_tight(Bicgstab()), p=cn_tight(Cg())),
            schemes=(U=Schemes(time=scheme), p=Schemes(time=scheme)), hardware=cn_hardware,
            boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        U = model.momentum.U
        U.x.values .= sin.(2pi .* yc); U.y.values .= 0.0; U.z.values .= 0.0
        initialise!(model.momentum.p, 0.0)
        run!(model, config; output=VTK(), progress=false, pref=0.0, inner_loops=inner_loops)
        copy(U.x.values)
    end

    @testset "PISO shear wave, inner_loops=$n" for n ∈ (1, 2)
        tend = 0.05
        exact = sin.(2pi .* yc) .* exp(λs*ν*tend)
        for (scheme, p) ∈ ((Euler, 1), (CrankNicolson, 2))
            e = [maximum(abs.(shear(scheme, dt, tend, n) .- exact)) for dt ∈ (0.005, 0.0025)]
            @test cn_order(e) ≈ p atol=0.1
        end
    end

    # PISO with an eddy-viscosity model: ν_t varies in space and time, without convection or
    # pressure. ν_t must be extrapolated to tⁿ⁺¹ (first order with the lagged value)
    function shear_les(turbulence, dt, tend; steps=fill(dt, round(Int, tend/dt)))
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3),
            turbulence=turbulence, energy=Energy{Isothermal}(), domain=smesh)
        BCs = assign(region=smesh, (U=[px..., py...], p=[px..., py...], nut=[px..., py...],
            k=[px..., py...]))
        schemes = (U=Schemes(time=CrankNicolson), p=Schemes(time=CrankNicolson),
            k=Schemes(time=CrankNicolson))
        config = Configuration(solvers=(U=cn_tight(Bicgstab()), p=cn_tight(Cg()),
            k=cn_tight(Bicgstab())), schemes=schemes, hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=length(steps), write_interval=-1, time_step=steps[1]),
            postprocess=CNTimeSteps(steps))
        U = model.momentum.U
        U.x.values .= sin.(2pi .* yc); U.y.values .= 0.0; U.z.values .= 0.0
        initialise!(model.momentum.p, 0.0)
        hasproperty(model.turbulence, :k) &&
            (model.turbulence.k.values .= 1e-2 .* (1 .+ 0.5 .* cos.(2pi .* yc)))
        # ν_t of the initial field (the first step takes it as its explicit value)
        start = Configuration(solvers=config.solvers, schemes=config.schemes,
            hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=1, write_interval=-1, time_step=1e-14))
        run!(model, start; output=VTK(), progress=false, pref=0.0)
        run!(model, config; output=VTK(), progress=false, pref=0.0)
        copy(U.x.values)
    end

    @testset "PISO eddy viscosity, $(nameof(typeof(t).parameters[1]))" for t ∈ (
            LES{Smagorinsky}(C=1.0),)
        tend = 1.0
        ref = shear_les(t, 0.025/16, tend)
        e = [maximum(abs.(shear_les(t, dt, tend) .- ref)) for dt ∈ (0.05, 0.025)]
        @test cn_order(e) > 1.85
    end

    @testset "variable time steps (1:2:3 pattern)" begin
        # steps dt/2, dt, 3dt/2, ... (mean dt): extrapolation weights from the step history
        tend = 1.2
        run(dt) = shear_les(LES{Smagorinsky}(C=1.0), dt, tend;
            steps=[dt/2*mod1(i, 3) for i ∈ 1:round(Int, tend/dt)])
        ref = run(0.05/16)
        e = [maximum(abs.(run(dt) .- ref)) for dt ∈ (0.05, 0.025)]
        @test cn_order(e) > 1.85
    end

    # PISO, Stokes oscillating wall: u(0,t) = sin(ωt) through a DirichletFunction
    wmesh = cn_strip_mesh(2, 20, 0.1, 1.0)
    wx = construct_periodic(wmesh, CPU(), :left, :right)
    function stokes(dt, tend)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=0.1),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=wmesh)
        BCs = assign(region=wmesh, (
            U=[DirichletFunction(:bottom, (x, t, i) -> SVector(sin(10pi*t), 0.0, 0.0)),
               Dirichlet(:top, [0.0, 0.0, 0.0]), wx...],
            p=[Zerogradient(:bottom), Zerogradient(:top), wx...]))
        config = Configuration(solvers=(U=cn_tight(Bicgstab()), p=cn_tight(Cg())),
            schemes=(U=Schemes(time=CrankNicolson), p=Schemes(time=CrankNicolson)),
            hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
        run!(model, config; output=VTK(), progress=false, pref=0.0)
        copy(model.momentum.U.x.values)
    end

    @testset "PISO time-dependent Dirichlet" begin
        tend = 0.1
        ref = stokes(0.005/16, tend)
        e = [maximum(abs.(stokes(dt, tend) .- ref)) for dt ∈ (0.01, 0.005)]
        @test cn_order(e) > 1.9
    end

    # PISO, advected Taylor-Green vortex: convection and pressure coupling. The convecting flux
    # must be extrapolated to tⁿ⁺¹ (first order with the start-of-step flux)
    n = 24
    tmesh = cn_strip_mesh(n, n, 1.0, 1.0)
    tx = construct_periodic(tmesh, CPU(), :left, :right)
    ty = construct_periodic(tmesh, CPU(), :bottom, :top)
    txc = [c.centre[1] for c ∈ tmesh.cells]; tyc = [c.centre[2] for c ∈ tmesh.cells]
    function tgv(dt, tend)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=0.01),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=tmesh)
        BCs = assign(region=tmesh, (U=[tx..., ty...], p=[tx..., ty...]))
        config = Configuration(solvers=(U=cn_tight(Bicgstab()), p=cn_tight(Cg())),
            schemes=(U=Schemes(time=CrankNicolson), p=Schemes(time=CrankNicolson)),
            hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        U = model.momentum.U
        U.x.values .= 1.0 .- cos.(2pi .* txc) .* sin.(2pi .* tyc)
        U.y.values .= 0.5 .+ sin.(2pi .* txc) .* cos.(2pi .* tyc)
        U.z.values .= 0.0
        initialise!(model.momentum.p, 0.0)
        run!(model, config; output=VTK(), progress=false, pref=0.0)
        [copy(U.x.values); copy(U.y.values)]
    end

    @testset "PISO non-solenoidal initial field" begin
        # the backward-Euler first step projects it
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=0.01),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=tmesh)
        BCs = assign(region=tmesh, (U=[tx..., ty...], p=[tx..., ty...]))
        config = Configuration(solvers=(U=cn_tight(Bicgstab()), p=cn_tight(Cg())),
            schemes=(U=Schemes(time=CrankNicolson), p=Schemes(time=CrankNicolson)),
            hardware=cn_hardware, boundaries=BCs,
            runtime=Runtime(iterations=3, write_interval=-1, time_step=0.01))
        U = model.momentum.U
        U.x.values .= sin.(2pi .* txc); U.y.values .= 0.0; U.z.values .= 0.0
        initialise!(model.momentum.p, 0.0)
        run!(model, config; output=VTK(), progress=false, pref=0.0)
        @test maximum(abs.(U.x.values)) < 0.1
    end

    @testset "PISO advected Taylor-Green vortex" begin
        tend = 0.2
        ref = tgv(0.01/16, tend)
        e = [maximum(abs.(tgv(dt, tend) .- ref)) for dt ∈ (0.02, 0.01)]
        @test cn_order(e) > 1.6
    end
end
