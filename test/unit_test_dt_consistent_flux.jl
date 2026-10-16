using XCALibre
using Test

function dt_flux_cavity_mesh(n=16)
    P, line = XCALibre.Point, XCALibre.line!
    pts = [P(0.0,0.0,0.0), P(1.0,0.0,0.0), P(0.0,1.0,0.0), P(1.0,1.0,0.0)]
    edges = [line(pts,1,2,n), line(pts,3,4,n), line(pts,1,3,n), line(pts,2,4,n)]
    patches = [XCALibre.Patch(:left,[3]), XCALibre.Patch(:right,[4]), XCALibre.Patch(:bottom,[1]),
        XCALibre.Patch(:top,[2])]
    XCALibre.UNV2.update_mesh_format(XCALibre.generate!(XCALibre.MeshBuilder2D(pts, edges,
        patches, [XCALibre.quad(edges,[1,2,3,4])])), Int64, Float64)
end

# Lid-driven cavity, steady state with Euler at Δt 100 times apart. Old-time source from the cells:
# they differ by 6.0e-2 of the lid speed (pressure 0.38); weighted face correction: 2.8e-2 (0.18).
# Carried in full, the face's old flux gives 2.6e-3 but is unstable (graded mesh below).
@testset "Δt-consistent face flux" begin
    mesh = dt_flux_cavity_mesh()
    function cavity(dt, tend)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=0.05),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
        BCs = assign(region=mesh, (U=[Wall(:left,[0.0,0,0]), Wall(:right,[0.0,0,0]),
            Wall(:bottom,[0.0,0,0]), Dirichlet(:top,[1.0,0,0])],
            p=[Wall(:left), Wall(:right), Wall(:bottom), Wall(:top)]))
        s() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
            relax=1.0, rtol=1e-12, atol=1e-15)
        config = Configuration(solvers=(U=s(), p=s()), schemes=(U=Schemes(time=Euler),
            p=Schemes()), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        initialise!(model.momentum.U, [0.0,0,0]); initialise!(model.momentum.p, 0.0)
        run!(model, config; progress=false, pref=0.0)
        copy(model.momentum.U.x.values), copy(model.momentum.p.values)
    end
    a, b = cavity(0.002, 20.0), cavity(0.2, 20.0)
    @test maximum(abs.(a[1] .- b[1])) < 3.5e-2
    @test maximum(abs.(a[2] .- b[2])) < 0.25
end

# The same cavity in multiphase! (one phase, ν = 0.05): its face flux takes the same correction,
# so the two steady states differ by 2.8e-2 as above (6.0e-2 without it).
@testset "Δt-consistent face flux in multiphase!" begin
    mesh = dt_flux_cavity_mesh()
    function cavity(dt, tend)
        model = Physics(time=Transient(), fluid=Fluid{Multiphase}(
            model=VOF(cAlpha=0.0, sigma=0.0),
            phases=(Phase(rho=1000.0, mu=50.0), Phase(rho=1.2, mu=1.8e-5)),
            gravity=Gravity([0.0, 0.0, 0.0])),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
        zg = [Zerogradient(:left), Zerogradient(:right), Zerogradient(:bottom), Zerogradient(:top)]
        BCs = assign(region=mesh, (U=[Wall(:left,[0.0,0,0]), Wall(:right,[0.0,0,0]),
            Wall(:bottom,[0.0,0,0]), Dirichlet(:top,[1.0,0,0])], p_rgh=zg, alpha=zg))
        s() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
            relax=1.0, rtol=1e-12, atol=1e-15)
        sch = Schemes(time=Euler)
        config = Configuration(solvers=(U=s(), p_rgh=s(), alpha=s()),
            schemes=(U=sch, p=sch, p_rgh=sch, alpha=Schemes(time=Euler, divergence=Upwind)),
            hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        initialise!(model.momentum.U, [0.0,0,0]); initialise!(model.fluid.p_rgh, 0.0)
        initialise!(model.fluid.alpha, 1.0)
        run!(model, config; progress=false, pref=0.0)
        copy(model.momentum.U.x.values), copy(model.momentum.p.values)
    end
    a, b = cavity(0.002, 20.0), cavity(0.2, 20.0)
    @test maximum(abs.(a[1] .- b[1])) < 3.5e-2
    @test maximum(abs.(a[2] .- b[2]))/1000 < 0.25
end

# The cavity in cpiso! with a gas of density 0.35. The face/cell mismatch of the mass flux grows by
# ρ_f·rD_f/Δt per step, so the bound of the correction needs the face density: bounded by Δt/rD_f
# alone it keeps a fraction ρ of the correction (difference 5.3e-2); with it, 2.8e-2 as above.
@testset "Δt-consistent face flux in cpiso!" begin
    mesh = dt_flux_cavity_mesh()
    function cavity(dt, tend)
        cp = 1005.0
        model = Physics(time=Transient(),
            fluid=Fluid{WeaklyCompressible}(nu=0.05, cp=cp, gamma=1.4, Pr=0.7),
            turbulence=RANS{Laminar}(), energy=Energy{SensibleEnthalpy}(Tref=288.15), domain=mesh)
        h(name) = FixedTemperature(name, T=300.0, Enthalpy(cp=cp, Tref=288.15))
        BCs = assign(region=mesh, (U=[Wall(:left,[0.0,0,0]), Wall(:right,[0.0,0,0]),
            Wall(:bottom,[0.0,0,0]), Dirichlet(:top,[1.0,0,0])],
            p=[Wall(:left), Wall(:right), Wall(:bottom), Wall(:top)],
            he=[h(:left), h(:right), h(:bottom), h(:top)]))
        s() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
            relax=1.0, rtol=1e-12, atol=1e-15)
        sp = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0,
            rtol=1e-10, atol=1e-12)
        sch = Schemes(time=Euler)
        config = Configuration(solvers=(U=s(), p=sp, he=s()),
            schemes=(rho=sch, U=sch, p=sch, he=sch),
            hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs,
            runtime=Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt))
        initialise!(model.momentum.U, [0.0,0,0]); initialise!(model.momentum.p, 3.0e4)
        initialise!(model.energy.T, 300.0)
        run!(model, config; progress=false)
        copy(model.momentum.U.x.values), copy(model.momentum.p.values), copy(model.fluid.rho.values)
    end
    a, b = cavity(0.002, 20.0), cavity(0.2, 20.0)
    @test 0.34 < minimum(a[3]) && maximum(a[3]) < 0.36
    @test maximum(abs.(a[1] .- b[1])) < 3.5e-2
    @test maximum(abs.(a[2] .- b[2]))/a[3][1] < 0.25
end

# 2-D channel, cells shrink fivefold along the flow, Linear convection, ν = 1e-5, Courant number 1.2
# in the smallest cells. There a_P < V/Δt: the face's old flux carried in full grew the face/cell
# mismatch by rD_f/Δt ≈ 1.3 per step (max|Ux| 1.3e8 after 500 steps); the weighted one is bounded.
@testset "Δt-consistent face flux on a graded mesh" begin
    mesh = FOAM3D_mesh(pkgdir(XCALibre, "test/grids/OF_graded_channel/polyMesh"), scale=1)
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-5),
        turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    BCs = assign(region=mesh, (
        U=[Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
            Zerogradient(:top), Empty(:frontAndBack)],
        p=[Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Dirichlet(:top, 0.0),
            Empty(:frontAndBack)]))
    s(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-8, atol=1e-10,
        relax=1.0)
    config = Configuration(solvers=(U=s(Bicgstab()), p=s(Cg())),
        schemes=(U=Schemes(time=Euler, divergence=Linear), p=Schemes(time=Euler)),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs,
        runtime=Runtime(iterations=500, write_interval=-1, time_step=0.003))
    initialise!(model.momentum.U, [1.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    run!(model, config; progress=false)
    @test maximum(abs, model.momentum.U.x.values) < 1.1
    @test maximum(abs, model.momentum.U.y.values) < 1e-2
end
