using XCALibre
using Test

# Lid-driven cavity run to a steady state with Euler at two time steps 100 times apart: the steady
# state should depend little on Δt. With the old-time source of the face flux taken from the cells
# the two differ by 6.0e-2 of the lid speed (pressure 0.38); with the weighted face correction by
# 2.8e-2 (0.18). Carried in full the face's old flux gives 2.6e-3, but it is unstable (below).
@testset "Δt-consistent face flux" begin
    n = 16
    P, line = XCALibre.Point, XCALibre.line!
    pts = [P(0.0,0.0,0.0), P(1.0,0.0,0.0), P(0.0,1.0,0.0), P(1.0,1.0,0.0)]
    edges = [line(pts,1,2,n), line(pts,3,4,n), line(pts,1,3,n), line(pts,2,4,n)]
    patches = [XCALibre.Patch(:left,[3]), XCALibre.Patch(:right,[4]), XCALibre.Patch(:bottom,[1]),
        XCALibre.Patch(:top,[2])]
    mesh = XCALibre.UNV2.update_mesh_format(XCALibre.generate!(XCALibre.MeshBuilder2D(pts, edges,
        patches, [XCALibre.quad(edges,[1,2,3,4])])), Int64, Float64)
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

# A 2-D channel whose cells shrink fivefold along the flow (blockMeshDict next to the mesh), Linear
# convection, ν = 1e-5, Courant number about 1.2 in the smallest cells. There a_P < V/Δt, so the
# face's old flux carried in full grew the face/cell mismatch by rD_f/Δt ≈ 1.3 per step (max|Ux|
# 1.3e8 after 500 steps); the weighted correction keeps it bounded.
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
