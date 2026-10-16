using XCALibre
using Test

# Lid-driven cavity run to a steady state with Euler at two time steps 100 times apart: the steady
# state must not depend on Δt. Before the face flux took its old-time source from the face's own
# old flux, the two differed by 7% of the lid speed (0.3% after).
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
    @test maximum(abs.(a[1] .- b[1])) < 5e-3
    @test maximum(abs.(a[2] .- b[2])) < 3e-2
end
