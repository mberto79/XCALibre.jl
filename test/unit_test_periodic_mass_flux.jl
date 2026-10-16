using XCALibre
using Test

# Periodic strip two cells across: each cell's periodic partner is also its interior neighbour, so
# the matrix entry of the pair holds both faces. The flux correction of the periodic face must use
# that face's coefficient only (with the matrix entry, continuity missed by 13% of the flux).
@testset "periodic mass flux, two cells across" begin
    P, line = XCALibre.Point, XCALibre.line!
    pts = [P(0.0,0.0,0.0), P(1.0,0.0,0.0), P(0.0,1.0,0.0), P(1.0,1.0,0.0)]
    edges = [line(pts,1,2,2), line(pts,3,4,2), line(pts,1,3,16), line(pts,2,4,16)]
    patches = [XCALibre.Patch(:left,[3]), XCALibre.Patch(:right,[4]), XCALibre.Patch(:bottom,[1]),
        XCALibre.Patch(:top,[2])]
    mesh = XCALibre.UNV2.update_mesh_format(XCALibre.generate!(XCALibre.MeshBuilder2D(pts, edges,
        patches, [XCALibre.quad(edges,[1,2,3,4])])), Int64, Float64)
    per = construct_periodic(mesh, CPU(), :left, :right)
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=0.05),
        turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    BCs = assign(region=mesh, (U=[Wall(:bottom,[0.0,0,0]), Wall(:top,[0.0,0,0]), per...],
        p=[Wall(:bottom), Wall(:top), per...]))
    s() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0,
        rtol=1e-12, atol=1e-15)
    config = Configuration(solvers=(U=s(), p=s()), schemes=(U=Schemes(time=Euler), p=Schemes()),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=0.01))
    initialise!(model.momentum.U, [0.0,0,0]); initialise!(model.momentum.p, 0.0)
    model.momentum.U.y.values .= [c.centre[1] < 0.5 ? 0.1 : -0.1 for c ∈ mesh.cells] # needs ∂p/∂x
    run!(model, config; progress=false, pref=0.0)
    mdotf = model.momentum.mdotf.values
    div = zeros(length(mesh.cells))
    for (fID, face) ∈ enumerate(mesh.faces) # boundary faces (periodic ones too) own one cell
        c1, c2 = face.ownerCells
        div[c1] += mdotf[fID]
        c1 == c2 || (div[c2] -= mdotf[fID])
    end
    @test maximum(abs, model.momentum.p.values) > 1 # the case does exercise the pressure
    @test maximum(abs, div) < 1e-10*maximum(abs, mdotf)
end
