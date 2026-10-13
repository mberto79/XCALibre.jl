# Deferred-correction source of LinearUpwind / LimitedLinearUpwind on a distributed mesh vs the
# undivided mesh, per rank under mpiexec. ∇U comes from grad! of a non-linear field, so the
# gradient of a ghost cell (partial face list) is wrong until discretise! exchanges it: the
# source of cells next to a processor face only matches the serial one if that exchange is done.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

grids = pkgdir(XCALibre, "examples/0_GRIDS")

function source_case(mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet),
             Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
        phi = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet), Zerogradient(:wall), Zerogradient(:top)],
    ))
    config = Configuration(schemes=(U=Schemes(),), solvers=(;),
        runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    centres = [c.centre for c ∈ mesh.cells]
    mdotf = FaceScalarField(mesh)
    # position-based flux with both signs, so the upwind cell is on either side of a processor face
    mdotf.values .= [1e-3*(0.4 + sin(30f.centre[1])*cos(50f.centre[2])) for f ∈ mesh.faces]

    U = VectorField(mesh)
    U.x.values .= [sin(40x)*cos(60y) for (x, y, _) ∈ centres]
    U.y.values .= [x*y*1e3 for (x, y, _) ∈ centres]
    phi = ScalarField(mesh)
    phi.values .= [cos(35x)*sin(55y) for (x, y, _) ∈ centres]
    sync!(U, mesh, config); sync!(phi, mesh, config)

    gradU = Grad{Gauss}(U)
    grad!(gradU, FaceVectorField(mesh), U, BCs.U, 0.0, config)
    gradphi = Grad{Gauss}(phi)
    grad!(gradphi, FaceScalarField(mesh), phi, BCs.phi, 0.0, config)

    out = []
    for scheme ∈ (LinearUpwind, LimitedLinearUpwind)
        ueqn = (Divergence{scheme}(mdotf, U, gradU.result) == Source(VectorField(mesh))
            ) → VectorEquation(U, BCs.U)
        discretise!(ueqn, U, config)
        peqn = (Divergence{scheme}(mdotf, phi, gradphi.result) == Source(ScalarField(mesh))
            ) → ScalarEquation(phi, BCs.phi)
        discretise!(peqn, phi, config)
        push!(out, (collect(ueqn.equation.bx), collect(ueqn.equation.by), collect(peqn.equation.b)))
    end
    out
end

load() = UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001)
gmesh = rank == 0 ? load() : nothing
ref = rank == 0 ? source_case(gmesh) : nothing
ref = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
res = source_case(dm)
n = dm.partition.n_owned
orig = dm.orig_cells[1:n]
@testset "LinearUpwind source distributed (rank $rank)" begin
    for (s, (r, g)) ∈ enumerate(zip(ref, res)), k ∈ 1:3
        scale = maximum(abs, r[k])
        @test scale > 0
        @test maximum(abs.(g[k][1:n] .- r[k][orig])) < 1e-12*scale
    end
end
MPI.Barrier(comm)
