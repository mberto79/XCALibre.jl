# LeastSquares gradients per rank vs the undivided mesh. The normal matrix is built
# from the local cells and the gradient is exchanged at the end, so owned cells (including those next to a
# processor face) must match the serial gradient.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

grids = pkgdir(XCALibre, "examples/0_GRIDS")

function gradients(mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet),
             Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
        phi = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet), Zerogradient(:wall), Zerogradient(:top)],
    ))
    config = Configuration(schemes=(U=Schemes(),), solvers=(;),
        runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)
    centres = [c.centre for c ∈ mesh.cells]
    U = VectorField(mesh)
    U.x.values .= [sin(40x)*cos(60y) for (x, y, _) ∈ centres]
    U.y.values .= [x*y*1e3 for (x, y, _) ∈ centres]
    phi = ScalarField(mesh)
    phi.values .= [cos(35x)*sin(55y) for (x, y, _) ∈ centres]
    sync!(U, mesh, config); sync!(phi, mesh, config)
    gU = Grad{LeastSquares}(U)
    grad!(gU, FaceVectorField(mesh), U, BCs.U, 0.0, config)
    gp = Grad{LeastSquares}(phi)
    grad!(gp, FaceScalarField(mesh), phi, BCs.phi, 0.0, config)
    ([collect(getproperty(gU.result, f).values) for f ∈
      (:xx, :xy, :xz, :yx, :yy, :yz, :zx, :zy, :zz)]...,
     [collect(getproperty(gp.result, f).values) for f ∈ (:x, :y, :z)]...)
end

load() = UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001)
gmesh = rank == 0 ? load() : nothing
ref = rank == 0 ? gradients(gmesh) : nothing
ref = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
res = gradients(dm)
n = dm.partition.n_owned
orig = dm.orig_cells[1:n]
@testset "least-squares gradient distributed (rank $rank)" begin
    for (r, g) ∈ zip(ref, res)
        scale = max(maximum(abs, r), eps())
        @test maximum(abs.(g[1:n] .- r[orig])) < 1e-10*scale
    end
end
MPI.Barrier(comm)
