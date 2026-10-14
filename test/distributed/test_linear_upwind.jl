# LinearUpwind / LimitedLinearUpwind source per rank vs the undivided mesh. Ghost-cell gradients of a
# non-linear field are wrong until discretise! exchanges them, so cells next to a processor face
# match the serial source only if that exchange is done.
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

    out = []
    # gradient limiters exchange their ghost values: limited gradients must match serial too
    for limiter ∈ (nothing, Venkatakrishnan(5.0)), scheme ∈ (LinearUpwind, LimitedLinearUpwind)
        gradU = Grad{Gauss}(U)
        grad!(gradU, FaceVectorField(mesh), U, BCs.U, 0.0, config)
        limit_gradient!(limiter, gradU, U, config)
        gradphi = Grad{Gauss}(phi)
        grad!(gradphi, FaceScalarField(mesh), phi, BCs.phi, 0.0, config)
        limit_gradient!(limiter, gradphi, phi, config)
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
