# Explicit viscous stress ∇·(ν dev2((∇U)ᵀ)) on a distributed mesh vs the undivided mesh, per rank
# under mpiexec. ∇U comes from grad! of a non-linear field, so the ghost-cell gradients (partial
# face lists) are wrong until transpose_stress! exchanges them: the processor-face fluxes, and
# hence the owned values, only match the serial ones if the exchange is done.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

const SV = XCALibre.Solvers

grids = pkgdir(XCALibre, "examples/0_GRIDS")

function stress_case(mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet),
             Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
    ))
    config = Configuration(schemes=(U=Schemes(),), solvers=(;),
        runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    U = VectorField(mesh)
    centres = [c.centre for c ∈ mesh.cells]
    U.x.values .= [sin(40x)*cos(60y) for (x, y, _) ∈ centres]
    U.y.values .= [x*y*1e3 for (x, y, _) ∈ centres]
    nu = FaceScalarField(mesh)
    nu.values .= [1 + 30f.centre[1] + 50f.centre[2]^2 for f ∈ mesh.faces]

    sync!(U, mesh, config)
    gradU = Grad{Gauss}(U)
    grad!(gradU, FaceVectorField(mesh), U, BCs.U, 0.0, config)
    source = VectorField(mesh)
    SV.transpose_stress!(source, nu, gradU, BCs.U, config)
    # cell-viscosity form (interpolated product): the ghost cell viscosities must be exchanged
    nuc = ScalarField(mesh)
    nuc.values .= [1 + 30x + 50y^2 for (x, y, _) ∈ centres]
    source_cell = VectorField(mesh)
    SV.transpose_stress!(source_cell, nu, gradU, BCs.U, config; cell_mueff=SV.cell_nueff(nuc, nothing))
    source, source_cell
end

load() = UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001)
gmesh = rank == 0 ? load() : nothing
ref = rank == 0 ? ((s, c) = stress_case(gmesh);
    (collect(s.x.values), collect(s.y.values), collect(c.x.values), collect(c.y.values))) : nothing
sx, sy, cx, cy = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
s, sc = stress_case(dm)
n = dm.partition.n_owned
orig = dm.orig_cells[1:n]
scale = maximum(abs, sx) + maximum(abs, sy)
@testset "Explicit viscous stress distributed (rank $rank)" begin
    @test maximum(abs.(Array(s.x.values)[1:n] .- sx[orig])) < 1e-12*scale
    @test maximum(abs.(Array(s.y.values)[1:n] .- sy[orig])) < 1e-12*scale
    cscale = maximum(abs, cx) + maximum(abs, cy)
    @test maximum(abs.(Array(sc.x.values)[1:n] .- cx[orig])) < 1e-12*cscale
    @test maximum(abs.(Array(sc.y.values)[1:n] .- cy[orig])) < 1e-12*cscale
end
MPI.Barrier(comm)
