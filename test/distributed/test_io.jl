# Phase 7 I/O gate (local-only, needs dev/local_stack.sh): decomposed OpenFOAM writer.
#   source dev/local_stack.sh && mpiexec -n 2 julia --project test/distributed/test_io.jl
# (1) prun! with output=OpenFOAM()+write_interval writes processor<rank>/<iter>/{U,p};
# (2) on-disk internalField round-trips the in-memory owned field (disk==memory);
# (3) gather(field,dm) reconstructs the serial solution in original order (memory==serial).
# (1)&(2)&(3) ⇒ the decomposed case reconstructs to the serial field.
using XCALibre, PETSc, MPI, Test, StaticArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

mesh_path = joinpath(pkgdir(XCALibre), "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv")

io_bcs(mesh) = assign(region=mesh, (
    U = [Dirichlet(:x_min, [0.5, 0.0, 0.0]), Extrapolated(:x_max),
         Wall(:y_min, [0.0, 0.0, 0.0]), Wall(:y_max, [0.0, 0.0, 0.0]),
         Wall(:z_min, [0.0, 0.0, 0.0]), Wall(:z_max, [0.0, 0.0, 0.0])],
    p = [Extrapolated(:x_min), Dirichlet(:x_max, 0.0),
         Extrapolated(:y_min), Extrapolated(:y_max),
         Extrapolated(:z_min), Extrapolated(:z_max)]))

function io_case(mesh; iterations, write_interval)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.8, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.2, rtol=1e-8, atol=1e-12, itmax=2000))
    schemes = (U=Schemes(divergence=Linear), p=Schemes())
    runtime = Runtime(iterations=iterations, time_step=1, write_interval=write_interval)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=io_bcs(mesh))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    model, config
end

# parse an OF nonuniform List<vector> internalField into SVector{3,Float64}[]
function read_of_vectors(path)
    txt = read(path, String)
    m = match(r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\("s, txt)
    n = parse(Int, m.captures[1])
    body = @view txt[m.offset + ncodeunits(m.match):end]
    v = SVector{3,Float64}[]
    for mm ∈ eachmatch(r"\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)", body)
        push!(v, SVector(parse(Float64, mm[1]), parse(Float64, mm[2]), parse(Float64, mm[3])))
        length(v) == n && break
    end
    v
end

iterations = 40

# serial reference (rank 0), no file writing
ref = nothing
if rank == 0
    sm, sc = io_case(UNV3D_mesh(mesh_path, scale=0.001); iterations, write_interval=-1)
    run!(sm, sc)
    ref = (Array(sm.momentum.U.x.values), Array(sm.momentum.U.y.values),
           Array(sm.momentum.U.z.values), Array(sm.momentum.p.values))
end
Us_x, Us_y, Us_z, ps = MPI.bcast(ref, comm; root=0)

gmesh = rank == 0 ? UNV3D_mesh(mesh_path, scale=0.001) : nothing
dm = distribute(gmesh; comm)
model, config = io_case(dm; iterations, write_interval=iterations)

# all ranks share one cwd so processor<rank>/ folders land together
tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
cd(tmp)
MPI.Barrier(comm)

prun!(model, config; output=OpenFOAM())

n = dm.partition.n_owned
iterdir = "processor$rank/$iterations"

@testset "decomposed OF writer (rank $rank)" begin
    @test isfile("$iterdir/U")
    @test isfile("$iterdir/p")

    # (2) disk round-trips memory
    Ud = read_of_vectors("$iterdir/U")
    @test length(Ud) == n
    ux, uy, uz = Array(model.momentum.U.x.values), Array(model.momentum.U.y.values), Array(model.momentum.U.z.values)
    @test maximum(i -> abs(Ud[i][1] - ux[i]), 1:n; init=0.0) < 1e-10
    @test maximum(i -> abs(Ud[i][2] - uy[i]), 1:n; init=0.0) < 1e-10
    @test maximum(i -> abs(Ud[i][3] - uz[i]), 1:n; init=0.0) < 1e-10
end

# (3) gather reconstructs serial in original order (relative tol: the box case is a
# stiff closed-box that grows to ~1e5 U / ~1e8 p, so absolute deltas are meaningless;
# serial and distributed still agree to ~1e-8 relative)
reltol(g, s) = maximum(abs.(g .- s); init=0.0) / max(maximum(abs.(s); init=0.0), eps())
gU = gather(model.momentum.U, dm)
gp = gather(model.momentum.p, dm)
if rank == 0
    dux, duy, duz = reltol(gU.x, Us_x), reltol(gU.y, Us_y), reltol(gU.z, Us_z)
    dp = reltol(gp, ps)
    println("gather-vs-serial (relative): dux=$dux duy=$duy duz=$duz dp=$dp (dir=$tmp)")
    @testset "gather reconstructs serial" begin
        @test dux < 1e-6
        @test duy < 1e-6
        @test duz < 1e-6
        @test dp < 1e-6
    end
end

MPI.Barrier(comm)
