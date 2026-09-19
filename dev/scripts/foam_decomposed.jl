# REPART=<method> repartitions after loading and prints balance and edge-cut before and after.
# WRITE=1 writes results into the decomposed case at the last iteration, for reconstructPar.
# Serial FOAM3D_mesh run against distribute(FOAMCase) on a decomposePar case; laminar 3D BFS, tight inner tolerances.
using XCALibre, MPI
MODE, CASE, ITERS = ARGS[1], ARGS[2], parse(Int, ARGS[3])
SCALE = 0.001
WRITE = get(ENV, "WRITE", "0") == "1"
REPART = get(ENV, "REPART", "")

function bfs(mesh; write=false)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    bcs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
             Zerogradient(:sides), Zerogradient(:top)],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Extrapolated(:sides), Extrapolated(:top)]))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-15, relax=0.8, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-15, relax=0.2, rtol=1e-8, atol=1e-12, itmax=2000))
    config = Configuration(solvers=solvers, schemes=(U=Schemes(divergence=Linear), p=Schemes()),
        runtime=Runtime(iterations=ITERS, write_interval=write ? ITERS : -1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=bcs)
    initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    write ? run!(model, config; output=OpenFOAM()) : run!(model, config)
    model, config
end

ref(case) = joinpath(case, "serial_$(ITERS).bin")

if MODE == "serial"
    m, _ = bfs(FOAM3D_mesh(joinpath(CASE, "constant", "polyMesh"); scale=SCALE))
    U, p = m.momentum.U, m.momentum.p
    write(ref(CASE), hcat(U.x.values, U.y.values, U.z.values, p.values))
    println("SERIAL ncells=$(length(p.values)) written")
elseif MODE == "metis"
    partition_cells(FOAM3D_mesh(joinpath(CASE, "constant", "polyMesh"); scale=SCALE), parse(Int, ARGS[4]))
else
    using PETSc
    MPI.Init()
    comm = MPI.COMM_WORLD
    t = @elapsed dm = distribute(FOAMCase(ARGS[4]; scale=SCALE); comm)
    stats(d) = (extrema(MPI.Allgather(d.partition.n_owned, comm)),
        MPI.Allreduce(sum(pp -> length(pp.faces), d.procs; init=0), +, comm) ÷ 2)
    if !isempty(REPART)
        before = stats(dm)
        tr = @elapsed dm = repartition(dm; method=Symbol(REPART))
        after = stats(dm)
        MPI.Comm_rank(comm) == 0 && println("REPART method=$REPART s=$(round(tr; digits=2)) ",
            "before cells=$(before[1]) cut=$(before[2]) after cells=$(after[1]) cut=$(after[2])")
    end
    WRITE && cd(ARGS[4])
    m, config = bfs(dm; write=WRITE)
    ghosts = check_ghosts(m.momentum.p, dm, config)
    gU, gp = gather(m.momentum.U, dm), gather(m.momentum.p, dm)
    if MPI.Comm_rank(comm) == 0
        s = reshape(reinterpret(Float64, read(ref(CASE))), :, 4)
        rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
        d = (rel(gU.x, s[:, 1]), rel(gU.y, s[:, 2]), rel(gU.z, s[:, 3]), rel(gp, s[:, 4]))
        println("FOAMCASE n=$(MPI.Comm_size(comm)) load_s=$(round(t; digits=2)) ghosts=$ghosts ",
            "rel_Ux=$(d[1]) rel_Uy=$(d[2]) rel_Uz=$(d[3]) rel_p=$(d[4]) pass=$(all(<(1e-5), d))")
    end
end
