# Crank-Nicolson gate: WALE cavity spin-up vs serial piso!, per rank under mpiexec (Time-term
# old levels and extrapolated explicit terms on owned and ghost cells).
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

nsteps = 20
dt = 0.005

cn_bcs(mesh) = assign(region=mesh, (
    U = [Wall(:inlet, [0.0, 0.0, 0.0]), Wall(:outlet, [0.0, 0.0, 0.0]),
         Dirichlet(:top, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0])],
    p = [Zerogradient(:inlet), Zerogradient(:outlet),
         Zerogradient(:top), Zerogradient(:bottom)],
    nut = [Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0),
         Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0)]))

function cn_case(mesh)
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-4),
        turbulence=LES{WALE}(), energy=Energy{Isothermal}(), domain=mesh)
    tight(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-15,
        relax=1.0, rtol=1e-10, atol=1e-13, itmax=2000)
    config = Configuration(solvers=(U=tight(Bicgstab()), p=tight(Cg())),
        schemes=(U=Schemes(divergence=Linear, time=CrankNicolson), p=Schemes(time=CrankNicolson)),
        runtime=Runtime(iterations=nsteps, time_step=dt, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=cn_bcs(mesh))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    model, config
end

gmesh = rank == 0 ? cavity_mesh() : nothing
ref = if rank == 0
    model, config = cn_case(gmesh)
    piso!(model, config; pref=0.0)
    (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
     collect(model.momentum.p.values), collect(model.turbulence.nut.values))
else
    nothing
end
Us_x, Us_y, ps, nuts = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = cn_case(dm)
run!(model, config; pref=0.0)

dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
ux = Array(model.momentum.U.x.values)
nut = Array(model.turbulence.nut.values)
dnut = maximum(abs.(nut[1:n] .- nuts[orig[1:n]]); init=0.0)

@testset "Crank-Nicolson WALE cavity (rank $rank)" begin
    @test all(isfinite, nuts) && maximum(nuts) > 0
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test dnut < 1e-8
    @test all(abs(ux[i] - Us_x[orig[i]]) < 1e-6 for i ∈ n+1:nloc)
    @test all(abs(nut[i] - nuts[orig[i]]) < 1e-8 for i ∈ n+1:nloc)
end
rank == 0 && println("CN cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp dnut=$dnut")
