# WALE LES gate: transient cavity spin-up vs serial piso!, per rank under mpiexec.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

nsteps = 20
dt = 0.005

wale_bcs(mesh) = assign(region=mesh, (
    U = [Wall(:inlet, [0.0, 0.0, 0.0]), Wall(:outlet, [0.0, 0.0, 0.0]),
         Dirichlet(:top, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0])],
    p = [Zerogradient(:inlet), Zerogradient(:outlet),
         Zerogradient(:top), Zerogradient(:bottom)],
    nut = [Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0),
         Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0)]))

function wale_case(mesh)
    model, config = incompressible_case(mesh, wale_bcs;
        iterations=nsteps, time=Transient(), time_step=dt)
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-4),
        turbulence=LES{WALE}(), energy=Energy{Isothermal}(), domain=mesh)
    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    model, config
end

gmesh = rank == 0 ? cavity_mesh() : nothing
ref = if rank == 0
    model, config = wale_case(gmesh)
    piso!(model, config; pref=0.0)
    (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
     collect(model.momentum.p.values), collect(model.turbulence.nut.values))
else
    nothing
end
Us_x, Us_y, ps, nuts = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = wale_case(dm)
run!(model, config; pref=0.0)

dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
nut = Array(model.turbulence.nut.values)
dnut = maximum(abs.(nut[1:n] .- nuts[orig[1:n]]); init=0.0)

@testset "WALE cavity spin-up (rank $rank)" begin
    @test all(isfinite, nuts) && maximum(nuts) > 0
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test dnut < 1e-8
    @test all(abs(nut[i] - nuts[orig[i]]) < 1e-8 for i ∈ n+1:nloc)
end
rank == 0 && println("WALE cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp dnut=$dnut")
