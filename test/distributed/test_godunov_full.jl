# godunov! distributed vs serial, full set (not in the gate; runtests_mpi.jl --ranks=2,3
# test_godunov_full.jl): fixed, adaptive and local time stepping; laminar and KOmegaSST:
# owned/ghost fields, residual and force histories. Explicit updates match to round-off; with
# KOmegaSST the k and omega solve tolerance (Krylov.jl serial, PETSc distributed) sets the level.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "godunov_case.jl"))

cases = [
    ("Rusanov Upwind FEuler fixed dt", (flux=Rusanov(), reconstruction=Upwind(), time_stepping=FEuler())),
    ("HLLC MUSCL RK2 adaptive dt", (adaptive=AdaptiveTimeStepping(maxCo=0.5),)),
    ("HLLC MUSCL RK3 LTS LeastSquares", (time_stepping=RK3(), gradient=LeastSquares,
        adaptive=LocalTimeStepping(maxCo=0.8, startCo=0.2, ramp=10))),
    ("SST HLLC MUSCL RK2 LTS", (sst=true, iterations=20,
        adaptive=LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=10, turbulence_dt=:mean))),
]
gmesh = rank == 0 ? cylinder_mesh() : nothing
dm = distribute(gmesh; comm=comm)
for (name, kw) ∈ cases
    compare_godunov(name, kw, gmesh, dm, comm)
end

# 3D with KOmegaSST, written in the decomposed OpenFOAM layout at the last iteration
gbox = rank == 0 ? box_mesh() : nothing
ref = if rank == 0
    ms, cs = box_case(gbox)
    run!(ms, cs; progress=false)
    map(f -> collect(f.values), godunov_fields(ms))
else
    nothing
end
fields_s = MPI.bcast(ref, comm; root=0)

dbox = distribute(gbox; comm=comm)
tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0) # one cwd for processor<rank>/
cd(tmp) do
    model, config = box_case(dbox; write_interval=10)
    run!(model, config; output=OpenFOAM(), progress=false)
    nb = dbox.partition.n_owned
    ob = dbox.orig_cells
    err = maximum(relerr(Array(f.values)[1:nb], fs[ob[1:nb]], fs)
        for (f, fs) ∈ zip(godunov_fields(model), fields_s))
    @testset "3D box KOmegaSST with OpenFOAM output (rank $rank)" begin
        @test err < 1e-9
        # one time directory (named after the pseudo-time), with the fields that have no
        # boundary conditions of their own (rho, y) as well
        times = filter(!=("constant"), readdir("processor$rank"))
        @test length(times) == 1
        for name ∈ ("U", "p", "T", "rho", "k", "omega", "nut", "y")
            @test isfile(joinpath("processor$rank", only(times), name))
        end
    end
    rank == 0 && println("godunov 3D box n=$(MPI.Comm_size(comm)): owned=$err")
end
