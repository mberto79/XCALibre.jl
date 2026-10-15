# Distributed density-based solver (godunov!) vs serial, per rank under mpiexec: fixed, adaptive
# and local time stepping, first- and second-order reconstruction, Rusanov and HLLC, laminar and
# KOmegaSST. Owned and ghost cells, residual and force histories must match the serial run: to
# round-off when the update is explicit; with KOmegaSST the k and omega solves stop at their
# tolerance (Krylov.jl serially, PETSc distributed), which sets the level of agreement.
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
tolerance(kw) = get(kw, :sst, false) ? 1e-7 : 1e-9
forces = (patches=[:cylinder], interval=5)

gmesh = rank == 0 ? cylinder_mesh() : nothing
dm = distribute(gmesh; comm=comm)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells

# difference relative to the largest value of the whole serial field (or history)
relerr(a, b, ref=b) = maximum(abs.(a .- b); init=0.0) / max(maximum(abs, ref; init=0.0), eps())

for (name, kw) ∈ cases
    ref = if rank == 0
        ms, cs = godunov_case(gmesh; kw...)
        rs = run!(ms, cs; progress=false, forces)
        (map(f -> collect(f.values), godunov_fields(ms)), rs)
    else
        nothing
    end
    fields_s, res_s = MPI.bcast(ref, comm; root=0)

    model, config = godunov_case(dm; kw...)
    res = run!(model, config; progress=false, forces)
    fields_d = map(f -> Array(f.values), godunov_fields(model))

    owned = [relerr(fd[1:n], fs[orig[1:n]], fs) for (fd, fs) ∈ zip(fields_d, fields_s)]
    ghost = [relerr(fd[n+1:nloc], fs[orig[n+1:nloc]], fs) for (fd, fs) ∈ zip(fields_d, fields_s)]
    F_s = res_s.forces.cylinder
    F_d = res.forces.cylinder
    dF = maximum(relerr(F_d.pressure[i], F_s.pressure[i]) for i ∈ eachindex(F_s.pressure))

    tol = tolerance(kw)
    @testset "$name (rank $rank)" begin
        @test all(isfinite, res.rho)
        @test maximum(owned) < tol
        @test maximum(ghost) < tol
        @test relerr(res.rho, res_s.rho) < tol
        @test relerr(res.rhoU, res_s.rhoU) < tol
        @test relerr(res.rhoE, res_s.rhoE) < tol
        @test F_d.iteration == F_s.iteration
        @test dF < tol
    end
    rank == 0 && println("godunov $name n=$(MPI.Comm_size(comm)): owned=$(maximum(owned)) ",
        "ghost=$(maximum(ghost)) rho_res=$(relerr(res.rho, res_s.rho)) force=$dF")
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
