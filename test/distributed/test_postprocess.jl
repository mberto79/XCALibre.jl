# Runtime post-processing on a distributed mesh: FieldAverage, FieldRMS, ReynoldsStress, Qcriterion
# and Probe match a serial run (piso!, simple!, laplace!), probe files are written once by rank 0,
# and post-processed fields are written to the processor directories.
using XCALibre, PETSc, MPI, Test, StaticArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))
include(joinpath(@__DIR__, "laplace_case.jl"))

tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
rank == 0 && foreach(d -> mkpath(joinpath(tmp, d)), ("serial", "dist"))
MPI.Barrier(comm)

with_config(config, postprocess) = Configuration(solvers=config.solvers, schemes=config.schemes,
    runtime=config.runtime, hardware=config.hardware, boundaries=config.boundaries,
    postprocess=postprocess)

relerr(a, b) = maximum(abs.(a .- b); init=0.0) / max(maximum(abs.(b); init=0.0), eps())
read_probe(file) = [parse.(Float64, split(l, ',')) for l ∈ readlines(file)[2:end]]

# runs `case` serially on rank 0 and distributed on all ranks; returns, on rank 0, the serial and
# gathered post-processed fields (by name) and the probe files of both runs
function compare(case, gmesh_fn, names, probes; run_kw=(;))
    ser = nothing
    if rank == 0
        cd(joinpath(tmp, "serial")) do
            model, config, pp, fields = case(gmesh_fn())
            run!(model, with_config(config, pp); progress=false, run_kw...)
            ser = Dict(n => collect(f.values) for (n, f) ∈ zip(names, fields(pp)))
        end
    end
    gmesh = rank == 0 ? gmesh_fn() : nothing
    dm = distribute(gmesh; comm)
    dis = Dict{String,Any}()
    cd(joinpath(tmp, "dist")) do
        model, config, pp, fields = case(dm)
        run!(model, with_config(config, pp); progress=false, run_kw...)
        for (n, f) ∈ zip(names, fields(pp))
            dis[n] = gather(f, dm)
        end
    end
    MPI.Barrier(comm)
    files = [(read_probe(joinpath(tmp, "serial", p * ".csv")), read_probe(joinpath(tmp, "dist", p * ".csv")))
             for p ∈ probes]
    dm, ser, dis, files
end

# transient piso!: every accumulator type, two probes
nsteps = 20
function piso_case(mesh)
    model, config = incompressible_case(mesh, cavity_bcs; iterations=nsteps, time=Transient(),
        time_step=0.005, rtol=1e-12)
    (; U, p) = model.momentum
    rs = ReynoldsStress(U)
    pp = [FieldAverage(U; name="U_mean"), FieldRMS(U; name="U_rms"),
          FieldAverage(p; name="p_mean"), FieldRMS(p; name="p_rms"), rs, Qcriterion(U),
          Probe(U, mesh; location=[0.021, 0.013, 0.0], name="probe_U"),
          Probe(p, mesh; location=[0.005, 0.034, 0.0], name="probe_p")]
    fields(pp) = (pp[1].mean.x, pp[1].mean.y, pp[2].rms.x, pp[2].rms.y, pp[3].mean, pp[4].rms,
                  pp[5].rs.xx, pp[5].rs.xy, pp[5].rs.yy, pp[6].Q)
    model, config, pp, fields
end
piso_names = ["U_mean.x", "U_mean.y", "U_rms.x", "U_rms.y", "p_mean", "p_rms",
              "R.xx", "R.xy", "R.yy", "Q"]
dm, ser, dis, files = compare(piso_case, cavity_mesh, piso_names, ["probe_U", "probe_p"])

if rank == 0
    @testset "piso! post-processing vs serial" begin
        for n ∈ piso_names
            e = relerr(dis[n], ser[n])
            println("piso $n: $e")
            @test maximum(abs, ser[n]) > 0 # non-vacuous
            @test e < 1e-9
        end
        for (s, d) ∈ files
            @test length(d) == length(s) == nsteps # one row per step, written once
            @test first.(d) == first.(s)
            @test maximum(maximum(abs.(d[i][2:end] .- s[i][2:end])) for i ∈ eachindex(s)) < 1e-9
        end
    end
end

# steady simple!
function simple_case(mesh)
    model, config = incompressible_case(mesh, bfs_bcs; iterations=10, rtol=1e-12)
    (; U, p) = model.momentum
    pp = [FieldAverage(U; name="U_mean", start=3),
          Probe(p, mesh; location=[0.1, 0.01, 0.0], name="probe_simple")]
    model, config, pp, pp -> (pp[1].mean.x, pp[1].mean.y)
end
dm, ser, dis, files = compare(simple_case, bfs_mesh, ["U_mean.x", "U_mean.y"], ["probe_simple"])
if rank == 0
    @testset "simple! post-processing vs serial" begin
        for n ∈ ("U_mean.x", "U_mean.y")
            @test relerr(dis[n], ser[n]) < 1e-9
        end
        s, d = only(files)
        @test length(d) == length(s) == 10
        @test maximum(abs(d[i][2] - s[i][2]) for i ∈ eachindex(s)) < 1e-9
    end
end

# laplace! with decomposed OpenFOAM output of the post-processed field
liters = 5
function laplace_pp_case(mesh)
    model, config = laplace_case(mesh, box_bcs; iterations=liters, convergence=1e-20)
    config = Configuration(solvers=config.solvers, schemes=config.schemes,
        runtime=Runtime(iterations=liters, write_interval=liters, time_step=1),
        hardware=config.hardware, boundaries=config.boundaries)
    (; T) = model.energy
    pp = [FieldAverage(T; name="T_mean"), Probe(T, mesh; location=[0.3, 0.6, 0.45], name="probe_T")]
    model, config, pp, pp -> (pp[1].mean,)
end
dm, ser, dis, files = compare(laplace_pp_case, box_mesh, ["T_mean"], ["probe_T"];
    run_kw=(; output=OpenFOAM()))
if rank == 0
    @testset "laplace! post-processing vs serial" begin
        @test relerr(dis["T_mean"], ser["T_mean"]) < 1e-9
        s, d = only(files)
        @test length(d) == length(s) == liters
        @test maximum(abs(d[i][2] - s[i][2]) for i ∈ eachindex(s)) < 1e-9
    end
end
@testset "post-processed field in the processor directory (rank $rank)" begin
    f = joinpath(tmp, "dist", "processor$rank", "$liters", "T_mean")
    @test isfile(f)
    vals = XCALibre.Distribute._read_foam_lists(f, Float64)[1]
    @test length(vals) == dm.partition.n_owned
end

# rank 0 removes its temporary directory at exit: wait until every rank has read its files
MPI.Barrier(comm)
