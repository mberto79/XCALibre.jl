# Spawns each distributed test file under mpiexec at the requested rank counts.
using Test
using MPI, PETSc # both are declared test dependencies: a load failure here is a regression

# after MPI moves behind an extension, `using` bumps the world age inside this function
mpiexec_available() = try
    success(`$(Base.invokelatest(MPI.mpiexec)) --version`)
catch err
    @warn "distributed gate skipped: mpiexec is not runnable from this environment" err
    false
end

function run_mpi_tests(files; ranks=[1, 2])
    dir = @__DIR__
    project = dirname(Base.active_project())
    julia = Base.julia_cmd()
    # precompile serially first: parallel first-use of the same cache races
    run(`$julia --project=$project --startup-file=no -e "using XCALibre, MPI, PETSc, Test"`)
    # the outer set keeps a failing (file, n) from aborting the rest: a top-level for-testset
    # throws at the end of the iteration that fails
    @testset "distributed suite" begin
        @testset "mpi $file n=$n" for file ∈ files, n ∈ ranks
            cmd = `$(Base.invokelatest(MPI.mpiexec)) -n $n $julia --project=$project --startup-file=no $(joinpath(dir, file))`
            out = IOBuffer()
            ok = success(pipeline(cmd; stdout=out, stderr=out))
            ok || print(String(take!(out)))
            @test ok
        end
    end
end
