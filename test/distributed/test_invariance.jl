# Rank-invariance gate (P1-M19-S3): BFS psimple! with Jacobi, 50 iterations. Jacobi is
# partition-invariant, so residual histories agree across rank counts to Q1's four significant
# figures (observed 3e-6, D89). The n=1 run writes the reference; other rank counts compare.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

iterations = 50
refpath = joinpath(tempdir(), "xcalibre_invariance_bfs_jacobi.csv")

gmesh = rank == 0 ? bfs_mesh() : nothing
dm = distribute(gmesh; comm)
model, config = incompressible_case(dm, bfs_bcs; iterations)
residuals = run!(model, config)
hist = hcat(residuals.Ux, residuals.Uy, residuals.p)

if nranks == 1
    rank == 0 && open(io -> foreach(r -> println(io, join(r, ',')), eachrow(hist)), refpath, "w")
    rank == 0 && println("INVARIANCE reference written: $refpath")
elseif isfile(refpath)
    ref = reduce(vcat, (permutedims(parse.(Float64, split(l, ','))) for l ∈ eachline(refpath)))
    spread = maximum(abs.(hist .- ref) ./ max.(abs.(ref), eps()))
    rank == 0 && println("INVARIANCE n=$nranks max relative spread vs n=1: $spread")
    @testset "rank invariance n=$nranks (rank $rank)" begin
        @test size(ref) == size(hist)
        @test spread < 1e-4
    end
else
    rank == 0 && @warn "test_invariance: no n=1 reference at $refpath; run with --ranks=1,<n>"
end
MPI.Barrier(comm)
