# Run named serial-suite files with runtests.jl's preamble; one testset per file, summary per file.
#   julia --project=test -t 1 dev/scripts/suite_file.jl test/unit_test_laplace.jl test/0_TEST_CASES/2d_EFM.jl ...
using XCALibre, LinearAlgebra, SparseArrays, SparseMatricesCSR, StaticArrays, Statistics, Test
workgroupsize(mesh) = length(mesh.cells) ÷ Threads.nthreads()
TEST_CASES_DIR = pkgdir(XCALibre, "test/0_TEST_CASES")
failed = String[]
for f in ARGS
    t0 = time()
    try
        @testset "$(basename(f))" begin
            include(abspath(f))
        end
        println("SUITE ", basename(f), " PASS s=", round(time() - t0, digits=1))
    catch e
        println("SUITE ", basename(f), " FAIL s=", round(time() - t0, digits=1), " ", sprint(showerror, e)[1:min(end, 300)])
        push!(failed, f)
    end
end
isempty(failed) || (println("FAILED: ", join(failed, " ")); exit(1))
