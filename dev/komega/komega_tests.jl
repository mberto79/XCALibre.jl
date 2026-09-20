# The KOmega and assembly-touching cases from test/runtests.jl, run on their own.
# Preamble copied from test/runtests.jl so the case files see what they expect.
using XCALibre
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using StaticArrays
using Statistics
using Test

workgroupsize(mesh) = length(mesh.cells) ÷ Threads.nthreads()
TEST_CASES_DIR = pkgdir(XCALibre, "test/0_TEST_CASES")

@testset verbose = true "KOmega + assembly-touching cases" begin
    for t in ["2d_incompressible_flatplate_KOmega_lowRe.jl",
              "2d_incompressible_flatplate_KOmega_HighRe.jl",
              "2d_incompressible_transient_KOmega_BFS_lowRe.jl",
              "2d_compressible_KOmega_flatplate_fixedT.jl",
              "2d_incompressible_laminar_BFS.jl",
              "3d_incompressible_laminar_BFS.jl"]
        @testset "$t" begin include(joinpath(TEST_CASES_DIR, t)) end
    end
end
