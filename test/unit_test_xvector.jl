using XCALibre, LinearAlgebra, Test
const Krylov = XCALibre.Multithread.Krylov
const XVector = XCALibre.Multithread.XVector

# XVector primitives against plain Vector; sums differ only in reduction order
@testset "XVector primitives ($(Threads.nthreads()) threads)" begin
    for T ∈ (Float64, Float32), n ∈ (1, 7, 100_003)
        a, b = rand(T, n), rand(T, n)
        x, y = XVector(copy(a)), XVector(copy(b))
        s, t = T(0.7), T(-1.3)
        tol = 10eps(T)*n
        @test similar(x) isa XVector{T} && length(similar(x, 0)) == 0
        @test x == a && pointer(x) == pointer(parent(x))
        @test isapprox(Krylov.kdot(n, x, y), dot(a, b); rtol=tol)
        @test Krylov.kdot(n, x, y) == Krylov.kdot(n, x, y)
        @test isapprox(Krylov.knorm(n, x), norm(a); rtol=tol)
        @test Krylov.kaxpy!(n, s, x, copy(y)) ≈ b .+ s .* a
        @test Krylov.kaxpby!(n, s, x, t, XVector(copy(b))) ≈ s .* a .+ t .* b
        @test Krylov.kscal!(n, s, XVector(copy(a))) ≈ s .* a
        @test Krylov.kscalcopy!(n, similar(x), s, x) ≈ s .* a
        @test Krylov.kdivcopy!(n, similar(x), x, s) ≈ a ./ s
        @test Krylov.kcopy!(n, similar(x), x) == a
        @test all(==(s), Krylov.kfill!(similar(x), s))
    end
    # a Krylov workspace built from an XVector keeps XVector storage and solves as a Vector does
    n = 50
    A = SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1))
    b = rand(n)
    ws = Krylov.CgWorkspace(Krylov.KrylovConstructor(XVector(copy(b))))
    @test ws.x isa XVector{Float64}
    Krylov.krylov_solve!(ws, A, XVector(copy(b)); rtol=1e-12, atol=0.0)
    @test Krylov.solution(ws) ≈ A \ b
end

@testset "SparseXCSR product" begin
    n = 100_003
    A = SparseXCSR(XCALibre.Multithread.SparseMatricesCSR.sparsecsr(collect(1:n), collect(1:n), fill(2.0, n), n, n))
    x = rand(n)
    @test A*x == 2 .* x
    @test A*XVector(copy(x)) == 2 .* x
end
