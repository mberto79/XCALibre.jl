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

# threaded primitives must run inside another threaded region or from concurrent tasks
@testset "XVector nested and concurrent ($(Threads.nthreads()) threads)" begin
    n = 200_000
    a, b = rand(n), rand(n)
    x, y = XVector(copy(a)), XVector(copy(b))
    d = Krylov.kdot(n, x, y)
    nested = zeros(4)
    Threads.@threads for i ∈ 1:4
        nested[i] = Krylov.kdot(n, x, y)
    end
    @test all(==(d), nested)
    tasks = [Threads.@spawn Krylov.kdot(n, x, y) for _ ∈ 1:4]
    @test all(==(d), fetch.(tasks))
    A = SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1))
    sols = Vector{Any}(undef, 2)
    Threads.@threads for i ∈ 1:2
        ws = Krylov.CgWorkspace(Krylov.KrylovConstructor(XVector(copy(b))))
        Krylov.krylov_solve!(ws, A, XVector(copy(b)); rtol=1e-10, atol=0.0)
        sols[i] = Krylov.solution(ws)
    end
    @test sols[1] == sols[2] && isapprox(A*sols[1], b; rtol=1e-8)
end

@testset "AutoTune empty range" begin
    backend = XCALibre.CPU()
    @test XCALibre.Multithread._setup(backend, AutoTune(), 0)[2] ≥ 1
    hits = Int[]
    XCALibre.Multithread._xcal_foreach(i -> push!(hits, i), Int[], backend, AutoTune())
    @test isempty(hits)
end

# first touch changes where pages live, never values: every copy must equal its source
@testset "First touch ($(Threads.nthreads()) threads)" begin
    MT = XCALibre.Multithread
    n = 200_003
    a = rand(n)
    @test MT.first_touch_copy(a) == a && MT.first_touch_copy(a) !== a
    # entries cut by per-row ranges; 0:0 empty ranges fall back to the plain cut
    ranges = [3i-2:3i for i ∈ 1:n]
    v = rand(Int32, 3n)
    @test MT.first_touch_copy(v, ranges) == v
    @test MT.first_touch_copy(v, [i == 2 ? (0:0) : r for (i, r) ∈ enumerate(ranges)]) == v
    A = SparseXCSR(MT.SparseMatricesCSR.sparsecsr([1:n; 1:n-1], [1:n; 2:n], rand(2n - 1), n, n))
    B = first_touch(A)
    @test typeof(B) == typeof(A)
    @test all(getfield(parent(B), f) == getfield(parent(A), f) for f ∈ fieldnames(typeof(parent(A))))
    @test B*a == A*a
    mesh = FOAM3D_mesh(joinpath(pkgdir(XCALibre), "test", "grids", "OF_cavity_hex", "polyMesh"), scale=0.001)
    m = first_touch(mesh)
    @test typeof(m) == typeof(mesh)
    @test all(getfield(m, f) == getfield(mesh, f) for f ∈ fieldnames(typeof(mesh)))
    # construction sites copy only when enabled; activate_multithread resets the switch
    activate_multithread(XCALibre.CPU(static=true); first_touch=true)
    @test MT.first_touch_enabled() == (Threads.nthreads() > 1)
    @test MT.first_touch_zeros(XCALibre.CPU(), Float64, n) == zeros(n)
    @test ScalarField(mesh).values == zeros(length(mesh.cells))
    activate_multithread(XCALibre.CPU(static=true))
    @test !MT.first_touch_enabled()
end

# xcal_foreach on CPU(static=true) runs the solver's fixed chunks: same result as CPU(), and every
# chunk on one thread, so field loops touch the pages first touch placed for that thread
@testset "Static xcal_foreach ($(Threads.nthreads()) threads)" begin
    MT = XCALibre.Multithread
    config(backend) = (; hardware=Hardware(backend=backend, workgroup=AutoTune()))
    for n ∈ (1_000, 300_001)
        x = rand(n)
        ys, yd = zeros(n), zeros(n)
        xcal_foreach(ys, config(XCALibre.CPU(static=true))) do i
            ys[i] = 2x[i] + i
        end
        xcal_foreach(yd, config(XCALibre.CPU())) do i
            yd[i] = 2x[i] + i
        end
        @test ys == yd == 2 .* x .+ (1:n)
    end
    n, k = 300_001, Threads.nthreads()
    tid = zeros(Int, n)
    xcal_foreach(tid, config(XCALibre.CPU(static=true))) do i
        tid[i] = Threads.threadid()
    end
    owners = [unique(tid[MT._chunk(n, k, c)]) for c ∈ 1:k]
    @test all(length(o) == 1 for o ∈ owners)
    @test length(unique(first.(owners))) == k
end
