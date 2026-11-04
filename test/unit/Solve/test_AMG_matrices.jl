# Correctness + loose regression suite over named structured matrices

using XCALibre, Test, LinearAlgebra, SparseArrays, SparseMatricesCSR

const SV = XCALibre.Solve

# MATRIX BUILDERS

function poisson1d(n=50)
    i = vcat(1:n, 1:n-1, 2:n)
    j = vcat(1:n, 2:n, 1:n-1)
    v = vcat(fill(2.0, n), fill(-1.0, 2*(n-1)))
    SparseXCSR(sparsecsr(i, j, v, n, n)), ones(n)
end

function poisson2d(m=10)
    n = m * m
    rows, cols, vals = Int[], Int[], Float64[]
    for r in 1:m, c in 1:m
        k = (r-1)*m + c
        push!(rows, k); push!(cols, k); push!(vals, 4.0)
        if c > 1; push!(rows, k); push!(cols, k-1); push!(vals, -1.0); end
        if c < m; push!(rows, k); push!(cols, k+1); push!(vals, -1.0); end
        if r > 1; push!(rows, k); push!(cols, k-m); push!(vals, -1.0); end
        if r < m; push!(rows, k); push!(cols, k+m); push!(vals, -1.0); end
    end
    SparseXCSR(sparsecsr(rows, cols, vals, n, n)), ones(n)
end

function anisotropic1d(n=50)
    # graded coefficients: d[i] = 2+sin(i), symmetric off-diag; SPD, diag dominant
    d = [2.0 + sin(Float64(i)) for i in 1:n]
    off = [-0.5 - 0.3*cos(Float64(i)) for i in 1:n-1]  # |off| < 0.9 < min(d); weakly diag dominant, SPD
    i = vcat(1:n, 1:n-1, 2:n)
    j = vcat(1:n, 2:n, 1:n-1)
    v = vcat(d, off, off)
    SparseXCSR(sparsecsr(i, j, v, n, n)), ones(n)
end

# SHARED HELPERS

function _make_config()
    setup = SolverSetup(
        solver=Cg(),
        preconditioner=AMG(SmoothedAggregation()),
        convergence=1e-7,
        relax=1.0,
        rtol=1e-8,
        atol=1e-8
    )
    Configuration(
        solvers=(T=setup,),
        schemes=(T=Schemes(),),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        hardware=Hardware(backend=CPU(), workgroup=64),
        boundaries=(T=(),)
    )
end

function _solve_amg(A, b, solver, outer, config; itmax=500)
    ws = _workspace(solver, b; solver=outer)
    ws = SV.update!(ws, A, solver, config)
    x = zeros(eltype(b), length(b))
    SV._amg_solve_mode!(ws, ws.hierarchy, solver, outer, SV.outer_operator(ws.hierarchy, A), b, x;
        itmax=itmax, atol=1e-8, rtol=1e-8)
    relres = norm(b - Array(parent(A)) * x) / norm(b)
    relres, ws
end

# TESTS

const _config = _make_config()

@testset "poisson1d" begin
    A, b = poisson1d()

    # A. CORRECTNESS (smoother sweep lives on poisson2d)
    for outer in (Cg(), Richardson()), coarsening in (SmoothedAggregation(), RugeStuben(), Pairwise())
        relres, _ = _solve_amg(A, b, AMG(coarsening; coarsest=8), outer, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION: Cg + SmoothedAggregation + Jacobi, coarsest=8
    reg_solver = AMG(SmoothedAggregation(); coarsest=8)
    _, ws_reg = _solve_amg(A, b, reg_solver, Cg(), _config)
    recorded_p1d = 8 # loose regression baseline (observed 8)
    @test ws_reg.iterations <= 2 * recorded_p1d + 20

    # C. WEIGHT CONVENTION
    ws_omega = SV.update!(_workspace(reg_solver, b), A, reg_solver, _config)
    level = ws_omega.hierarchy.levels[1]
    @test SV._level_jacobi_omega(Jacobi(), level) ≈ 4 / (3 * level.lambda_max)
end

@testset "poisson2d" begin
    A, b = poisson2d()

    # A. CORRECTNESS
    # outer iterations alternate along the grid so every (outer, coarsening) and (outer, smoother) pair is covered
    grid = Iterators.product((SmoothedAggregation(), RugeStuben(), Pairwise()),
                             (Jacobi(), Chebyshev(), SymGaussSeidel(), SSOR(1.0), GaussSeidel(), SOR(1.2)))
    for (i, (coarsening, smoother)) in enumerate(grid)
        outer = isodd(i) ? Cg() : Richardson()
        relres, _ = _solve_amg(A, b, AMG(coarsening; smoother, coarsest=8), outer, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION
    _, ws_reg = _solve_amg(A, b, AMG(SmoothedAggregation(); coarsest=8), Cg(), _config)
    recorded_p2d = 8 # loose regression baseline (observed 8)
    @test ws_reg.iterations <= 2 * recorded_p2d + 20
end

@testset "anisotropic1d" begin
    A, b = anisotropic1d()

    # A. CORRECTNESS
    for outer in (Cg(), Richardson()), coarsening in (SmoothedAggregation(), RugeStuben(), Pairwise())
        relres, _ = _solve_amg(A, b, AMG(coarsening; coarsest=8), outer, _config)
        @test relres < 1e-7
    end

    # B. REGRESSION
    _, ws_reg = _solve_amg(A, b, AMG(SmoothedAggregation(); coarsest=8), Cg(), _config)
    recorded_ani1d = 5 # loose regression baseline (observed 5)
    @test ws_reg.iterations <= 2 * recorded_ani1d + 20
end

@testset "sweeps" begin
    # unequal pre/post and per-level growth; Cg tolerates the non-symmetric cycle through its flexible update
    A, b = poisson2d(24)
    for sweeps in (Sweeps(0, 2), Sweeps(1, 1; grow=1)), outer in (Cg(), Richardson())
        relres, ws = _solve_amg(A, b, AMG(; sweeps, coarsest=8), outer, _config)
        @test length(ws.hierarchy.levels) > 2
        @test ws.converged && relres < 1e-7
    end
end

# 2D upwind convection-diffusion: non-symmetric
function convection2d(m=80)
    n = m * m
    rows, cols, vals = Int[], Int[], Float64[]
    for r in 1:m, c in 1:m
        k = (r-1)*m + c
        push!(rows, k); push!(cols, k); push!(vals, 6.0)
        if c > 1; push!(rows, k); push!(cols, k-1); push!(vals, -2.0); end # upwind side
        if c < m; push!(rows, k); push!(cols, k+1); push!(vals, -1.0); end
        if r > 1; push!(rows, k); push!(cols, k-m); push!(vals, -2.0); end
        if r < m; push!(rows, k); push!(cols, k+m); push!(vals, -1.0); end
    end
    SparseXCSR(sparsecsr(rows, cols, vals, n, n)), ones(n)
end

@testset "default AMG()" begin
    # 6400 rows > coarsest=4096: one Pairwise coarsening, then the coarsest solve
    A, b = poisson2d(80)
    relres, ws = _solve_amg(A, b, AMG(), Cg(), _config)
    @test ws.hierarchy isa SV.MatrixFreeHierarchy && ws.hierarchy.coarse_n < size(A, 1)
    @test ws.converged && relres < 1e-7
    Ac, bc = convection2d(80)
    relres, ws = _solve_amg(Ac, bc, AMG(), Bicgstab(), _config)
    @test !ws.hierarchy.is_symmetric
    @test ws.hierarchy.coarse_n < size(Ac, 1)
    @test ws.converged && relres < 1e-7
end

@testset "amg index type" begin
    S = XCALibre.Solve
    @test S._amg_index_type(100) == Int32
    @test S._amg_index_type(typemax(Int32)) == Int32
    @test S._amg_index_type(Int64(typemax(Int32)) + 1) == Int64
end

@testset "mf multilevel cycle (5b-B)" begin
    # Full-hierarchy matrix-free V-cycle (P/R erased at EVERY level) must match the independent host
    # oracle built from reference sparse P_l/A_l, across multiple coarsening depths.
    A, _ = poisson2d(24)
    for (ml, pre, post) in ((1, 2, 2), (2, 2, 2), (3, 2, 2), (2, 1, 1), (2, 0, 2))
        res = SV.validate_matrix_free_cycle(A, ml, CPU(); pre=pre, post=post, max_coarse=64)
        @test res.relerr < 1e-12
        @test res.levels >= 2
        @test res.vram_saved_bytes > 0
    end
end

@testset "mf scale_correction (sc)" begin
    # GAMG scale_correction (sf=(r_l·c)/(c·Ac)) in the MF up-sweep must (1) match the host oracle
    # extended with the SAME sf to ~eps (sf arithmetic + sc buffer mgmt), across depths/sweeps, and
    # (2) materially improve the stationary V-cycle (sc on converges where plain-additive stalls).
    A, _ = poisson2d(24)
    for (ml, pre, post) in ((1, 2, 2), (2, 2, 2), (3, 2, 2), (2, 1, 1))
        res = SV.validate_matrix_free_cycle(A, ml, CPU(); pre=pre, post=post, max_coarse=64,
                                               scale_correction=true)
        @test res.relerr < 1e-12
    end
    Ac, _ = poisson2d(40)
    off = SV.validate_matrix_free_convergence(Ac, 2, CPU(); scale_correction=false, itmax=80)
    on  = SV.validate_matrix_free_convergence(Ac, 2, CPU(); scale_correction=true,  itmax=80)
    @test on.converged
    @test on.iters <= off.iters
    # fused_top>0 with sc: Ac comes from the matrix-free Galerkin chain -> must equal the
    # materialized sc cycle exactly (same operator, same sf)
    for k in (1, 2)
        hk = SV.build_matrix_free_hierarchy(A, 2, CPU(); max_coarse=64, fused_top=k, scale_correction=true)
        h0 = SV.build_matrix_free_hierarchy(A, 2, CPU(); max_coarse=64, fused_top=0, scale_correction=true)
        bsc = rand(size(A, 1))
        xk = Array(SV.matrix_free_cycle!(hk.st, copy(bsc)))
        x0 = Array(SV.matrix_free_cycle!(h0.st, copy(bsc)))
        @test maximum(abs.(xk .- x0)) / maximum(abs.(x0)) < 1e-12
    end
end

@testset "mf top-k matrix-free Galerkin (5b-C)" begin
    # Top-k coarse operators applied matrix-free (A_l = R-chain·A0·P-chain) must reproduce the
    # materialized cycle EXACTLY (same operator), validating the prolong/restrict chain composition.
    A, _ = poisson2d(24)
    for ml in (2, 3), k in (1, 2)
        res = SV.validate_fused_operator(A, ml, CPU(), k; pre=2, post=2, max_coarse=64)
        @test res.relerr < 1e-12
        @test res.fused_top >= 1
        @test res.vram_erased_bytes > 0
    end
end

@testset "mf coarse refresh (G1)" begin
    # Frozen-sparsity device refresh (refresh A1->A2) must reproduce a full rebuild on A2 to ~eps at
    # every level + the coarsest, including the recomputed omega. A2 shares A1's sparsity, different
    # values — this tests the frozen pattern is adequate for changed values, not just the arithmetic.
    A1, _ = poisson2d(24)
    A2 = deepcopy(A1)
    let rp = SV._rowptr(A2), cv = SV._colval(A2), nz = SV._nzval(A2)
        for i in 1:SV._m(A2), p in rp[i]:(rp[i+1]-1)
            nz[p] *= (cv[p] == i ? 1.25 : 0.85)
        end
    end
    for ml in (1, 2, 3)
        res = SV.validate_refresh(A1, A2, ml, CPU(); max_coarse=64)
        @test res.relerr < 1e-12
        @test res.omega_relerr < 1e-10
        @test res.levels >= 2
    end
    # composed (fused top-k) levels refresh through the gathered member lists; a refresh is
    # bitwise repeatable (one work item per coarse row, no atomic adds)
    for ml in (2, 3), k in (1, 2)
        res = SV.validate_refresh(A1, A2, ml, CPU(); max_coarse=64, fused_top=k)
        @test res.relerr < 1e-12
        @test res.omega_relerr < 1e-10
    end
    B1, _ = poisson2d(96) # many workgroups per refresh launch
    B2 = deepcopy(B1)
    SV._nzval(B2) .*= 0.9
    for k in (0, 1)
        h = SV.build_matrix_free_hierarchy(B1, 2, CPU(); max_coarse=64, fused_top=k)
        plan = SV.build_refresh_plan(h)
        coarse(h) = [(copy(Array(SV._nzval(lv.A))), copy(Array(lv.invdiag))) for lv in h.st.levels]
        SV.refresh_matrix_free_hierarchy!(h.st, plan, B2)
        first_values = (coarse(h), copy(Array(SV._nzval(plan.coarsest_csr))))
        @test all(1:20) do _
            SV.refresh_matrix_free_hierarchy!(h.st, plan, B2)
            (coarse(h), Array(SV._nzval(plan.coarsest_csr))) == first_values
        end
    end
    # split precision (finest F64 / coarse F32): refresh must reproduce the F64 build to coarse-F32
    # accuracy and stay iteration-equivalent (the GPU type match is guarded in test_AMG.jl)
    let ml = 2
        rm = SV.validate_refresh(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float32)
        @test rm.relerr < 1e-5
        c64 = SV.validate_refresh_convergence(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float64)
        c32 = SV.validate_refresh_convergence(A1, A2, ml, CPU(); max_coarse=64, coarse_storage=Float32)
        @test c32.converged && c32.iters <= c64.iters + 1
    end
end

@testset "mf zero-alloc coarse solve (G2)" begin
    # The coarse handoff must allocate nothing that scales with coarse_n (was Array+backslash+adapt
    # every cycle). Both branches: device dense-inverse GEMV (small coarse) and reusable-buffer host
    # LU (large coarse). Per-cycle alloc must be constant; correctness preserved to ~eps either way.
    A, _ = poisson2d(24)
    for cmr in (512, 0)  # 512 -> GEMV branch (coarse_n=64<=512); 0 -> host-LU branch
        al = SV.measure_cycle_allocations(A, 2, CPU(); max_coarse=64, coarse_max_rows=cmr)
        @test al.constant
        @test al.branch == (cmr == 0 ? :host_lu : :gemv)
        sp = SV.validate_matrix_free_cycle(A, 2, CPU(); max_coarse=64, coarse_max_rows=cmr)
        @test sp.relerr < 1e-12
    end
end

@testset "T1 F32 512^2 regression (P13 residual-form lock)" begin
    # All-F32 Cg + Pairwise + Float32 coarse storage: the residual-form Jacobi kernel keeps the F32 cycle stable.
    # True residual floors at eps_f32*cond(512^2 Poisson), so assert on the reported convergence, not a recomputed one.
    Ad, _ = poisson2d(512)
    i, j, v = findnz(parent(Ad)); n = size(Ad, 1)
    Af = SparseXCSR(sparsecsr(i, j, Float32.(v), n, n))
    bf = ones(Float32, n)
    s = SV._with_storage(AMG(Pairwise(2); coarsest=1024), Float32)
    ws = _workspace(s, bf)
    ws = SV.update!(ws, Af, s, _config)
    @test eltype(SV._nzval(ws.hierarchy.levels[1].A)) === Float32
    x = zeros(Float32, n)
    SV._amg_solve_mode!(ws, ws.hierarchy, s, Cg(), SV.outer_operator(ws.hierarchy, Af), bf, x; itmax=100, atol=0f0, rtol=1f-5)
    @test ws.converged
    @test ws.iterations <= 40
    @test ws.last_relative_residual <= 1e-5
end

@testset "T2 CPU fuse" begin
    # fused coarse levels apply the same operator through the finer matrix: same iterations (±1) and solution
    A, b = poisson2d(64)
    runfl(fl; smoother=Jacobi()) = begin
        s = AMG(Pairwise(1); smoother, coarsest=64, native=NativeAMG(fuse=fl))
        ws = SV.update!(_workspace(s, b), A, s, _config)
        x = zeros(eltype(b), length(b))
        SV._amg_solve_mode!(ws, ws.hierarchy, s, Cg(), SV.outer_operator(ws.hierarchy, A), b, x; itmax=200, atol=1e-8, rtol=1e-8)
        (ws=ws, x=x)
    end
    r0 = runfl(0)
    @test r0.ws.hierarchy isa SV.MatrixFreeHierarchy && length(r0.ws.hierarchy.levels) > 3
    for r in (runfl(1), runfl(2), runfl(2; smoother=Chebyshev()))
        @test r.ws.converged
        @test norm(b - Array(parent(A)) * r.x) / norm(b) < 1e-7
    end
    for fl in (1, 2)
        r = runfl(fl)
        @test r.ws.hierarchy.fused_top == fl
        @test abs(r.ws.iterations - r0.ws.iterations) <= 1
        @test norm(r.x - r0.x) / norm(r0.x) < 1e-8
    end
    @test_throws ArgumentError AMG(SmoothedAggregation(); native=NativeAMG(fuse=1))
    @test_throws ArgumentError AMG(smoother=GaussSeidel(), native=NativeAMG(fuse=1))
end

# equal couplings pair across the faces the bias favours (here y), whatever the cell numbering,
# so uniform 2D cells form 2×2 boxes after two pairing passes
@testset "Pairwise face-normal bias builds box-shaped aggregates" begin
    n = 16
    for id in ((x, y) -> x + (y - 1) * n, (x, y) -> y + (x - 1) * n)
        xy = Vector{NTuple{2,Int}}(undef, n^2)
        foreach(c -> xy[id(c...)] = c, Iterators.product(1:n, 1:n))
        nbrs = [Int[] for _ in 1:n^2]; fids = [Int[] for _ in 1:n^2]; normals = NTuple{3,Float64}[]
        for (x, y) in xy, (dx, dy) in ((1, 0), (0, 1))
            (x + dx > n || y + dy > n) && continue
            push!(normals, (dx, dy, 0.0))
            a, b = id(x, y), id(x + dx, y + dy)
            push!(nbrs[a], b); push!(fids[a], length(normals)); push!(nbrs[b], a); push!(fids[b], length(normals))
        end
        I1, J1, V1 = Int[], Int[], Float64[]
        for a in 1:n^2
            push!(I1, a); push!(J1, a); push!(V1, 4)
            foreach(b -> (push!(I1, a); push!(J1, b); push!(V1, -1)), nbrs[a])
        end
        A = SV._amg_matrix(SparseMatricesCSR.sparsecsr(I1, J1, V1, n^2, n^2))
        lens = cumsum(vcat(0, length.(nbrs)))
        mesh = (cell_faces_range=[(lens[a] + 1):lens[a + 1] for a in 1:n^2], cell_faces=reduce(vcat, fids),
            cell_neighbours=reduce(vcat, nbrs), face_normal=normals)
        bias = SV._pairwise_bias(A, mesh)
        pairs, np = SV._pairwise_aggregates(A, 1, bias)
        @test np == n^2 ÷ 2 && all(g -> allequal(first.(xy[findall(==(g), pairs)])), 1:np)
        agg, nc = SV._pairwise_aggregates(A, 2, bias)
        @test nc == n^2 ÷ 4
        spans(c) = (extrema(first.(xy[c])), extrema(last.(xy[c])))
        @test all(g -> (c = findall(==(g), agg); length(c) == 4 && all(e -> e[2] - e[1] == 1, spans(c))), 1:nc)
    end
end
