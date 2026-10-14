using XCALibre
using Test
using StaticArrays
using LinearAlgebra

# LUSTDC: LUST with an upwind matrix and the 3/4 (linear - upwind) face value as an explicit source

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

uniform_flux(mesh, u) = begin
    Uf = FaceVectorField(mesh)
    for fID ∈ eachindex(mesh.faces)
        Uf.x[fID], Uf.y[fID], Uf.z[fID] = u
    end
    mdotf = FaceScalarField(mesh)
    flux!(mdotf, Uf, config)
    mdotf
end

# row-wise A*x - b over the assembled sparse matrix
residual(A, x, b) = begin
    nzval, colval, rowptr = get_sparse_fields(A)
    [sum(nzval[k]*x[colval[k]] for k ∈ rowptr[i]:rowptr[i+1]-1) - b[i] for i ∈ eachindex(b)]
end

# assembled scalar system (A, b) with boundary conditions applied
assemble_scalar(scheme, mesh, mdotf, phi, BCs) = begin
    eqn = (Divergence{scheme}(mdotf, phi) == Source(ScalarField(mesh))) → ScalarEquation(phi, BCs)
    discretise!(eqn, phi, config)
    apply_boundary_conditions!(eqn, BCs, nothing, 0.0, config)
    eqn.equation.A, eqn.equation.b
end

# assembled x and y systems of a vector equation with boundary conditions applied
assemble_vector(scheme, mesh, mdotf, U, BCs) = begin
    eqn = (Divergence{scheme}(mdotf, U) == Source(VectorField(mesh))) → VectorEquation(U, BCs)
    discretise!(eqn, U, config)
    out = map((XDir(), YDir())) do dir
        update_equation!(eqn, config)
        apply_boundary_conditions!(eqn, BCs, dir, 0.0, config)
        copy(get_sparse_fields(eqn.equation.A)[1]), copy(dir isa XDir ? eqn.equation.bx : eqn.equation.by)
    end
    eqn.equation.A, out
end

trig = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv")))
quad = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "quad40.unv")))
u = SVector(1.0, 0.6, 0.0)

scalar_sets = (
    [Dirichlet(:inlet, 1.0), Dirichlet(:bottom, 0.5), Zerogradient(:outlet), Zerogradient(:top)],
    [Neumann(:inlet, 0.1), Extrapolated(:bottom), Robin(:outlet; a=1.0, b=0.5, value=0.2), Wall(:top, 0.0)],
    [Slip(:inlet), Symmetry(:bottom), Outlet(:outlet), Symmetry(:top)],
)
vector_sets = (
    [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0]), Zerogradient(:outlet), Symmetry(:top)],
    [Slip(:inlet), Symmetry(:bottom), Outlet(:outlet), Symmetry(:top)],
)

@testset "LUSTDC has the residual of LUST (same fixed point)" begin
    for mesh ∈ (trig, quad)
        mdotf = uniform_flux(mesh, u)
        n = length(mesh.cells)
        x = [c[1] for c ∈ mesh.cells.centre]; y = [c[2] for c ∈ mesh.cells.centre]
        for bcs ∈ scalar_sets
            BCs = assign(region=mesh, (phi = bcs,)).phi
            phi = ScalarField(mesh)
            phi.values .= 0.3 .+ sin.(3e-3 .* x) .* cos.(2e-3 .* y) .+ 0.01 .* sin.(1:n)
            r = map((LUST, LUSTDC)) do scheme
                A, b = assemble_scalar(scheme, mesh, mdotf, phi, BCs)
                residual(A, phi.values, b)
            end
            @test maximum(abs, r[2] - r[1]) <= 1e-10*maximum(abs, r[1])
        end
        for bcs ∈ vector_sets
            BCs = assign(region=mesh, (U = bcs,)).U
            U = VectorField(mesh)
            U.x.values .= 1 .+ 0.2 .* sin.(3e-3 .* x); U.y.values .= 0.1 .* cos.(2e-3 .* y .+ (1:n))
            # A holds the coefficients of the last direction: restore each direction's copy
            for (k, comp) ∈ enumerate((U.x.values, U.y.values))
                rr = map((LUST, LUSTDC)) do scheme
                    A, sys = assemble_vector(scheme, mesh, mdotf, U, BCs)
                    nzval = get_sparse_fields(A)[1]
                    nzval .= sys[k][1]
                    residual(A, comp, sys[k][2])
                end
                @test maximum(abs, rr[2] - rr[1]) <= 1e-10*maximum(abs, rr[1])
            end
        end
    end
end

@testset "LUSTDC: linear part exact for a linear field, matrix is Upwind's" begin
    # uniform quads: face centres sit at the weighted midpoint, so Linear is exact for a linear field
    mesh = quad
    mdotf = uniform_flux(mesh, u)
    interior = setdiff(1:length(mesh.cells), unique(mesh.boundary_cellsID))
    a = SVector(0.3e-3, -0.7e-3, 0.0)
    phi = ScalarField(mesh)
    phi.values .= [a⋅c + 2.0 for c ∈ mesh.cells.centre]
    r = map((Linear, Upwind, LUSTDC)) do scheme
        A, b = assemble_scalar(scheme, mesh, mdotf, phi, ())
        residual(A, phi.values, b)
    end
    exact = (u⋅a) .* mesh.cells.volume
    @test r[1][interior] ≈ exact[interior] rtol=1e-10
    @test r[3][interior] ≈ 0.75 .* r[1][interior] .+ 0.25 .* r[2][interior] rtol=1e-10
    Aup, _ = assemble_scalar(Upwind, mesh, mdotf, phi, ())
    nz_up = copy(get_sparse_fields(Aup)[1])
    Adc, _ = assemble_scalar(LUSTDC, mesh, mdotf, phi, ())
    @test get_sparse_fields(Adc)[1] == nz_up
end

@testset "LUSTDC keeps an M-matrix where LUST does not" begin
    # pure convection (cell Peclet number infinite): LUST has positive off-diagonals
    for mesh ∈ (trig, quad)
        mdotf = uniform_flux(mesh, u)
        BCs = assign(region=mesh, (phi = scalar_sets[1],)).phi
        phi = ScalarField(mesh)
        offdiag_max = map((LUST, LUSTDC)) do scheme
            A, _ = assemble_scalar(scheme, mesh, mdotf, phi, BCs)
            nzval, colval, rowptr = get_sparse_fields(A)
            maximum(nzval[k] for i ∈ eachindex(mesh.cells) for k ∈ rowptr[i]:rowptr[i+1]-1 if colval[k] != i)
        end
        @test offdiag_max[1] > 0
        @test offdiag_max[2] <= 0
    end
end

@testset "LUSTDC: steady advection converges to the LUST solution" begin
    mesh = quad
    mdotf = uniform_flux(mesh, u)
    n = length(mesh.cells)
    BCs = assign(region=mesh, (phi = [Dirichlet(:inlet, 1.0), Dirichlet(:bottom, 0.0),
        Zerogradient(:outlet), Zerogradient(:top)],)).phi
    dense(A) = begin
        nzval, colval, rowptr = get_sparse_fields(A)
        M = zeros(n, n)
        for i ∈ 1:n, k ∈ rowptr[i]:rowptr[i+1]-1; M[i, colval[k]] += nzval[k]; end
        M
    end
    phi = ScalarField(mesh)
    A, b = assemble_scalar(LUST, mesh, mdotf, phi, BCs)
    ref = dense(A)\b
    # deferred-correction iteration (direct solves), started from zero
    phi.values .= 0
    change = Inf
    for iteration ∈ 1:200
        A, b = assemble_scalar(LUSTDC, mesh, mdotf, phi, BCs)
        new = dense(A)\b
        change = maximum(abs, new - phi.values)
        phi.values .= new
        change < 1e-12 && break
    end
    @test change < 1e-10
    @test maximum(abs, phi.values - ref) < 1e-8
    # the LUST step overshoots; LUSTDC converges to the same overshoot (no extra bounding)
    @test maximum(ref) > 1
end

@testset "LUSTDC needs no gradient and keeps mesh types" begin
    term = Divergence{LUSTDC}(uniform_flux(quad, u), ScalarField(quad))
    @test term.grad === nothing
    @test XCALibre.Discretise.has_face_source(typeof(term))
    @test !XCALibre.Discretise.is_second_order(LUSTDC)
    for (TF, TI) ∈ ((Float32, Int32), (Float64, Int64))
        m = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv"), float_type=TF, integer_type=TI))
        fl = FaceVectorField(m); fl.x.values .= TF(1); fl.y.values .= TF(0.6)
        flux = FaceScalarField(m)
        flux!(flux, fl, config)
        phi = ScalarField(m)
        phi.values .= [TF(2) + TF(1e-3)*c[1] for c ∈ m.cells.centre]
        BCs = assign(region=m, (phi = scalar_sets[1],)).phi
        r = map((LUST, LUSTDC)) do scheme
            A, b = assemble_scalar(scheme, m, flux, phi, BCs)
            @test eltype(b) == TF
            @test eltype(get_sparse_fields(A)[1]) == TF
            residual(A, phi.values, b)
        end
        @test eltype(r[2]) == TF
        @test maximum(abs, r[2] - r[1]) <= 100*eps(TF)*maximum(abs, r[1])
    end
end
