using XCALibre
using Test
using StaticArrays
using LinearAlgebra
using SparseArrays

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

# uniform oblique velocity (both grids span a 1000 x 1000 square)
u = SVector(1.0, 0.6, 0.0)
uniform_flux(mesh) = begin
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

interior_cells(mesh) = setdiff(1:length(mesh.cells), unique(mesh.boundary_cellsID))

# on triangles the upwind face values are not exact for a linear field, so these tests
# discriminate: the gradient reconstruction recovers ∇·(uφ) = u·∇φ exactly in every interior cell
trig = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv")))
quad = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "quad40.unv")))

@testset "LinearUpwind second-order consistency (scalar)" begin
    mesh = trig
    mdotf = uniform_flux(mesh)
    interior = interior_cells(mesh)
    a = SVector(0.3e-3, -0.7e-3, 0.0)
    phi = ScalarField(mesh)
    phi.values .= [a⋅c + 2.0 for c ∈ mesh.cells.centre]
    gradphi = VectorField(mesh)
    for i ∈ eachindex(mesh.cells)
        gradphi.x[i], gradphi.y[i], gradphi.z[i] = a
    end
    exact = (u⋅a) .* mesh.cells.volume

    for (scheme, isexact) ∈ ((LinearUpwind, true), (Upwind, false))
        eqn = (
            Divergence{scheme}(mdotf, phi, gradphi) == Source(ConstantScalar(0.0))
        ) → ScalarEquation(phi, ())
        discretise!(eqn, phi, config)
        r = residual(eqn.equation.A, phi.values, eqn.equation.b)
        @test isapprox(r[interior], exact[interior]; rtol=1e-10) == isexact
    end
end

@testset "LinearUpwind second-order consistency (vector)" begin
    mesh = trig
    mdotf = uniform_flux(mesh)
    interior = interior_cells(mesh)
    volume = mesh.cells.volume
    G = @SMatrix [0.2e-3 -0.5e-3 0.0; 0.4e-3 0.1e-3 0.0; 0.0 0.0 0.0] # G[i,j] = ∂Uᵢ/∂xⱼ
    U = VectorField(mesh)
    gradU = TensorField(mesh)
    for i ∈ eachindex(mesh.cells)
        U.x[i], U.y[i], U.z[i] = G*mesh.cells.centre[i] + SVector(1.0, 2.0, 0.0)
        gradU[i] = G
    end
    exact = G*u

    for (scheme, isexact) ∈ ((LinearUpwind, true), (Upwind, false))
        eqn = (
            Divergence{scheme}(mdotf, U, gradU) == Source(VectorField(mesh))
        ) → VectorEquation(U, ())
        discretise!(eqn, U, config)
        (; A0, bx, by) = eqn.equation
        rx = residual(A0, U.x.values, bx)
        ry = residual(A0, U.y.values, by)
        @test isapprox(rx[interior], exact[1] .* volume[interior]; rtol=1e-10) == isexact
        @test isapprox(ry[interior], exact[2] .* volume[interior]; rtol=1e-10) == isexact
    end
end

@testset "LimitedLinearUpwind is inactive for a linear field" begin
    mesh = quad
    mdotf = uniform_flux(mesh)
    a = SVector(0.3e-3, -0.7e-3, 0.0)
    phi = ScalarField(mesh)
    phi.values .= [a⋅c + 2.0 for c ∈ mesh.cells.centre]
    gradphi = VectorField(mesh)
    for i ∈ eachindex(mesh.cells)
        gradphi.x[i], gradphi.y[i], gradphi.z[i] = a
    end
    b = map((LinearUpwind, LimitedLinearUpwind)) do scheme
        eqn = (
            Divergence{scheme}(mdotf, phi, gradphi) == Source(ConstantScalar(0.0))
        ) → ScalarEquation(phi, ())
        discretise!(eqn, phi, config)
        copy(eqn.equation.b)
    end
    @test maximum(abs, b[1]) > 0
    @test b[2] ≈ b[1]
end

@testset "Boundary conditions reuse the Upwind coefficients" begin
    mesh = quad
    mdotf = uniform_flux(mesh)
    BCs = assign(
        region = mesh,
        (
            U = [
                Dirichlet(:inlet, [1.0, 0.0, 0.0]),
                Wall(:bottom, [0.0, 0.0, 0.0]),
                Zerogradient(:outlet),
                Symmetry(:top),
            ],
        )
    )
    U = VectorField(mesh)
    U.x.values .= rand(length(mesh.cells))
    U.y.values .= rand(length(mesh.cells))
    gradU = TensorField(mesh) # zero gradient: no face correction
    assembled = map((Upwind, LinearUpwind, LimitedLinearUpwind)) do scheme
        eqn = (
            Divergence{scheme}(mdotf, U, gradU) == Source(VectorField(mesh))
        ) → VectorEquation(U, BCs.U)
        discretise!(eqn, U, config)
        update_equation!(eqn, config)
        apply_boundary_conditions!(eqn, BCs.U, XDir(), 0.0, config)
        copy(get_sparse_fields(eqn.equation.A)[1]), copy(eqn.equation.bx)
    end
    for (nzval, bx) ∈ assembled[2:3]
        @test nzval == assembled[1][1]
        @test bx == assembled[1][2]
    end
end

@testset "Advected step: limited reconstruction stays bounded" begin
    mesh = quad
    mdotf = uniform_flux(mesh)
    ncells = length(mesh.cells)
    BCs = assign(
        region = mesh,
        (
            phi = [
                Dirichlet(:inlet, 1.0),
                Dirichlet(:bottom, 0.0),
                Zerogradient(:outlet),
                Zerogradient(:top),
            ],
        )
    )
    phiBCs = BCs.phi

    # deferred-correction iterations of the steady advection equation, solved directly
    solve_step(scheme, limiter) = begin
        phi = ScalarField(mesh)
        phif = FaceScalarField(mesh)
        gradphi = Grad{Gauss}(phi)
        eqn = (
            Divergence{scheme}(mdotf, phi, gradphi.result) == Source(ConstantScalar(0.0))
        ) → ScalarEquation(phi, phiBCs)
        change = Inf
        for iteration ∈ 1:200
            grad!(gradphi, phif, phi, phiBCs, 0.0, config)
            limit_gradient!(limiter, gradphi, phi, config)
            discretise!(eqn, phi, config)
            apply_boundary_conditions!(eqn, phiBCs, nothing, 0.0, config)
            nzval, colval, rowptr = get_sparse_fields(eqn.equation.A)
            rows = [i for i ∈ 1:ncells for _ ∈ rowptr[i]:rowptr[i+1]-1]
            new = sparse(rows, colval, nzval, ncells, ncells)\eqn.equation.b
            change = maximum(abs, new - phi.values)
            phi.values .= new
        end
        overshoot = max(maximum(phi.values) - 1, -minimum(phi.values))
        overshoot, change
    end

    unlimited, _ = solve_step(LinearUpwind, nothing)
    limited_gradient, _ = solve_step(LinearUpwind, CellBased())
    bounded, change = solve_step(LimitedLinearUpwind, nothing)

    @test unlimited > 1e-2
    @test limited_gradient < unlimited/4
    @test change < 1e-6
    @test bounded < 1e-6
end
