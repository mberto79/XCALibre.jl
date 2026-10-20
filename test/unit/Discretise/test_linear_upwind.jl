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
            change < 1e-10 && break
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

@testset "Every boundary condition reuses the Upwind coefficients" begin
    mesh = quad
    mdotf = uniform_flux(mesh)
    ncells = length(mesh.cells)
    sets = (
        scalar = (
            [Dirichlet(:inlet, 1.0), Dirichlet(:bottom, 0.5), Zerogradient(:outlet), Zerogradient(:top)],
            [Neumann(:inlet, 0.1), Extrapolated(:bottom), Robin(:outlet; a=1.0, b=0.5, value=0.2),
                Wall(:top, 0.0)],
            [Slip(:inlet), Symmetry(:bottom), Outlet(:outlet), Symmetry(:top)],
        ),
        vector = (
            [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0]), Zerogradient(:outlet),
                Symmetry(:top)],
            [Slip(:inlet), Symmetry(:bottom), Outlet(:outlet), Symmetry(:top)],
        ),
    )
    assemble(scheme, bcs, vector) = begin
        BCs = assign(region=mesh, (phi = bcs,)).phi
        if vector
            phi = VectorField(mesh)
            phi.x.values .= 0.3 .+ 0.001 .* (1:ncells); phi.y.values .= 0.1 .* sin.(1:ncells)
            eqn = (Divergence{scheme}(mdotf, phi, TensorField(mesh)) == Source(VectorField(mesh))
                ) → VectorEquation(phi, BCs)
            discretise!(eqn, phi, config)
            update_equation!(eqn, config)
            apply_boundary_conditions!(eqn, BCs, XDir(), 0.0, config)
            b = eqn.equation.bx
        else
            phi = ScalarField(mesh)
            phi.values .= 0.3 .+ 0.001 .* (1:ncells)
            eqn = (Divergence{scheme}(mdotf, phi, VectorField(mesh)) == Source(ScalarField(mesh))
                ) → ScalarEquation(phi, BCs)
            discretise!(eqn, phi, config)
            apply_boundary_conditions!(eqn, BCs, nothing, 0.0, config)
            b = eqn.equation.b
        end
        copy(get_sparse_fields(eqn.equation.A)[1]), copy(b)
    end
    for (kind, groups) ∈ pairs(sets), bcs ∈ groups, scheme ∈ (LinearUpwind, LimitedLinearUpwind)
        @test assemble(scheme, bcs, kind == :vector) == assemble(Upwind, bcs, kind == :vector)
    end
end

@testset "Mesh float and index types are preserved" begin
    for (TF, TI) ∈ ((Float32, Int32), (Float64, Int64))
        m = adapt(backend, UNV2D_mesh(
            joinpath(grids_dir, "trig40.unv"), float_type=TF, integer_type=TI))
        fl = FaceVectorField(m); fl.x.values .= TF(1); fl.y.values .= TF(0.6)
        flux = FaceScalarField(m)
        flux!(flux, fl, config)
        phi = ScalarField(m)
        phi.values .= [TF(2) + TF(1e-3)*c[1] for c ∈ m.cells.centre]
        g = Grad{Gauss}(phi)
        grad!(g, FaceScalarField(m), phi, assign(region=m, (phi = [
            Dirichlet(:inlet, 1.0), Dirichlet(:bottom, 0.0), Zerogradient(:outlet),
            Zerogradient(:top)],)).phi, 0.0, config)
        for scheme ∈ (LinearUpwind, LimitedLinearUpwind)
            eqn = (Divergence{scheme}(flux, phi, g.result) == Source(ScalarField(m))
                ) → ScalarEquation(phi, ())
            discretise!(eqn, phi, config)
            @test eltype(eqn.equation.b) == TF
            @test eltype(get_sparse_fields(eqn.equation.A)[1]) == TF
            @test all(isfinite, eqn.equation.b)
        end
    end
end

@testset "Operators without a second-order scheme carry no face source" begin
    mesh = quad
    phi = ScalarField(mesh)
    for scheme ∈ (Upwind, Linear, LUST, BoundedUpwind)
        term = Divergence{scheme}(uniform_flux(mesh), phi)
        @test !XCALibre.Discretise.has_face_source(typeof(term))
        @test XCALibre.Discretise._face_source_init((term,), 0.0) === nothing
    end
    term = Divergence{LinearUpwind}(uniform_flux(mesh), phi, VectorField(mesh))
    @test XCALibre.Discretise.has_face_source(typeof(term))
end

# The solvers pass the gradients: short runs through every cascade must stay finite
@testset "Solvers pass the gradient to second-order schemes" begin
    mesh_file = joinpath(grids_dir, "backwardFacingStep_10mm.unv")
    bfs_mesh() = adapt(backend, UNV2D_mesh(mesh_file, scale=0.001))
    solver(relax; limit=nothing) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=relax, rtol=1e-2, limit=limit)
    hw = Hardware(backend=backend, workgroup=1024)

    @testset "simple! with k-omega transport" begin
        mesh = bfs_mesh()
        model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3),
            turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh)
        BCs = assign(region=mesh, (
            U = [Dirichlet(:inlet, [1.5, 0.0, 0.0]), Extrapolated(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
            p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Symmetry(:top)],
            k = [Dirichlet(:inlet, 0.01), Extrapolated(:outlet), KWallFunction(:wall), Symmetry(:top)],
            omega = [Dirichlet(:inlet, 10.0), Extrapolated(:outlet), OmegaWallFunction(:wall),
                Symmetry(:top)],
            nut = [Dirichlet(:inlet, 1e-3), Extrapolated(:outlet), NutWallFunction(:wall),
                Symmetry(:top)],
        ))
        sch(S) = Schemes(divergence=S, limiter=CellBased())
        schemes = (U=sch(LinearUpwind), p=Schemes(), k=sch(LimitedLinearUpwind),
            omega=sch(LinearUpwind))
        solvers = (U=solver(0.7), p=solver(0.3), k=solver(0.6), omega=solver(0.6))
        config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs, hardware=hw,
            runtime=Runtime(iterations=60, write_interval=-1, time_step=1))
        initialise!(model.momentum.U, [1.5, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
        initialise!(model.turbulence.k, 0.01); initialise!(model.turbulence.omega, 10.0)
        initialise!(model.turbulence.nut, 1e-3)
        res = run!(model, config, progress=false)
        @test all(isfinite, model.momentum.U.x.values)
        @test all(isfinite, model.turbulence.k.values)
        @test res.Ux[end] < res.Ux[1]
    end

    @testset "piso! transient" begin
        mesh = bfs_mesh()
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
        BCs = assign(region=mesh, (
            U = [Dirichlet(:inlet, [1.5, 0.0, 0.0]), Extrapolated(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
            p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Symmetry(:top)],
        ))
        schemes = (U=Schemes(time=Euler, divergence=LinearUpwind, limiter=CellBased()),
            p=Schemes(time=Euler))
        solvers = (U=solver(1.0), p=solver(1.0))
        config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs, hardware=hw,
            runtime=Runtime(iterations=20, write_interval=-1, time_step=0.005))
        initialise!(model.momentum.U, [1.5, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
        run!(model, config, progress=false)
        @test all(isfinite, model.momentum.U.x.values)
    end

    @testset "csimple! with energy" begin
        mesh = adapt(backend, UNV2D_mesh(
            joinpath(grids_dir, "compression_corner_2d_32_64_SF4.unv"), scale=0.001))
        gamma, cp = 1.4, 1005.0
        R = cp*(1 - 1/gamma)
        T_inf, p_inf = 1000.0, 100000.0
        U_inf = 2*sqrt(gamma*R*T_inf)
        model = Physics(time=Steady(),
            fluid=Fluid{Compressible}(nu=0.0, cp=cp, gamma=gamma, Pr=0.7),
            turbulence=RANS{Laminar}(), energy=Energy{SensibleEnthalpy}(Tref=0.0), domain=mesh)
        BCs = assign(region=mesh, (
            U = [Dirichlet(:inlet, [U_inf, 0.0, 0.0]), Zerogradient(:outlet), Zerogradient(:top),
                Slip(:wall)],
            p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:top),
                Zerogradient(:wall)],
            he = [FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=0.0)), Zerogradient(:outlet),
                Zerogradient(:top), Zerogradient(:wall)],
        ))
        sch(S) = Schemes(divergence=S, limiter=CellBased())
        schemes = (U=sch(LinearUpwind), p=Schemes(divergence=Upwind), he=sch(LimitedLinearUpwind))
        solvers = (U=solver(0.7), p=solver(0.3; limit=(0.5*p_inf, 5*p_inf)),
            he=solver(0.3; limit=(800.0, 3000.0)))
        config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs, hardware=hw,
            runtime=Runtime(iterations=300, write_interval=-1, time_step=1))
        initialise!(model.momentum.U, [U_inf, 0.0, 0.0]); initialise!(model.momentum.p, p_inf)
        initialise!(model.energy.T, T_inf)
        run!(model, config, progress=false, consistent=true)
        T = model.energy.T.values
        @test all(isfinite, T)
        @test maximum(T) > 1.2*T_inf # shock formed
        @test minimum(T) > 0.99*T_inf
    end
    @testset "cpiso! transient with energy" begin
        mesh = adapt(backend, UNV2D_mesh(
            joinpath(grids_dir, "compression_corner_2d_32_64_SF4.unv"), scale=0.001))
        gamma, cp = 1.4, 1005.0
        R = cp*(1 - 1/gamma)
        T_inf, p_inf = 1000.0, 100000.0
        U_inf = 2*sqrt(gamma*R*T_inf)
        model = Physics(time=Transient(),
            fluid=Fluid{Compressible}(nu=0.0, cp=cp, gamma=gamma, Pr=0.7),
            turbulence=RANS{Laminar}(), energy=Energy{SensibleEnthalpy}(Tref=0.0), domain=mesh)
        BCs = assign(region=mesh, (
            U = [Dirichlet(:inlet, [U_inf, 0.0, 0.0]), Zerogradient(:outlet), Zerogradient(:top),
                Slip(:wall)],
            p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:top),
                Zerogradient(:wall)],
            he = [FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=0.0)), Zerogradient(:outlet),
                Zerogradient(:top), Zerogradient(:wall)],
        ))
        sch(S) = Schemes(time=Euler, divergence=S, limiter=CellBased())
        schemes = (U=sch(LinearUpwind), p=Schemes(time=Euler, divergence=Upwind),
            he=sch(LimitedLinearUpwind))
        solvers = (U=solver(1.0), p=solver(1.0; limit=(0.2*p_inf, 10*p_inf)),
            he=solver(1.0; limit=(200.0, 4000.0)))
        config = Configuration(solvers=solvers, schemes=schemes, boundaries=BCs, hardware=hw,
            runtime=Runtime(iterations=30, write_interval=-1, time_step=1e-7))
        initialise!(model.momentum.U, [U_inf, 0.0, 0.0]); initialise!(model.momentum.p, p_inf)
        initialise!(model.energy.T, T_inf)
        run!(model, config, progress=false)
        @test all(isfinite, model.energy.T.values)
    end
end
