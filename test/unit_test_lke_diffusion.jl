using XCALibre
using LinearAlgebra
using Test

# KOmegaLKE diffusivities: nu + sigma*gamma*k/omega for k and omega, and nu + sigmakL*sqrt(kl)*y for kl.
# Boundary faces take the boundary values of k, omega and kl, so a no-slip wall (k = kl = 0) gets nu.
# One call to turbulence!, before which k = k0 and omega = omega0 everywhere.

@testset "KOmegaLKE diffusivities" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"),
                               "flatplate_2D_lowRe.unv"), scale=0.001)
    backend  = CPU()
    hardware = Hardware(backend=backend, workgroup=length(mesh.cells) ÷ Threads.nthreads())

    nu = 1.48e-5
    U0 = [5.4, 0.0, 0.0]
    k0 = 0.0575
    kl0 = 0.0115
    ω0 = 275.0
    a, b = 1.0, 30.0 # U = (5.4 + a x + b y, -a y, 0): shear sets gamma < 1

    model = Physics(
        time       = Steady(),
        fluid      = Fluid{Incompressible}(nu=nu),
        turbulence = RANS{KOmegaLKE}(Tu=0.01, walls=(:wall,)),
        energy     = Energy{Isothermal}(),
        domain     = mesh,
    )

    BCs = assign(region=mesh, (
        U     = [Dirichlet(:inlet, U0), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Extrapolated(:top)],
        p     = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Extrapolated(:top)],
        k     = [Dirichlet(:inlet, k0), Zerogradient(:outlet), Dirichlet(:wall, 0.0), Extrapolated(:top)],
        kl    = [Dirichlet(:inlet, kl0), Zerogradient(:outlet), Dirichlet(:wall, 0.0), Extrapolated(:top)],
        omega = [Dirichlet(:inlet, ω0), Zerogradient(:outlet), OmegaWallFunction(:wall), Extrapolated(:top)],
        nut   = [Dirichlet(:inlet, k0/ω0), Extrapolated(:outlet), Dirichlet(:wall, 0.0), Extrapolated(:top)],
    ))

    schemes = (
        U = Schemes(divergence=LUST), p = Schemes(divergence=LUST), k = Schemes(divergence=LUST),
        y = Schemes(gradient=Gauss), kl = Schemes(divergence=LUST), omega = Schemes(divergence=LUST),
    )
    setup(relax) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=relax, rtol=1e-2)
    solvers = (
        U = setup(0.7), p = setup(0.2), kl = setup(0.3), k = setup(0.3), omega = setup(0.3),
        y = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, rtol=1e-3, relax=0.9),
    )
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=hardware, boundaries=BCs)

    (; U, p, Uf) = model.momentum
    for i ∈ eachindex(mesh.cells)
        x, y, _ = mesh.cells[i].centre
        U.x.values[i] = U0[1] + a*x + b*y
        U.y.values[i] = -a*y
        U.z.values[i] = 0.0
    end
    initialise!(p, 0.0)
    initialise!(model.turbulence.k, k0)
    initialise!(model.turbulence.kl, kl0)
    initialise!(model.turbulence.omega, ω0)
    initialise!(model.turbulence.nut, k0/ω0)

    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    p_eqn = (-Laplacian{Linear}(rDf, p) == -Source(ScalarField(mesh))) → ScalarEquation(p, BCs.p)
    rans, config = XCALibre.ModelPhysics.initialise(model.turbulence, model, mdotf, p_eqn, config)

    gradU = Grad{Gauss}(U)
    S = StrainRate(gradU, XCALibre.Calculate.T(gradU), U, Uf)
    prev = zeros(length(mesh.cells))
    XCALibre.ModelPhysics.turbulence!(rans, model, S, prev, 1, config)

    (; σk, σω, σkL) = model.turbulence.coeffs
    γ = rans.γ.values
    ω = model.turbulence.omega.values # omega after its solve, used by the k diffusivity
    y = model.turbulence.y.values

    # Cell diffusivities carry gamma (non-vacuous: gamma is clearly below 1 in part of the domain)
    @test count(γ .< 0.99) > length(γ) ÷ 50
    @test rans.nueffωS.values ≈ nu .+ σω .* γ .* k0 ./ ω0 rtol=1e-12
    @test rans.nueffkS.values ≈ nu .+ σk .* γ .* k0 ./ max.(ω, 1e-15) rtol=1e-12

    nueffkL = get_flux(rans.kl_eqn, 3).values
    nueffk  = get_flux(rans.k_eqn, 3).values
    nueffω  = get_flux(rans.ω_eqn, 3).values
    patch(name) = mesh.boundaries[boundary_index(mesh.boundaries, name)].IDs_range
    owner = mesh.boundary_cellsID

    # Wall faces: k = kl = 0, so each diffusivity is nu (it was 0, taken from the eddy viscosity)
    wall = patch(:wall)
    @test all(nueffkL[wall] .≈ nu)
    @test all(nueffk[wall] .≈ nu)
    @test all(nueffω[wall] .≈ nu)

    # Inlet faces: evaluated from the inlet values of k, omega and kl
    inlet = patch(:inlet)
    cID = owner[inlet]
    @test nueffk[inlet] ≈ nu .+ σk .* γ[cID] .* k0 ./ ω0 rtol=1e-12
    @test nueffω[inlet] ≈ nu .+ σω .* γ[cID] .* k0 ./ ω0 rtol=1e-12
    @test nueffkL[inlet] ≈ nu .+ σkL .* sqrt(kl0) .* y[cID] rtol=1e-12
end
