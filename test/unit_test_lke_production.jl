using XCALibre
using LinearAlgebra
using Test

# The KOmegaLKE production is the double contraction gradU && dev(twoSymm(gradU)). A velocity field with
# both dU/dx and dU/dy non-zero separates it from the matrix product sum((2g)*dev(symm(g))), which only
# agrees in pure shear or pure strain. Pk is read back from the omega source, built before omega is solved.

@testset "KOmegaLKE production is a double contraction" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"),
                               "flatplate_2D_lowRe.unv"), scale=0.001)
    backend  = CPU()
    hardware = Hardware(backend=backend, workgroup=length(mesh.cells) ÷ Threads.nthreads())

    nu = 1.48e-5
    U0 = [5.4, 0.0, 0.0]
    k0 = 0.0575
    kl0 = 0.0115
    ω0 = 275.0
    a, b = 1.0, 3.0 # U = (5.4 + a x + b y, -a y, 0): dU/dx = a, dU/dy = b, divergence free

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

    # Pω = Cω1*Pk - (2/3)*Cω1*divU*ω with ω = ω0 when the source is built
    Cω1 = model.turbulence.coeffs.Cω1
    Pω = get_source(rans.ω_eqn, 1).values
    Pk = Pω ./ Cω1 .+ (2/3) .* rans.divU.values .* ω0

    function production(g, product)
        S_dev = 0.5*(g + g') - tr(g)/3*I
        return product ? sum((2*g)*S_dev) : 2*sum(g .* S_dev)
    end
    Pk_contraction = [production(gradU[i], false) for i ∈ eachindex(mesh.cells)]
    Pk_product = [production(gradU[i], true) for i ∈ eachindex(mesh.cells)]

    # Non-vacuous: the two forms differ clearly in most cells of this field ...
    @test count(abs.(Pk_product .- Pk_contraction) .> 0.1 .* abs.(Pk_contraction)) > length(Pk) ÷ 2
    # ... and the model uses the contraction. Before the fix it matched Pk_product.
    @test Pk ≈ Pk_contraction rtol=1e-10
end
