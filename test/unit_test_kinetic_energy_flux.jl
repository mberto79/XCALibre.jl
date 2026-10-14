using XCALibre
using Random
using Test
using StaticArrays
using LinearAlgebra

# Kinetic-energy flux of the sensible-enthalpy equation. The explicit ∇·(ṁ K) term takes the face
# interpolation of the enthalpy convection scheme, so h + K is convected by one discrete operator.

ke_grids = pkgdir(XCALibre, "examples/0_GRIDS")

@testset "Face value of a divergence scheme" begin
    for TF ∈ (Float64, Float32)
        mesh = UNV2D_mesh(joinpath(ke_grids, "quad40.unv"), scale=0.001, float_type=TF)
        (; cells, faces) = mesh
        config = Configuration(solvers=(;), schemes=(;), boundaries=(;),
            runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
            hardware=Hardware(backend=CPU(), workgroup=64))
        rng = Xoshiro(1)
        phi = ScalarField(mesh); phi.values .= rand(rng, TF, length(cells))
        mdotf = FaceScalarField(mesh); mdotf.values .= randn(rng, TF, length(faces))
        nbfaces = length(mesh.boundary_cellsID)
        internal = nbfaces+1:length(faces)
        upwind(f) = mdotf.values[f] >= 0 ? phi.values[faces[f].ownerCells[1]] : phi.values[faces[f].ownerCells[2]]
        linear(f) = faces[f].weight*phi.values[faces[f].ownerCells[1]] +
            (1 - faces[f].weight)*phi.values[faces[f].ownerCells[2]]
        expected = Dict(Upwind => upwind, BoundedUpwind => upwind, Linear => linear,
            LUST => f -> TF(0.75)*linear(f) + TF(0.25)*upwind(f))
        for S ∈ (Upwind, BoundedUpwind, Linear, LUST)
            phif = FaceScalarField(mesh); phif.values .= TF(-1)
            interpolate!(phif, phi, mdotf, Divergence{S}(), config)
            @test eltype(phif.values) === TF
            @test all(==(TF(-1)), phif.values[1:nbfaces]) # boundary faces untouched
            @test all(f -> phif.values[f] ≈ expected[S](f), internal)
            S === BoundedUpwind && continue
            # flux*phi_f is the owner-side face flux assembled by the implicit operator
            term = Divergence{S}(mdotf, phi)
            @test all(internal) do f
                o, n = faces[f].ownerCells
                ac, an = scheme!(term, nothing, cells, faces, n, 1, 0, 0, f, nothing, nothing)
                mdotf.values[f]*phif.values[f] ≈ ac*phi.values[o] + an*phi.values[n]
            end
        end
        # Upwind reproduces the upwind interpolation exactly
        phif = FaceScalarField(mesh); interpolate!(phif, phi, mdotf, Divergence{Upwind}(), config)
        phiu = FaceScalarField(mesh); interpolate_upwind!(phiu, phi, mdotf, config)
        @test phif.values[internal] == phiu.values[internal]
    end
end

# LinearUpwind and LimitedLinearUpwind reconstruct K from its gradient, as the he scheme does h: on uniform
# quads the face value of a linear K is exact wherever the upwind cell is an interior cell, and with
# h + K uniform the face values of h (as assembled) and K add up to it, with a gradient limiter too
@testset "Second-order kinetic-energy face value" begin
    MP = XCALibre.ModelPhysics
    for TF ∈ (Float64, Float32)
        mesh = UNV2D_mesh(joinpath(ke_grids, "quad40.unv"), scale=0.001, float_type=TF)
        (; cells, faces, boundary_cellsID) = mesh
        BCs = assign(region=mesh, (he=[Zerogradient(b) for b ∈ (:inlet, :outlet, :bottom, :top)],))
        config = Configuration(solvers=(;), schemes=(;), boundaries=BCs,
            runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
            hardware=Hardware(backend=CPU(), workgroup=64))
        a = SVector{3,TF}(3, -7, 0)
        x0 = SVector{3,TF}(0.4, 0.6, 0)
        mdotf = FaceScalarField(mesh); mdotf.values .= randn(Xoshiro(2), TF, length(faces))
        nbfaces = length(boundary_cellsID)
        interior = setdiff(eachindex(cells), boundary_cellsID)
        upwind(f) = mdotf.values[f] > 0 ? faces[f].ownerCells : reverse(faces[f].ownerCells)
        inner = filter(f -> first(upwind(f)) ∈ interior, nbfaces+1:length(faces))
        # h face value of the assembled he term (scheme_face_source!)
        hface(S, h, gh, f) = begin
            u, d = upwind(f)
            δ = gh[u]⋅(faces[f].centre - cells.centre[u])
            S === LinearUpwind ? h[u] + δ :
                h[u] + XCALibre.Discretise._limit_tvd(δ, gh[u]⋅(cells.centre[d] - cells.centre[u]), h[d] - h[u])
        end
        run(S, lim, Kfun) = begin
            K = ScalarField(mesh); K.values .= Kfun.(cells.centre)
            he = ScalarField(mesh); he.values .= TF(5) .- K.values
            Kf = FaceScalarField(mesh); Kf.values .= TF(-1)
            cg = MP.convection_gradient(S, he, Gauss)
            cgK = MP.kinetic_convection_gradient(S, K, Gauss)
            MP.kinetic_face!(Kf, K, cgK, cg, he, BCs.he, lim, mdotf, Divergence{S}(), zero(TF), config)
            Kf, he, cg
        end
        for S ∈ (LinearUpwind, LimitedLinearUpwind, Upwind)
            Kf, _, _ = run(S, nothing, c -> a⋅c + TF(2))
            @test eltype(Kf.values) === TF
            @test all(==(TF(-1)), Kf.values[1:nbfaces]) # boundary faces untouched
            exact = [a⋅faces[f].centre + TF(2) for f ∈ inner]
            @test isapprox(Kf.values[inner], exact; rtol=sqrt(eps(TF))) == (S !== Upwind)
        end
        for S ∈ (LinearUpwind, LimitedLinearUpwind), lim ∈ (nothing, CellBased())
            Kf, he, cg = run(S, lim, c -> TF(2) + TF(100)*sum(abs2, c - x0))
            H = [hface(S, he.values, cg.grad.result, f) + Kf.values[f] for f ∈ inner]
            @test maximum(abs, H .- TF(5)) < 100eps(TF)
        end
    end
end

struct NoFaceValueScheme end

# A scheme without a defined face value must fail loudly, never fall back silently.
@testset "Unsupported scheme for the kinetic-energy flux" begin
    mesh = UNV2D_mesh(joinpath(ke_grids, "quad40.unv"), scale=0.001)
    config = Configuration(solvers=(;), schemes=(;), boundaries=(;),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=64))
    phi = ScalarField(mesh); mdotf = FaceScalarField(mesh); phif = FaceScalarField(mesh)
    err = try; interpolate!(phif, phi, mdotf, Divergence{NoFaceValueScheme}(), config); nothing
        catch e; e; end
    @test err isa ArgumentError
    @test occursin("NoFaceValueScheme", err.msg)
    @test occursin("kinetic-energy flux", err.msg)
end

# Steady inviscid adiabatic flow conserves the total temperature T0 = T + |U|²/(2 cp), also through
# the oblique shock of a Mach 2 compression corner, whatever the enthalpy convection scheme.
@testset "Total temperature through a compression corner" begin
    mesh = UNV2D_mesh(joinpath(ke_grids, "compression_corner_2d_32_64_SF4.unv"), scale=0.001)
    gamma, cp, temp, pressure = 1.4, 1005.0, 1000.0, 1e5
    R = cp*(1 - 1/gamma)
    Umag = 2*sqrt(gamma*R*temp)
    velocity = [Umag, 0.0, 0.0]
    T0_inf = temp + Umag^2/(2cp)
    for (hes, lim) ∈ ((LUST, nothing), (Linear, nothing), (LinearUpwind, nothing), (LimitedLinearUpwind, nothing),
            (LinearUpwind, CellBased()), (LimitedLinearUpwind, CellBased()))
        model = Physics(time=Steady(), fluid=Fluid{Compressible}(nu=0.0, cp=cp, gamma=gamma, Pr=0.7),
            turbulence=RANS{Laminar}(), energy=Energy{SensibleEnthalpy}(Tref=0.0), domain=mesh)
        BCs = assign(region=mesh, (
            U=[Dirichlet(:inlet, velocity), Zerogradient(:outlet), Zerogradient(:top), Slip(:wall)],
            p=[Dirichlet(:inlet, pressure), Zerogradient(:outlet), Zerogradient(:top), Zerogradient(:wall)],
            he=[FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=0.0)), Zerogradient(:outlet),
                Zerogradient(:top), Zerogradient(:wall)]))
        setup(relax; kw...) =
            SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=relax, rtol=0.0, atol=1e-2; kw...)
        solvers = (U=setup(0.7), p=setup(0.3; limit=(0.5pressure, 5pressure)), he=setup(0.3; limit=(800.0, 3000.0)))
        schemes = (U=Schemes(divergence=Upwind), p=Schemes(divergence=Upwind), he=Schemes(divergence=hes, limiter=lim))
        config = Configuration(; solvers, schemes, boundaries=BCs,
            runtime=Runtime(iterations=1000, write_interval=-1, time_step=1),
            hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))
        initialise!(model.momentum.U, velocity)
        initialise!(model.momentum.p, pressure)
        initialise!(model.energy.T, temp)
        residuals = run!(model, config; progress=false)
        (; U) = model.momentum
        T0 = @. model.energy.T.values + (U.x.values^2 + U.y.values^2)/(2cp)
        @test last(residuals.p) < 1e-6
        @test maximum(model.energy.T.values) > 1.25temp # the shock is there
        @test maximum(abs, T0 ./ T0_inf .- 1) < 1e-4 # upwind K: 3e-2 to 5e-2; h and K limited apart: 5e-3 to 7e-3
    end
end
