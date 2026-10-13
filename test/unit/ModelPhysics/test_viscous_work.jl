using XCALibre
using LinearAlgebra
using StaticArrays
using Test

# Viscous work ∇·(τ·U) of the sensible-enthalpy equation, as Σ_f U_f·F_f with the momentum viscous face force.
# For the shear U = (γ y, 0, 0) its volume integral is the boundary work (zero between walls at rest, μγ U_w A_wall
# with a moving wall), whereas the dissipation τ:∇U = μγ² integrates to μγ²V.

const VW = XCALibre.ModelPhysics

vw_grids = pkgdir(XCALibre, "examples/0_GRIDS")

function vw_case(TF, γ, μ; top=nothing, slip=false, updown=1)
    mesh = UNV2D_mesh(joinpath(vw_grids, "quad40.unv"), scale=0.001, float_type=TF)
    names = [b.name for b ∈ mesh.boundaries]
    # the two wall patches: find which one lies at the largest y
    ymean(b) = sum(mesh.faces[f].centre[2] for f ∈ b.IDs_range)/length(b.IDs_range)
    itop = ymean(mesh.boundaries[3]) > ymean(mesh.boundaries[4]) ? 3 : 4
    ibot = 7 - itop
    # a moving top wall moves with the profile, U_w = γ y_top, so the wall gradient is γ as well
    Uw = isnothing(top) ? zero(TF) : TF(γ*ymean(mesh.boundaries[itop]))
    Utop = [Uw, 0.0, 0.0]
    walls = slip ? [Slip(names[ibot]), Slip(names[itop])] :
        [Wall(names[ibot], [0.0, 0.0, 0.0]), Wall(names[itop], Utop)]
    BCs = assign(region=mesh, (U = [Zerogradient(names[1]), Zerogradient(names[2]), walls...],))
    config = Configuration(
        solvers=(U=SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
        schemes=(U=Schemes(),), runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    U = VectorField(mesh)
    gradU = Grad{Gauss}(U)
    G = SMatrix{3,3,TF}(0, 0, 0, γ, 0, 0, 0, 0, 0) # ∂U_x/∂y = γ
    for i ∈ eachindex(mesh.cells)
        U.x.values[i] = γ*mesh.cells[i].centre[2]
        gradU.result[i] = G
    end
    Uf = FaceVectorField(mesh)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, config.boundaries.U, zero(TF), config)
    mueff = FaceScalarField(mesh); mueff.values .= μ
    mueff_cell = ScalarField(mesh); mueff_cell.values .= μ

    # mass flux across the shear (sign of n_y), so the upwind face velocity is that of the lower (updown=1)
    # or upper (updown=-1) cell; faces normal to x carry no flux and no work in this profile
    mdotf = FaceScalarField(mesh)
    for f ∈ eachindex(mesh.faces); mdotf.values[f] = updown*sign(mesh.faces[f].normal[2]); end

    work = ScalarField(mesh)
    VW.viscous_work!(work, mueff, mueff_cell, U, Uf, gradU, mdotf, config.boundaries.U, config)
    dissipation = ScalarField(mesh)
    VW.viscous_dissipation!(dissipation, mueff_cell.values, gradU, config.boundaries.U, config)

    V = [c.volume for c ∈ mesh.cells]
    A_top = sum(mesh.faces[f].area for f ∈ mesh.boundaries[itop].IDs_range)
    (; mesh, work, dissipation, V, A_top, Uw)
end

@testset "Viscous work ∇·(τ·U) of the enthalpy equation" begin
    for TF ∈ (Float64, Float32)
        γ, μ = TF(50), TF(1.8e-5)
        tol = TF === Float64 ? 1e-10 : 1e-4
        rtol = TF === Float64 ? 1e-8 : 1e-3

        # walls at rest: no work enters the domain, although τ:∇U > 0 everywhere
        c = vw_case(TF, γ, μ)
        @test eltype(c.work.values) === TF
        Φ = μ*γ^2*sum(c.V)
        @test sum(c.dissipation.values .* c.V) ≈ Φ rtol=rtol
        @test abs(sum(c.work.values .* c.V)) < tol*Φ

        # moving top wall: the work equals the shear work of that wall
        c = vw_case(TF, γ, μ; top=true)
        @test sum(c.work.values .* c.V) ≈ μ*γ*c.Uw*c.A_top rtol=rtol

        # interior cells: ∇·(τ·U) = τ:∇U + U·∇·τ = μγ² for this linear profile, for either upwind direction
        for updown ∈ (1, -1)
            c = vw_case(TF, γ, μ; top=true, updown)
            interior = setdiff(eachindex(c.mesh.cells), c.mesh.boundary_cellsID)
            @test maximum(abs.(c.work.values[interior] .- μ*γ^2)) < 1e-3*μ*γ^2
        end

        # slip walls do no shear work
        c = vw_case(TF, γ, μ; slip=true)
        @test abs(sum(c.work.values .* c.V)) < tol*Φ
    end
end
