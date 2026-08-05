# Peng-Robinson cubic equation of state.
#
# The tests that matter here are the ones with an answer known independently of
# the implementation:
#
#   * the ideal-gas limit - at low pressure and high temperature ANY cubic must
#     give Z -> 1, and this pins R, a, b and the alpha function together. A sign
#     error or a molar/mass mix-up cannot survive it.
#   * the cubic must actually satisfy p(v,T) - substituting the root back into
#     the Peng-Robinson equation must return the pressure it was solved for. That
#     checks Cardano independently of any thermodynamics.
#   * psi and beta are finite differences of the density, so for a nearly-ideal
#     state they must approach 1/p and 1/T.
#
# What is NOT asserted is agreement with the Helmholtz EOS. Peng-Robinson is a
# generic cubic fitted to hydrocarbons and hydrogen has a negative acentric
# factor; the liquid density is ~9% out at the LH2 operating point. That is a
# known property of the model, not a defect, and pinning it would be asserting
# the wrong thing.

using XCALibre
using Test

const PR_TSAT = 29.1548      # T_sat(0.7 MPa) from the Helmholtz saturation curve
const PR_PSAT = 0.7e6

@testset "PengRobinson: construction" begin
    pv = PengRobinson(H2(), branch=:vapour)
    @test pv isa PengRobinson
    @test pv.Tc ≈ 33.145
    @test pv.R ≈ XCALibre.ModelPhysics.R_UNIVERSAL/2.01588e-3 rtol=1e-3
    @test pv.b > 0
    @test pv.a > 0

    # isbits is what makes it usable in a kernel at all - the whole point of the
    # analytic path over a table.
    @test isbits(pv)

    # Explicit critical properties must agree with the fluid-derived form.
    pe = PengRobinson(Tc=33.145, pc=1.2964e6, omega=-0.219, M=2.01588e-3, branch=:vapour)
    @test pe.a ≈ pv.a rtol=1e-6
    @test pe.b ≈ pv.b rtol=1e-6
    @test pe.kappa ≈ pv.kappa rtol=1e-6

    @test_throws ArgumentError PengRobinson(Tc=33.1, pc=1.3e6, omega=0.0, branch=:solid, M=2e-3)
    @test_throws ArgumentError PengRobinson(Tc=33.1, pc=1.3e6, omega=0.0)              # neither M nor R
    @test_throws ArgumentError PengRobinson(Tc=33.1, pc=1.3e6, omega=0.0, M=2e-3, R=4000.0)
end

@testset "PengRobinson: ideal-gas limit" begin
    # Low pressure, high temperature: every cubic must collapse to pv = RT.
    pv = PengRobinson(H2(), branch=:vapour)
    for (p, T) in ((1.0e3, 300.0), (1.0e2, 500.0), (1.0e4, 400.0))
        Z, _ = pr_compressibility_factor(pv, p, T)
        @test Z ≈ 1.0 rtol=1e-4
        @test pr_density(pv, p, T) ≈ p/(pv.R*T) rtol=1e-4
    end

    # psi -> 1/p and beta -> 1/T in the same limit. These come from finite
    # differences of the density, so they also confirm the derivative steps.
    p, T = 1.0e3, 300.0
    @test phase_compressibility(pv, p, T) ≈ 1/p rtol=1e-3
    @test XCALibre.ModelPhysics.pr_expansivity(pv, p, T) ≈ 1/T rtol=1e-3
end

@testset "PengRobinson: the root satisfies the cubic" begin
    # Substitute the root back into p = RT/(v-b) - a*alpha/(v^2 + 2bv - b^2).
    # Independent of any thermodynamic expectation - it tests Cardano alone.
    for branch in (:liquid, :vapour), (p, T) in
            ((PR_PSAT, PR_TSAT), (0.3e6, 25.0), (1.1e6, 31.0), (0.5e6, 40.0))
        eos = PengRobinson(H2(), branch=branch)
        (; R, Tc, a, b, kappa) = eos
        Z, _ = pr_compressibility_factor(eos, p, T)
        v = Z*R*T/p
        alpha = (1 + kappa*(1 - sqrt(T/Tc)))^2
        p_back = R*T/(v - b) - a*alpha/(v^2 + 2b*v - b^2)
        @test p_back ≈ p rtol=1e-8
        @test v > b                     # positive free volume
        @test Z > 0
    end
end

@testset "PengRobinson: branch selection" begin
    pl = PengRobinson(H2(), branch=:liquid)
    pv = PengRobinson(H2(), branch=:vapour)

    # Where both branches are distinct roots the liquid must be the denser one.
    @test pr_branch_exists(pl, PR_PSAT, PR_TSAT)
    @test pr_density(pl, PR_PSAT, PR_TSAT) > pr_density(pv, PR_PSAT, PR_TSAT)

    # Above Tc there is no liquid: one root, and both branches return it.
    T_super = 40.0
    @test !pr_branch_exists(pl, PR_PSAT, T_super)
    @test pr_density(pl, PR_PSAT, T_super) ≈ pr_density(pv, PR_PSAT, T_super) rtol=1e-12
end

@testset "PengRobinson: liquid branch is smooth over its own range" begin
    # This is the property the tabulated path could not deliver, and the reason
    # this EOS exists here: within the region where the branch is real, rho must
    # fall smoothly and monotonically with T, with no fallback steps.
    pl = PengRobinson(H2(), branch=:liquid)
    Ts = range(19.0, 29.0, length=81)
    for p in (0.3e6, 0.7e6, 1.1e6)
        rhos = [pr_density(pl, p, T) for T in Ts]
        @test all(diff(rhos) .< 0)                          # strictly decreasing
        ratios = [max(rhos[i]/rhos[i+1], rhos[i+1]/rhos[i]) for i in 1:length(rhos)-1]
        @test maximum(ratios) < 1.05                        # no cliff
        @test all(r -> 20.0 < r < 120.0, rhos)              # physically sane
    end
end

@testset "PengRobinson: psi and beta are positive and physical" begin
    pl = PengRobinson(H2(), branch=:liquid)
    pv = PengRobinson(H2(), branch=:vapour)

    # A fluid compresses under pressure and expands when heated, so both must be
    # positive wherever the branch is real.
    for T in 20.0:2.0:28.0
        @test phase_compressibility(pl, PR_PSAT, T) > 0
        @test XCALibre.ModelPhysics.pr_expansivity(pl, PR_PSAT, T) > 0
    end
    # The vapour is far more compressible than the liquid.
    @test phase_compressibility(pv, PR_PSAT, 31.0) > phase_compressibility(pl, PR_PSAT, 25.0)

    @test XCALibre.ModelPhysics.phase_betaT(pl, 0.05, 300.0) ≈ 15.0
    @test XCALibre.ModelPhysics.specific_gas_constant(pl) ≈ pl.R
end

@testset "PengRobinson: envelope report" begin
    pl = PengRobinson(H2(), branch=:liquid)
    r = pr_table_report(pl, p=(0.3e6, 1.1e6), T=(19.0, 29.0), np=9, nT=21)
    @test r.worst_ratio < 1.1               # smooth over its valid range
    @test r.rho_min > 0
    @test r.rho_max > r.rho_min
end

@testset "PengRobinson: fills a Phase and updates per cell" begin
    # The solver-facing path: build a Phase from it and confirm the density and
    # expansivity fields actually get filled from the cubic.
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)
    backend = CPU()
    hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)
    config = Configuration(solvers=(;), schemes=(;), runtime=(;),
                           hardware=hardware, boundaries=(;))

    pl = PengRobinson(H2(), branch=:liquid)
    rho_f = ScalarField(mesh_dev)
    beta_f = ScalarField(mesh_dev)
    p_abs = ScalarField(mesh_dev); initialise!(p_abs, PR_PSAT)
    T = ScalarField(mesh_dev);     initialise!(T, 25.0)

    XCALibre.ModelPhysics.update_phase_property!(rho_f, pl, p_abs, T, config)
    XCALibre.ModelPhysics.update_phase_property!(beta_f, PengRobinsonBeta(pl), p_abs, T, config)

    @test all(isfinite, rho_f.values)
    @test all(≈(pr_density(pl, PR_PSAT, 25.0)), rho_f.values)
    @test all(>(0), beta_f.values)
    @test all(≈(XCALibre.ModelPhysics.pr_expansivity(pl, PR_PSAT, 25.0)), beta_f.values)
end
