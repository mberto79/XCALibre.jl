using XCALibre
using Test

# =============================================================================
#  Turbulent dispersion must not reach the momentum equation
# =============================================================================
#
#  The drift-flux momentum stress is
#
#      div( alpha*(1-alpha)*(rho1 + rho2)/rho_m * Ur (x) Ur )
#
#  and it is derived from the MEAN slip between the phases. Turbulent dispersion
#  is a different object: it closes the fluctuation correlation <alpha' u'>,
#  i.e. a diffusive flux, and is written as a drift velocity
#
#      Ur_disp = D_t * grad(alpha) / (alpha*(1 - alpha))
#
#  purely so it can ride the same transport machinery. Folding it into `Ur`
#  before the momentum stress is formed therefore squares a diffusive closure.
#
#  What that produces, substituting into the stress:
#
#      D_t^2 * |grad(alpha)|^2 / (alpha*(1 - alpha))
#
#  QUADRATIC in the volume-fraction gradient, so a cell-to-cell oscillation -
#  the field that maximises that gradient for a given amplitude - is its
#  preferred mode; and divided by alpha*(1 - alpha), which is ~0.002 in a nearly
#  pure liquid, so it is amplified ~500x exactly where the flow is least
#  two-phase.
#
#  These tests pin the ALGEBRA of that claim. The solver-level guarantee - that
#  `div_slip_outer!` is handed the pre-dispersion face velocity - is enforced by
#  `Urf_slip` in `Solvers_5_Multiphase.jl`.
# =============================================================================

@testset "Turbulent dispersion and the momentum slip stress" begin

    # The dispersion drift velocity, as `_turbulent_dispersion!` forms it.
    ur_disp(D_t, grad_alpha, alpha) =
        D_t*grad_alpha/(max(alpha, 1e-3)*max(1 - alpha, 1e-3))

    # The momentum stress coefficient, as `_slip_coeff` forms it.
    slip_coeff(alpha, rho1, rho2) =
        alpha*(1 - alpha)*(rho1 + rho2)/((1 - alpha)*rho2 + alpha*rho1)

    # Saturated LH2/GH2 at 0.4 MPa, and the LH2 pipe's near-wall cell.
    RHO_L, RHO_V = 62.95, 4.84
    H_CELL = 4.292e-5          # wall cell height [m]
    NUT    = 2.0e-6            # near-wall eddy viscosity [m^2/s]
    SC_T   = 0.7
    D_T    = NUT/SC_T

    @testset "dispersion velocity blows up as alpha approaches 1" begin
        g = 0.01/H_CELL                      # a 1% oscillation across one cell
        # It is the 1/(alpha(1-alpha)) factor that does it, not D_t or the
        # gradient - both of which are held fixed here.
        u_mid  = ur_disp(D_T, g, 0.5)
        u_bulk = ur_disp(D_T, g, 0.998)
        @test u_bulk/u_mid > 100
        @test isapprox(u_bulk/u_mid, 0.25/(0.998*0.002), rtol = 1e-6)
    end

    @testset "a 1% oscillation out-runs the physical buoyant slip" begin
        # Stokes terminal velocity of a 0.5 mm bubble - the scale the drift flux
        # is actually meant to carry.
        d_b, mu_l = 0.5e-3, 1.2e-5
        u_stokes = 9.81*abs(RHO_L - RHO_V)*d_b^2/(18*mu_l)

        u_noise = ur_disp(D_T, 0.01/H_CELL, 0.998)
        @test u_noise > 0.1                      # order 0.3 m/s
        # The point: numerical noise generating more slip than gravity does.
        @test u_noise > 0.1*u_stokes
    end

    @testset "the stress is quadratic in the gradient, not linear" begin
        # This is why a checkerboard is the preferred mode: doubling the
        # oscillation amplitude QUADRUPLES the momentum stress it produces.
        alpha = 0.998
        c = slip_coeff(alpha, RHO_L, RHO_V)
        stress(g) = c*ur_disp(D_T, g, alpha)^2

        g1 = 0.01/H_CELL
        @test stress(2*g1)/stress(g1) ≈ 4.0 rtol = 1e-9
        @test stress(4*g1)/stress(g1) ≈ 16.0 rtol = 1e-9

        # And the closed form the docstring quotes: the alpha(1-alpha) in the
        # coefficient cancels ONE of the two in the squared velocity, leaving a
        # single inverse power.
        expected = D_T^2*g1^2*(RHO_L + RHO_V) /
                   (alpha*(1 - alpha)*((1 - alpha)*RHO_V + alpha*RHO_L))
        @test stress(g1) ≈ expected rtol = 1e-9
    end

    @testset "the force-balance slip has neither pathology" begin
        # A genuine mean slip is bounded and carries no gradient at all, so it
        # is indifferent to an oscillation in alpha. That is the property the
        # momentum equation needs and the reason `Urf_slip` exists.
        u_slip = 0.05                            # any force-balance value
        c(alpha) = slip_coeff(alpha, RHO_L, RHO_V)
        stress(alpha) = c(alpha)*u_slip^2

        # Bounded across the whole range, and vanishing in a single-phase cell.
        # (The coefficient peaks near alpha ~ 0.25 rather than 0.5, because
        # rho_m in the denominator is 13x larger on the liquid side.)
        @test stress(1.0) == 0.0
        @test stress(0.0) == 0.0
        peak = maximum(stress(a) for a in range(0.0, 1.0, length = 2001))
        @test isfinite(peak)
        for a in range(0.0, 1.0, length = 51)
            @test 0.0 <= stress(a) <= peak + 1e-12
        end

        # THE CONTRAST. As alpha -> 1 the force-balance stress goes to ZERO,
        # while the dispersion form diverges: the alpha*(1-alpha) in the
        # coefficient cancels only one of the two inverse powers in the squared
        # velocity, leaving 1/(alpha*(1-alpha)).
        for a in (0.9, 0.99, 0.999)
            @test stress(a) < stress(0.5)
        end
        d(a) = c(a)*ur_disp(D_T, 0.01/H_CELL, a)^2
        @test d(0.999) > d(0.99) > d(0.9)

        # At alpha = 0.998 the two closures differ by orders of magnitude for
        # the same 1% oscillation - this is the size of what was being fed into
        # the momentum equation.
        c998 = c(0.998)
        disp_stress = c998*ur_disp(D_T, 0.01/H_CELL, 0.998)^2
        @test disp_stress/stress(0.998) > 10
    end
end
