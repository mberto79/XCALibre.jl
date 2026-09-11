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

# =============================================================================
#  Which boundaries may the phases slip across?
# =============================================================================
#
#  `Urf` is the RELATIVE velocity between the phases, so zeroing it on a face
#  means "no relative phase flux here". That is not the same question as for the
#  mixture flux, and getting it wrong is quiet:
#
#    * zero it at an OUTFLOW and gas arrives in the last cell at U + Ur but
#      leaves at U, so it accumulates until the raised alpha closes the balance;
#    * fail to zero it at a SYMMETRY plane and gas crosses the mirror while its
#      image crosses the other way, i.e. gas appears or vanishes at the boundary.
#
#  MEASURED before the fix, adiabatic air-water pipe, outlet cell layer against
#  its neighbour: alpha 0.1288 -> 0.1708, +32.6%, and identical at iteration 1000
#  and 2500 - stationary, not transient. A steady balance needs
#  alpha_out/alpha_up = 1 + Ur/U, so +32.6% implies Ur = 0.282 m/s, a plausible
#  slip for that case's 2.7 mm bubble. That arithmetic is what identified the
#  mechanism, and it is why the test below is about TYPES rather than values.
# =============================================================================

@testset "Boundary faces the drift velocity is zeroed on" begin
    using XCALibre.Solvers: build_drift_zero_faces

    struct MockMesh{V}; boundary_cellsID::V; end
    noSlip = [0.0, 0.0, 0.0]
    mesh = MockMesh(collect(1:70))

    inlet  = Dirichlet(:inlet, [0.0, 0.0, 0.753], 1:10)
    outlet = Zerogradient(:outlet, 0, 11:20)
    wall   = Wall(:wall, noSlip, 21:30)
    symX   = Symmetry(:symX, 0, 31:40)
    symY   = Symmetry(:symY, 0, 41:50)
    extrap = Extrapolated(:far, 0, 51:60)
    neum   = Neumann(:n, 0.0, 61:70)

    @testset "walls, symmetry and inlets are zeroed" begin
        f = Set(build_drift_zero_faces((inlet, outlet, wall, symX, symY), mesh))
        @test all(in(f), 1:10)      # inlet: specified homogeneously, alpha*U is j_g
        @test all(in(f), 21:30)     # wall: solid, neither phase crosses
        @test all(in(f), 31:50)     # symmetry: mirror, every normal flux vanishes
    end

    @testset "outflows are NOT zeroed" begin
        # The bug. Gas must be free to leave at the GAS velocity.
        f = Set(build_drift_zero_faces((inlet, outlet, wall, symX, symY), mesh))
        @test !any(in(f), 11:20)
        # Every Neumann-family patch behaves the same way, not just Zerogradient.
        for bc in (outlet, extrap, neum)
            g = Set(build_drift_zero_faces((wall, bc), mesh))
            @test !any(in(g), bc.IDs_range)
            @test all(in(g), 21:30)
        end
    end

    @testset "symmetry is the largest group and must not be dropped" begin
        # The quarter-pipe case has 2640 symmetry faces against 240 wall faces:
        # a leak there would corrupt the void field along the whole domain, not
        # just in a boundary layer. Guarding against a "walls only" reading of
        # the fix.
        f = build_drift_zero_faces((symX, symY), mesh)
        @test length(f) == 20
        @test f == collect(31:50)
    end

    @testset "classification is by type, not by patch name" begin
        # A patch called :outlet that is declared as a Wall must still be zeroed,
        # and one called :wall declared Zerogradient must not be - names are not
        # load bearing anywhere else in the solver either.
        odd_wall = Wall(:outlet, noSlip, 11:20)
        odd_out  = Zerogradient(:wall, 0, 21:30)
        f = Set(build_drift_zero_faces((odd_wall, odd_out), mesh))
        @test all(in(f), 11:20)
        @test !any(in(f), 21:30)
    end

    @testset "result is sorted, unique and inside the boundary range" begin
        f = build_drift_zero_faces((inlet, outlet, wall, symX, symY), mesh)
        @test issorted(f)
        @test length(unique(f)) == length(f)
        @test all(1 .<= f .<= length(mesh.boundary_cellsID))
        # An interior face would stop the phases slipping mid-domain, so a patch
        # range outside the boundary block is an error rather than a warning.
        @test_throws ArgumentError build_drift_zero_faces((Wall(:w, noSlip, 65:80),), mesh)
    end

    @testset "no eligible patches gives an empty list, not an error" begin
        @test isempty(build_drift_zero_faces((outlet,), mesh))
        @test isempty(build_drift_zero_faces((), mesh))
    end
end
