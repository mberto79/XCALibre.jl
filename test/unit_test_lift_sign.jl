using XCALibre
using Test

# =============================================================================
#  Tomiyama lift: the sign change is the model
# =============================================================================
#
#  The lift force is what makes bubbly upflow WALL PEAKED, and its sign is set
#  by bubble size alone:
#
#      F_L = -C_L*rho_c*alpha_d*(U_r x curl(U_c))
#
#      C_L > 0  ->  bubbles driven TOWARD the wall  ->  wall-peaked void
#      C_L < 0  ->  bubbles driven toward the core  ->  core-peaked void
#
#  Tomiyama's correlation reverses sign at the root of its polynomial branch,
#  Eo_d = 6.06 - NOT at Eo_d = 4, which is only where that branch takes over
#  from the capped small-bubble form (fE is still +0.205 there). For AIR-WATER
#  the reversal is a ~5.9 mm bubble, and the resulting
#  inversion of the radial void profile is a MEASURED result - Liu & Bankoff
#  (1993), Int. J. Heat Mass Transfer 36:1049, observed exactly this transition
#  as bubble size was increased at fixed flow rate.
#
#  These tests pin the correlation itself. The profile inversion it predicts is
#  exercised by the adiabatic bubbly pipe case, which needs a mesh; this file
#  needs nothing and therefore runs in CI.
#
#  WHY IT IS WORTH PINNING. The sign convention here is easy to get wrong and
#  silent when wrong - a flipped lift still produces a plausible-looking void
#  field, just peaked in the wrong place. On the LH2 pipe the ABSENCE of lift
#  held near-wall void at a ceiling of 0.70 for weeks while dispersion, mesh
#  resolution and the dryout closure were all investigated in turn.
# =============================================================================

@testset "Tomiyama lift coefficient" begin

    # Air-water at 20 C, 1 atm.
    RHO_L, RHO_G = 998.2, 1.204
    SIGMA        = 0.0728         # [N/m]
    MU_L         = 1.002e-3
    G            = 9.81
    DRHO         = RHO_L - RHO_G

    # Eotvos numbers as the correlation forms them, for documenting the tests.
    eo(d)   = G*DRHO*d^2/SIGMA
    d_h(d)  = d*cbrt(1 + 0.163*eo(d)^0.757)
    eo_d(d) = G*DRHO*d_h(d)^2/SIGMA

    # A slip velocity in the range a few-mm air bubble actually rises at, so
    # Re_p lands where the tanh factor is saturated and C_L is set by size.
    U_R = 0.23                     # [m/s]
    re_p(d) = RHO_L*U_R*d/MU_L

    C_L(d) = lift_coefficient(TomiyamaLift(), d, DRHO, SIGMA, G, re_p(d))

    @testset "small bubbles give POSITIVE C_L (wall peaking)" begin
        # 3 mm: Eo_d ~ 1.36, well below the first crossing.
        @test eo_d(3.0e-3) < 4.0
        @test C_L(3.0e-3) > 0
        # In this regime C_L is capped by C_max through tanh(0.121*Re_p), which
        # is saturated at these Reynolds numbers.
        @test C_L(3.0e-3) ≈ 0.288 rtol = 0.02
    end

    @testset "large bubbles give NEGATIVE C_L (core peaking)" begin
        # 7 mm: Eo_d ~ 9.3, inside the 4 < Eo_d <= 10 branch where the
        # polynomial fE has already gone negative.
        @test 4.0 < eo_d(7.0e-3) <= 10.0
        @test C_L(7.0e-3) < 0
    end

    @testset "the crossing is where the correlation says it is" begin
        # Bisect for the diameter at which C_L changes sign, and confirm it
        # coincides with Eo_d = 4 - i.e. that the branch switch, not some other
        # part of the expression, is what flips it.
        lo, hi = 3.0e-3, 7.0e-3
        for _ in 1:60
            mid = 0.5*(lo + hi)
            if C_L(mid) > 0
                lo = mid
            else
                hi = mid
            end
        end
        d_cross = 0.5*(lo + hi)

        # NOTE the crossing is NOT at Eo_d = 4. That is the BRANCH boundary,
        # where the capped form min(C_max*tanh(..), fE) hands over to plain fE;
        # fE is still +0.205 there. The SIGN change is the root of fE itself,
        # at Eo_d = 6.06, which for air-water is a ~5.9 mm bubble.
        @test 5.0e-3 < d_cross < 6.5e-3
        @test eo_d(d_cross) ≈ 6.06 rtol = 0.02

        # Confirm the branch boundary is passed through smoothly rather than
        # being where the sign flips.
        d4 = 4.0e-3
        while eo_d(d4) < 4.0; d4 += 1.0e-5; end
        @test C_L(d4) > 0.15
    end

    @testset "C_L is monotone decreasing through the transition" begin
        ds = range(2.0e-3, 9.0e-3, length = 40)
        cs = [C_L(d) for d in ds]
        # Not strictly monotone across the whole range (the small-bubble branch
        # is flat at C_max), but it must never increase with size.
        @test all(diff(cs) .<= 1e-12)
        @test cs[1] > 0 && cs[end] < 0
    end

    @testset "deformed branch is constant and negative" begin
        # Above Eo_d = 10 Tomiyama returns C_deformed unchanged.
        d_big = 20.0e-3
        @test eo_d(d_big) > 10.0
        @test C_L(d_big) ≈ -0.27 rtol = 1e-6
    end

    @testset "low slip suppresses lift; ConstantLift does not" begin
        # The tanh(0.121*Re_p) factor is what makes the force vanish as slip
        # goes to zero. `ConstantLift` has no such cutoff, which is why it can
        # inject lift into low-slip cells where Tomiyama gives essentially none.
        d = 3.0e-3
        re_small = RHO_L*0.001*d/MU_L                 # 1 mm/s slip
        c_tom = lift_coefficient(TomiyamaLift(), d, DRHO, SIGMA, G, re_small)
        c_con = lift_coefficient(ConstantLift(C_L = 0.288), d, DRHO, SIGMA, G, re_small)
        @test c_tom < 0.5*c_con
        @test c_con ≈ 0.288 rtol = 1e-12
    end

    @testset "degenerate inputs return zero rather than NaN" begin
        @test lift_coefficient(TomiyamaLift(), 3.0e-3, DRHO, 0.0, G, 100.0) == 0
        @test lift_coefficient(TomiyamaLift(), 3.0e-3, 0.0, SIGMA, G, 100.0) == 0
        @test lift_coefficient(nothing, 3.0e-3, DRHO, SIGMA, G, 100.0) == 0
    end
end

# =============================================================================
#  Wall lubrication: the cutoff radius is closed-form
# =============================================================================
#
#  Both models switch off at a known distance from the wall, and those roots are
#  exact - so they are worth asserting rather than trusting.
#
#      Antal:  C_w = max(0, (Cw1 - 0.06*|Ur_par|)/d_b + Cw2/y)
#              zero at  y* = Cw2*d_b/(0.104 + 0.06*|Ur_par|)
#
#      Frank:  cuts off at  y = Cwc*d_b
# =============================================================================

@testset "Wall lubrication cutoff" begin
    d_b, u_par = 3.0e-3, 0.23

    @testset "Antal vanishes at its analytic root" begin
        y_star = 0.147*d_b/(0.104 + 0.06*u_par)
        cw_in  = wall_lubrication_coefficient(Antal(), d_b, 0.5*y_star, u_par)
        cw_out = wall_lubrication_coefficient(Antal(), d_b, 2.0*y_star, u_par)
        @test cw_in > 0
        @test cw_out == 0
        # Continuous approach to zero: the two terms cancel exactly at y*.
        @test wall_lubrication_coefficient(Antal(), d_b, y_star, u_par) ≈ 0 atol = 1e-6
    end

    @testset "Frank cuts off at Cwc bubble diameters" begin
        m = Frank()
        y_cut = 10.0*d_b                       # Cwc = 10 by default
        @test wall_lubrication_coefficient(m, d_b, 0.5*y_cut, u_par) > 0
        @test wall_lubrication_coefficient(m, d_b, 1.01*y_cut, u_par) == 0
    end

    @testset "Frank reaches further than Antal" begin
        # The documented reason to prefer Frank: range, not strength.
        y = 4.0*d_b
        @test wall_lubrication_coefficient(Antal(), d_b, y, u_par) == 0
        @test wall_lubrication_coefficient(Frank(), d_b, y, u_par) > 0
    end

    @testset "repulsion is strictly decreasing with distance" begin
        ys = range(0.2e-3, 8.0e-3, length = 30)
        for m in (Antal(), Frank())
            cs = [wall_lubrication_coefficient(m, d_b, y, u_par) for y in ys]
            @test all(diff(cs) .<= 1e-9)
        end
    end
end

# =============================================================================
#  Shaver & Podowski near-wall lift damping
# =============================================================================
#
#  Lubchenko et al. (2018), Int. J. Multiphase Flow 98:36-44, Eq. 6:
#
#      C_L = 0                                            y/D_b < 0.5
#      C_L = C_L0*(3*(2y/D_b - 1)^2 - 2*(2y/D_b - 1)^3)    0.5 < y/D_b < 1
#      C_L = C_L0                                          y/D_b > 1
#
#  This matters because lift is LARGEST at the wall - that is where the liquid
#  shear is largest - so the undamped force produces an unbounded gas spike in
#  the wall-layer cells, and every remedy in the Antal lineage then overcorrects
#  it. The shape is pinned here rather than trusted because a damping that is
#  merely APPROXIMATELY right is silent when wrong: the void profile stays
#  plausible and only the near-wall peak moves.
# =============================================================================

@testset "Shaver-Podowski lift damping" begin
    d_b = 3.0e-3
    m   = ShaverPodowski(inner = ConstantLift(C_L = 0.025))
    f(r) = lift_wall_damping(m, r*d_b, d_b)

    @testset "the three branches of Eq. 6" begin
        @test f(0.0)  == 0
        @test f(0.25) == 0
        @test f(0.5)  == 0           # closed at the lower end
        @test f(1.0)  == 1
        @test f(5.0)  == 1
        # Interior of the ramp against the polynomial written out longhand.
        for r in (0.55, 0.6, 0.75, 0.9, 0.99)
            t = 2r - 1
            @test f(r) ≈ 3t^2 - 2t^3 rtol = 1e-12
        end
        @test f(0.75) ≈ 0.5 rtol = 1e-12    # smoothstep is symmetric about 3/4
    end

    @testset "C1 continuous at both ends" begin
        # The ramp is a smoothstep precisely so the force does not jump; a
        # discontinuity here would appear as a one-cell kink in the void profile
        # and be easy to misread as a discretisation problem.
        eps_r = 1e-6
        @test f(0.5 + eps_r) < 1e-10             # value AND slope vanish at 0.5
        @test 1 - f(1.0 - eps_r) < 1e-10         # likewise at 1
        @test all(diff([f(r) for r in range(0.4, 1.1, length = 200)]) .>= -1e-12)
    end

    @testset "wraps its inner model without altering the coefficient" begin
        RHO_L, RHO_G, SIGMA, G = 998.2, 1.204, 0.0728, 9.81
        drho, Re_p = RHO_L - RHO_G, 700.0
        for inner in (ConstantLift(C_L = 0.025), TomiyamaLift(C_max = 0.15))
            w = ShaverPodowski(inner)
            @test lift_coefficient(w, d_b, drho, SIGMA, G, Re_p) ==
                  lift_coefficient(inner, d_b, drho, SIGMA, G, Re_p)
        end
        @test lift_coefficient(m, d_b, drho, SIGMA, G, Re_p) ≈ 0.025 rtol = 1e-12
    end

    @testset "every other model is undamped" begin
        # The damping is opt-in: `build_lift_damping` uses the value AT the wall
        # to decide whether a geometry pass is needed at all, so a model that
        # returned anything but 1 there would silently acquire a near-wall
        # correction it was never given.
        for other in (TomiyamaLift(), ConstantLift(C_L = 0.288), nothing)
            @test lift_wall_damping(other, 0.0, d_b) == 1
            @test lift_wall_damping(other, 10*d_b, d_b) == 1
        end
    end

    @testset "degenerate diameter does not divide by zero" begin
        @test lift_wall_damping(m, 1.0e-3, 0.0) == 1
        @test isfinite(lift_wall_damping(m, 0.0, 0.0))
    end

    @testset "on this mesh the wall cell is fully damped" begin
        # The adiabatic bubbly pipe (D = 38.1 mm) has a 1.52 mm wall cell, so its
        # centre sits at y = 0.76 mm = 0.25*d_b - inside the dead zone. That is
        # the cell carrying dUz/dr ~ 125 /s and the void spike, and it is the
        # whole reason this correction was adopted.
        y_c = 0.5*1.5195e-3
        @test y_c/d_b < 0.5
        @test lift_wall_damping(m, y_c, d_b) == 0
        # The second cell is on the ramp, not switched abruptly to full lift.
        @test 0 < lift_wall_damping(m, 2.3e-3, d_b) < 1
    end
end
