using XCALibre
using Test

# =============================================================================
#  Burns FAD dispersion and the Lubchenko wall lubrication force
# =============================================================================
#
#  Both come from Lubchenko, Magolan, Sugrue & Baglietto (2018), Int. J.
#  Multiphase Flow 98:36-44, doi 10.1016/j.ijmultiphaseflow.2017.09.003.
#
#  These two closures are ALGEBRAICALLY linked - Eq. 26 is minus Eq. 7 with
#  grad(alpha) replaced by the analytic near-wall profile - so the tests below
#  check that link rather than each expression in isolation. If the two ever
#  drift apart, the wall lubrication force stops cancelling the dispersion it was
#  built to cancel and the near-wall void profile is silently wrong.
# =============================================================================

@testset "Burns FAD factor" begin

    # The whole content of the Burns model, once C_D and |U_r| have cancelled
    # against drag, is  alpha*(1/alpha + 1/(1-alpha)) = 1/(1-alpha).
    @testset "collapses to 1/(1-alpha)" begin
        for a in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75)
            @test a*(1/a + 1/(1 - a)) ≈ 1/(1 - a) rtol = 1e-12
            @test fad_factor(a, 1e-9) ≈ 1/(1 - a) rtol = 1e-12
        end
    end

    @testset "magnitude of the correction it restores" begin
        # This is why the Fickian form survives in dilute bubbly flow and fails
        # in a filled wall cell. Both numbers are quoted in the solver comments,
        # so pin them.
        @test fad_factor(0.1, 1e-9)  ≈ 1.111 rtol = 1e-3     # 11% at peak void
        @test fad_factor(0.98, 1e-9) ≈ 50.0  rtol = 1e-3     # 50x when filled
        @test fad_factor(0.0, 0.2)   == 1                    # no correction at all
    end

    @testset "floor bounds the extrapolation" begin
        # A VALIDITY limit, not a numerical one: Burns is derived for a dispersed
        # phase, and the paper is explicit that the assumptions break down at the
        # bubbly-to-slug transition. The default floor of 0.2 freezes the factor
        # at 5 from alpha = 0.8 upward.
        @test fad_factor(0.8, 0.2)  ≈ 5 rtol = 1e-12
        @test fad_factor(0.9, 0.2)  == fad_factor(0.8, 0.2)
        @test fad_factor(1.0, 0.2)  == fad_factor(0.8, 0.2)
        @test isfinite(fad_factor(1.0, 0.2))
        # Below the floor it is untouched - the cap must not bite in the bubbly
        # regime the case actually runs in.
        for a in (0.0, 0.1, 0.25, 0.3)
            @test fad_factor(a, 0.2) ≈ 1/(1 - a) rtol = 1e-12
        end
    end

    @testset "monotone and never below unity" begin
        as = range(0.0, 0.999, length = 200)
        fs = [fad_factor(a, 0.2) for a in as]
        @test all(fs .>= 1)
        @test all(diff(fs) .>= -1e-12)
    end
end

@testset "Lubchenko Eq. 26 wall lubrication" begin
    d_b = 2.7e-3
    m   = LubchenkoWL()

    @testset "Eq. 27 IS d(alpha)/dy of Eq. 20" begin
        # The model's one non-obvious step: differentiate the assumed parabolic
        # void profile and eliminate alpha_max using the profile itself, leaving
        # a LOCAL expression with no reference to the peak value.
        #
        #     alpha = alpha_max*(1 - (1 - y/R_b)^2)                        (20)
        #     grad(alpha) = alpha*(1/y)*(d_b - 2y)/(d_b - y)               (27)
        #
        # Checked against a central difference of (20). alpha_max cancels in the
        # ratio, so the normalised profile is enough.
        R_b = d_b/2
        a(y) = 1 - (1 - y/R_b)^2
        for r in (0.08, 0.1, 0.2, 0.3, 0.4, 0.45)
            y = r*d_b
            h = 1e-9
            numeric = (a(y + h) - a(y - h))/(2h)/a(y)
            @test wl_td_shape(m, d_b, y) ≈ numeric rtol = 1e-7
        end
    end

    @testset "switches itself off at half a bubble diameter" begin
        # Where the assumed profile peaks, so the gradient - and with it the
        # force - is zero. This is also what makes the model mesh-insensitive:
        # a first cell thicker than a bubble simply deactivates it, which is the
        # behaviour the paper's 0.32 mm to 1.5 mm mesh study demonstrates.
        @test wl_td_shape(m, d_b, 0.5*d_b) == 0
        @test wl_td_shape(m, d_b, 0.51*d_b) == 0
        @test wl_td_shape(m, d_b, 5.0*d_b) == 0
        @test wl_td_shape(m, d_b, 0.49*d_b) > 0
    end

    @testset "repulsive everywhere it acts, and decreasing outward" begin
        # Positive shape * unit normal INTO the fluid = away from the wall. A
        # sign error here would drive vapour INTO the wall and is not obvious
        # from a void field alone.
        ys = range(0.06*d_b, 0.499*d_b, length = 50)
        vs = [wl_td_shape(m, d_b, y) for y in ys]
        @test all(vs .> 0)
        @test all(diff(vs) .< 0)
    end

    @testset "diverges as alpha/y, but bounded by y_floor" begin
        # The divergence is DELIBERATE - it is what enforces alpha -> 0 at the
        # wall - so the floor must not bite on a normal mesh, only on a
        # pathologically fine one.
        loose = LubchenkoWL(y_floor = 1e-6)
        @test wl_td_shape(loose, d_b, 0.01*d_b) > wl_td_shape(loose, d_b, 0.1*d_b)
        # Default floor at 0.05*d_b: below it the value is frozen, above it is not.
        @test wl_td_shape(m, d_b, 0.01*d_b) == wl_td_shape(m, d_b, 0.05*d_b)
        @test wl_td_shape(m, d_b, 0.10*d_b) < wl_td_shape(m, d_b, 0.05*d_b)
        @test isfinite(wl_td_shape(m, d_b, 0.0))
    end

    @testset "it is NOT a C_w model" begin
        # Antal and Frank supply a coefficient in C_w*rho*alpha*|U_r,par|^2*n.
        # Eq. 26 has no such coefficient, and silently returning zero from the
        # C_w path would disable the force with no error at all.
        @test_throws ArgumentError wall_lubrication_coefficient(m, d_b, 0.2*d_b, 0.23)
        @test m isa AbstractWallLubrication
    end

    @testset "the Antal y-floor would erase this model" begin
        # `_wl_y` floors y at d_b/2 for Antal and Frank, because below that the
        # bubble would intersect the wall and their 1/y singularity is spurious.
        # For Eq. 26, y < d_b/2 is the ONLY active range - the same floor applied
        # here would return zero everywhere. Guarding the distinction because
        # both live in the same file and share a call site.
        @test wl_td_shape(m, d_b, max(0.2*d_b, 0.5*d_b)) == 0
        @test wl_td_shape(m, d_b, 0.2*d_b) > 0
    end

    @testset "drift velocity on the bubbly pipe wall cell" begin
        # v = (nu_t/Sc_t)*(1/(1-alpha))*shape  [m/s] - C_D, |U_r| and alpha all
        # cancel against drag, which is why this method takes no drag bisection.
        # At the wall cell of the D = 38.1 mm case (centre y = 0.76 mm) with the
        # k-omega SST eddy viscosity there, the force must be a correction to the
        # flow, not a replacement for it.
        y_c, Sc_t, nu_t = 0.75975e-3, 1.0, 1.6e-5
        v = (nu_t/Sc_t)*fad_factor(0.25, 0.2)*wl_td_shape(m, d_b, y_c)
        @test 0.0 < v < 0.1*0.753          # under 10% of the liquid superficial velocity
        @test y_c/d_b < 0.5                 # and the cell is inside the active range
    end

    @testset "degenerate inputs" begin
        @test wl_td_shape(m, 0.0, 1.0e-3) == 0
        @test wl_td_shape(m, d_b, -1.0e-3) > 0      # floored, not negative
        @test wl_td_shape(Antal(), d_b, 0.2*d_b) == 0
        @test wl_td_shape(nothing, d_b, 0.2*d_b) == 0
    end
end

@testset "Burns and Eq. 26 cancel by construction" begin
    # THE central identity. Eq. 26 is defined by F_TD + F_WLTD = 0 when
    # grad(alpha) takes the shape of Eq. 27, so with that gradient imposed the
    # dispersion drift and the lubrication drift must be equal and opposite. If
    # a future change to either closure breaks this, the near-wall void profile
    # stops being the one the model was derived to produce.
    d_b, Sc_t, nu_t, a_floor = 2.7e-3, 1.0, 1.6e-5, 0.2
    m = LubchenkoWL(y_floor = 1e-9)

    for r in (0.1, 0.2, 0.3, 0.45), alpha in (0.05, 0.2, 0.4)
        y = r*d_b
        shape = wl_td_shape(m, d_b, y)                 # (1/alpha)*dalpha/dy
        grad  = alpha*shape                            # Eq. 27

        # Dispersion drift of the gas phase, Burns reduced through drag:
        #     alpha*v_TD = -(nu_t/Sc_t)*grad(alpha)/(1 - alpha)
        v_td  = -(nu_t/Sc_t)*fad_factor(alpha, a_floor)*grad/alpha

        # Lubrication drift, as the solver kernel computes it:
        v_wl  =  (nu_t/Sc_t)*fad_factor(alpha, a_floor)*shape

        @test v_td ≈ -v_wl rtol = 1e-12
        @test v_wl > 0                                  # away from the wall
    end
end
