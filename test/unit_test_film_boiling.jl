using XCALibre
using Test

# =============================================================================
#  Departure from nucleate boiling and the film boiling regime
# =============================================================================
#
#  Same style as `unit_test_wall_boiling.jl`: pure scalar physics on a
#  `BoilingState`, no mesh and no solver.
#
#  The properties are saturated LH2/GH2 at 0.7 MPa from the Helmholtz H2 EOS,
#  matching the nucleate boiling tests so the two sets can be read together,
#  with vapour transport properties added - film boiling transports heat THROUGH
#  the vapour and so needs `cp_v`, `k_v` and `mu_v`, which nucleate boiling
#  never touches.
# =============================================================================

const FB_T_SAT = 29.15
const FB_RHO_L = 56.75
const FB_RHO_V = 8.84
const FB_CP_L  = 24604.0
const FB_K_L   = 0.0953
const FB_MU_L  = 6.99e-6
const FB_SIGMA = 1.1e-3
const FB_H_FG  = 327.4e3
const FB_G_MAG = 9.81

const FB_CP_V  = 14800.0
const FB_K_V   = 0.0246
const FB_MU_V  = 1.55e-6

const FB_T_CRIT = 33.145        # hydrogen critical temperature [K]

fb_state(; T_w, T_l = FB_T_SAT, vapour = true) = BoilingState(
    T_w = T_w, T_l = T_l, T_sat = FB_T_SAT,
    rho_l = FB_RHO_L, rho_v = FB_RHO_V, cp_l = FB_CP_L, k_l = FB_K_L,
    mu_l = FB_MU_L, sigma = FB_SIGMA, h_fg = FB_H_FG, g = FB_G_MAG,
    cp_v = vapour ? FB_CP_V : 0.0,
    k_v  = vapour ? FB_K_V  : 0.0,
    mu_v = vapour ? FB_MU_V : 0.0)

@testset "Film boiling" begin

    @testset "vapour properties default to zero and are optional" begin
        # Nucleate boiling must keep working without them - that is what makes
        # this an additive change to `BoilingState` rather than a breaking one.
        s = BoilingState(
            T_w = FB_T_SAT + 1.0, T_l = FB_T_SAT, T_sat = FB_T_SAT,
            rho_l = FB_RHO_L, rho_v = FB_RHO_V, cp_l = FB_CP_L, k_l = FB_K_L,
            mu_l = FB_MU_L, sigma = FB_SIGMA, h_fg = FB_H_FG, g = FB_G_MAG)
        @test s.cp_v == 0.0
        @test s.k_v == 0.0
        @test s.mu_v == 0.0
    end

    @testset "Zuber critical heat flux" begin
        s = fb_state(T_w = FB_T_SAT)
        q = critical_heat_flux(Zuber(), s)

        # Recompute the closed form independently.
        expected = 0.131*FB_H_FG*sqrt(FB_RHO_V)*
                   (FB_SIGMA*FB_G_MAG*(FB_RHO_L - FB_RHO_V))^0.25
        @test q ≈ expected

        # Hydrogen CHF at this pressure is of order 10^5 W/m^2. A correlation
        # that came back orders away would be a units error, which is the
        # failure this guards.
        @test 1e4 < q < 1e6

        # Scaling with the coefficient is linear, which is what makes `C` usable
        # as a flow multiplier.
        @test critical_heat_flux(Zuber(C = 0.262), s) ≈ 2*q

        # No CHF once the phases have the same density: there is no buoyant
        # separation to become unstable.
        s_crit = BoilingState(
            T_w = FB_T_SAT, T_l = FB_T_SAT, T_sat = FB_T_SAT,
            rho_l = FB_RHO_V, rho_v = FB_RHO_V, cp_l = FB_CP_L, k_l = FB_K_L,
            mu_l = FB_MU_L, sigma = FB_SIGMA, h_fg = FB_H_FG, g = FB_G_MAG)
        @test critical_heat_flux(Zuber(), s_crit) == Inf
    end

    @testset "FixedCriticalHeatFlux passes a value through" begin
        s = fb_state(T_w = FB_T_SAT)
        @test critical_heat_flux(FixedCriticalHeatFlux(q = 64.0e3), s) == 64.0e3
    end

    @testset "homogeneous nucleation bounds the superheat" begin
        s = fb_state(T_w = FB_T_SAT)
        m = HomogeneousNucleation(T_crit = FB_T_CRIT)
        @test minimum_film_superheat(m, s) ≈ 0.9*FB_T_CRIT - FB_T_SAT

        # The bound is what makes cryogens different: the entire boiling curve
        # has to fit inside a few kelvin, because there is nowhere else for it
        # to go before the liquid cannot exist.
        @test minimum_film_superheat(m, s) < 5.0

        # Saturation above the limit leaves no superheat at all rather than a
        # negative one.
        s_hot = BoilingState(
            T_w = 32.0, T_l = 32.0, T_sat = 32.0,
            rho_l = FB_RHO_L, rho_v = FB_RHO_V, cp_l = FB_CP_L, k_l = FB_K_L,
            mu_l = FB_MU_L, sigma = FB_SIGMA, h_fg = FB_H_FG, g = FB_G_MAG)
        @test minimum_film_superheat(m, s_hot) == 0.0

        @test_throws ArgumentError HomogeneousNucleation(T_crit = -1.0)
        @test_throws ArgumentError HomogeneousNucleation(T_crit = 33.0, C = 1.5)
    end

    @testset "Berenson needs vapour properties" begin
        @test minimum_film_superheat(Berenson(), fb_state(T_w = FB_T_SAT,
                                                          vapour = false)) == 0.0
        @test minimum_film_superheat(Berenson(), fb_state(T_w = FB_T_SAT)) > 0.0
    end

    @testset "Berenson overshoots the thermodynamic limit for hydrogen" begin
        # This is the reason `FilmBoiling` accepts a tuple and takes the minimum.
        # Berenson is a pool-boiling correlation whose scales come from
        # water-like fluids; applied to hydrogen it returns a superheat far
        # beyond the critical temperature, i.e. a state the liquid cannot reach.
        s = fb_state(T_w = FB_T_SAT)
        dT_ber = minimum_film_superheat(Berenson(), s)
        dT_hn = minimum_film_superheat(HomogeneousNucleation(T_crit = FB_T_CRIT), s)

        @test dT_ber > dT_hn
        @test FB_T_SAT + dT_ber > FB_T_CRIT      # unreachable, as claimed

        # The combined closure must pick the smaller, so the unreachable branch
        # cannot be selected by accident.
        fb = FilmBoiling(chf = Zuber(),
                         minimum_film = (Berenson(),
                                         HomogeneousNucleation(T_crit = FB_T_CRIT)))
        @test minimum_film_superheat(fb.minimum_film, s) ≈ dT_hn
    end

    @testset "FixedMinimumFilmBoiling passes a value through" begin
        @test minimum_film_superheat(FixedMinimumFilmBoiling(dT = 4.0),
                                     fb_state(T_w = FB_T_SAT)) == 4.0
    end

    @testset "film heat transfer coefficients" begin
        s = fb_state(T_w = FB_T_SAT + 10.0)

        h_forced = film_boiling_htc(ForcedConvectionFilm(), s, 40.0, 0.3)
        h_bromley = film_boiling_htc(Bromley(D = 6.0e-3), s, 40.0, 0.3)

        @test h_forced > 0
        @test h_bromley > 0

        # Film boiling is a POOR heat transfer mode - that is the whole point of
        # CHF. Both must land well below a liquid-side coefficient, which for
        # this case is of order 10^4 W/m^2/K.
        @test h_forced < 1.0e4
        @test h_bromley < 1.0e4

        # Forced convection at a real flow velocity beats a buoyant film.
        @test h_forced > h_bromley

        # Without vapour properties there is nothing to conduct through.
        @test film_boiling_htc(ForcedConvectionFilm(),
                               fb_state(T_w = FB_T_SAT + 10.0, vapour = false),
                               40.0, 0.3) == 0.0

        # Bromley weakens as the film thickens with superheat: h ~ dT^(-1/4)
        # before the sensible-heat correction, so it must DECREASE.
        h_hot = film_boiling_htc(Bromley(D = 6.0e-3),
                                 fb_state(T_w = FB_T_SAT + 40.0), 40.0, 0.3)
        @test h_hot < h_bromley

        @test_throws ArgumentError Bromley(D = -1.0)
    end

    @testset "film boiling requires a wall thermal capacity" begin
        # Past CHF the curve turns over, so the algebraic inversion has multiple
        # roots. The constructor must refuse rather than silently pick one.
        fb = FilmBoiling(chf = Zuber(), minimum_film = Berenson())
        @test_throws ArgumentError RPI(patches = (:wall,), film_boiling = fb)
        @test_throws ArgumentError RPI(patches = (:wall,), film_boiling = fb,
                                       wall_capacity = 0.0)

        rpi = RPI(patches = (:wall,), film_boiling = fb, wall_capacity = 80.0)
        @test rpi.film_boiling === fb

        # And it stays optional: no film boiling model is still the default.
        @test RPI(patches = (:wall,)).film_boiling === nothing
    end

    @testset "the blend interval is ordered and non-degenerate" begin
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  film_boiling = FilmBoiling(
                      chf = Zuber(),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        fc = film_closure(rpi, rpi.film_boiling, s, 1.0e4, 40.0, 0.3)

        @test fc.dT_lo >= 0
        @test fc.dT_hi > fc.dT_lo          # never inverted, never zero width
        @test isfinite(fc.h_f)

        # `min_width` is a floor on the interval, so it binds whenever the two
        # correlations would otherwise collide.
        @test fc.dT_hi >= fc.dT_lo*1.05
    end

    @testset "no film boiling model leaves the partition untouched" begin
        rpi = RPI(patches = (:wall,))
        s = fb_state(T_w = FB_T_SAT + 2.0)
        h_c = 1.0e4

        @test film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3) === nothing

        p3 = wall_heat_partition(rpi, s, h_c)
        p4 = wall_heat_partition(rpi, s, h_c, nothing)

        @test p4.q_c ≈ p3.q_c
        @test p4.q_q ≈ p3.q_q
        @test p4.q_e ≈ p3.q_e
        @test p4.q_f == 0.0
        @test p4.w == 0.0
    end

    @testset "the blend weight is a smoothstep on superheat" begin
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  film_boiling = FilmBoiling(
                      chf = Zuber(),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        fc = film_closure(rpi, rpi.film_boiling, s, 1.0e4, 40.0, 0.3)

        @test film_boiling_fraction(fc, fc.dT_lo, 0.0) == 0.0
        @test film_boiling_fraction(fc, fc.dT_hi, 0.0) == 1.0
        @test film_boiling_fraction(fc, 0.5*(fc.dT_lo + fc.dT_hi), 0.0) ≈ 0.5

        # Clamped outside, so a wall far past the transition does not overshoot.
        @test film_boiling_fraction(fc, fc.dT_lo - 5.0, 0.0) == 0.0
        @test film_boiling_fraction(fc, 10*fc.dT_hi, 0.0) == 1.0

        # Monotone through the transition.
        ws = [film_boiling_fraction(fc, dT, 0.0)
              for dT in range(fc.dT_lo, fc.dT_hi, length = 25)]
        @test issorted(ws)

        # C1 at the ends is what stops the wall chattering at DNB: a smoothstep
        # has zero SLOPE at both ends, so the numerical derivative of the first
        # interval is far smaller than that of the middle one.
        d_end = ws[2] - ws[1]
        d_mid = ws[13] - ws[12]
        @test d_end < 0.25*d_mid
    end

    @testset "the boiling curve turns over past CHF" begin
        # The defining property of the post-CHF regime, and the reason the
        # algebraic inversion is refused: q_w(T_w) must DECREASE somewhere.
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  film_boiling = FilmBoiling(
                      chf = Zuber(),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        h_c = 1.0e4
        fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)

        total(dT) = begin
            p = wall_heat_partition(rpi, fb_state(T_w = FB_T_SAT + dT), h_c, fc)
            p.q_c + p.q_q + p.q_e + p.q_f
        end

        q_at_chf = total(fc.dT_lo)
        q_at_min = total(fc.dT_hi)

        # At the CHF superheat the partition reproduces the CHF correlation -
        # that is how `dT_lo` was constructed, and it is the check that the
        # inversion inside `film_closure` actually converged.
        @test q_at_chf ≈ critical_heat_flux(Zuber(), s) rtol = 1e-3

        # Turnover.
        @test q_at_min < q_at_chf

        # And the film branch rises again beyond it, so the curve has the
        # characteristic minimum rather than falling away for ever.
        @test total(2*fc.dT_hi) > q_at_min
    end

    @testset "components always sum to the wall flux" begin
        # The partition is reported already scaled by the blend weight and the
        # CHF cap, so the four components sum to the wall flux on every branch.
        # A diagnostic that only added up on one side of DNB would be worse than
        # none.
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  film_boiling = FilmBoiling(
                      chf = Zuber(),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        h_c = 1.0e4
        fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)

        for dT in (0.1, 1.0, fc.dT_lo, 0.5*(fc.dT_lo + fc.dT_hi), fc.dT_hi, 20.0)
            st = fb_state(T_w = FB_T_SAT + dT)
            p_n = wall_heat_partition(rpi, st, h_c)          # unblended
            p = wall_heat_partition(rpi, st, h_c, fc)
            q_n = p_n.q_c + p_n.q_q + p_n.q_e
            keep = (1 - p.w)*min(1.0, fc.q_chf/max(q_n, eps()))

            @test p.q_c ≈ keep*p_n.q_c
            @test p.q_q ≈ keep*p_n.q_q
            @test p.q_e ≈ keep*p_n.q_e
            @test p.q_f ≈ p.w*fc.h_f*max(dT, fc.dT_hi)
            @test all(≥(0), (p.q_c, p.q_q, p.q_e, p.q_f))
        end
    end

    @testset "the nucleate branch is capped at CHF" begin
        # Without the cap a site density calibrated with a large exponent makes
        # the evaporative term run away across the transition faster than the
        # linear blend weight can suppress it, and the curve spikes instead of
        # turning over. `n = 21.17` is the LH2 calibration; at the 0.4 MPa
        # operating point of `dnb_film_boiling_preview.jl`, where the transition
        # is 2 K wide, the uncapped total reaches 10^7 kW/m^2 mid-transition.
        # The margin here is smaller only because this state's transition is
        # narrower - the mechanism is the same.
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  site_density = LemmertChawla(m = 1.105, n = 21.17),
                  film_boiling = FilmBoiling(
                      chf = FixedCriticalHeatFlux(q = 64.0e3),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        h_c = 1.0e4
        fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)

        total(dT) = begin
            p = wall_heat_partition(rpi, fb_state(T_w = FB_T_SAT + dT), h_c, fc)
            p.q_c + p.q_q + p.q_e + p.q_f
        end

        # The unblended partition genuinely does explode - this is the thing
        # being defended against, so assert it rather than assume it.
        p_raw = wall_heat_partition(rpi, fb_state(T_w = FB_T_SAT + fc.dT_hi), h_c)
        @test p_raw.q_c + p_raw.q_q + p_raw.q_e > 2*fc.q_chf

        # Nothing anywhere in the blend may exceed CHF: below it the partition
        # is under the cap, above it the cap binds and the film branch is lower
        # still.
        for dT in range(0.05, 4*fc.dT_hi, length = 200)
            @test total(dT) <= fc.q_chf*(1 + 1e-9)
        end

        # Monotone decreasing through the transition - the actual boiling curve
        # shape, which the uncapped form did not have.
        qs = [total(dT) for dT in range(fc.dT_lo, fc.dT_hi, length = 30)]
        @test issorted(qs, rev = true)
    end

    @testset "void-driven transition" begin
        # The superheat driver needs a CHF value to locate the transition, which
        # is exactly what a new geometry or fluid does not have. The void driver
        # lets departure EMERGE from the solution instead: vapour accumulates at
        # the wall until liquid can no longer reach it.
        mk(tr) = RPI(patches = (:wall,), wall_capacity = 80.0,
                     film_boiling = FilmBoiling(
                         chf = Zuber(),
                         minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT),
                         transition = tr))

        s = fb_state(T_w = FB_T_SAT)
        h_c = 1.0e4

        @testset "superheat driver ignores the void" begin
            rpi = mk(SuperheatTransition())
            fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)
            for dT in (0.5*fc.dT_lo, fc.dT_lo, 0.5*(fc.dT_lo + fc.dT_hi), fc.dT_hi)
                @test film_boiling_fraction(fc, dT, 0.0) ==
                      film_boiling_fraction(fc, dT, 0.99)
            end
            # And it is the default, so existing setups are untouched.
            @test RPI(patches = (:wall,), wall_capacity = 80.0,
                      film_boiling = FilmBoiling(
                          chf = Zuber(),
                          minimum_film = Berenson())).film_boiling.transition isa
                  SuperheatTransition
        end

        @testset "void driver ignores the superheat" begin
            rpi = mk(VoidTransition(alpha_1 = 0.8, alpha_2 = 0.95))
            fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)
            for av in (0.0, 0.5, 0.85, 0.95, 1.0)
                @test film_boiling_fraction(fc, 0.1, av) ==
                      film_boiling_fraction(fc, 50.0, av)
            end

            # Smoothstep on the void, with the same C1 ends as the superheat arm.
            @test film_boiling_fraction(fc, 1.0, 0.8) == 0.0
            @test film_boiling_fraction(fc, 1.0, 0.95) == 1.0
            @test film_boiling_fraction(fc, 1.0, 0.875) ≈ 0.5
            @test film_boiling_fraction(fc, 1.0, 0.0) == 0.0
            @test film_boiling_fraction(fc, 1.0, 1.0) == 1.0

            ws = [film_boiling_fraction(fc, 1.0, a)
                  for a in range(0.8, 0.95, length = 25)]
            @test issorted(ws)
            @test (ws[2] - ws[1]) < 0.25*(ws[13] - ws[12])   # flat at the ends

            @test_throws ArgumentError VoidTransition(alpha_1 = 0.9, alpha_2 = 0.8)
            @test_throws ArgumentError VoidTransition(alpha_1 = -0.1, alpha_2 = 0.9)
        end

        @testset "measures: which void the driver reads" begin
            near  = mk(VoidTransition(measure = NearWallCell()))
            layer = mk(VoidTransition(measure = BubblyLayerAverage(cap = 0.6e-3)))

            # `alpha` is the LIQUID fraction, so a cell at alpha = 0.10 is 90%
            # vapour. `NearWallCell` reads that; `BubblyLayerAverage` reads the
            # layer average the solver assembles and ignores the cell entirely.
            @test wall_void_fraction(near.film_boiling,  0.10, 0.40) ≈ 0.90
            @test wall_void_fraction(layer.film_boiling, 0.10, 0.40) ≈ 0.40

            # Only the layer measure needs the solver-side stencil.
            @test !needs_layer_average(near.film_boiling)
            @test needs_layer_average(layer.film_boiling)
            @test !needs_layer_average(mk(SuperheatTransition()).film_boiling)

            # A superheat-driven model reads neither.
            @test wall_void_fraction(mk(SuperheatTransition()).film_boiling,
                                     0.10, 0.40) == 0.0

            @test_throws ArgumentError BubblyLayerAverage(cap = 0.0)
            @test_throws ArgumentError BubblyLayerAverage(cap = -1.0)
        end

        @testset "layer thickness is D_d, bounded by the cap" begin
            # THE MESH-INDEPENDENCE CLAIM. The averaging distance is set by the
            # departure diameter - physics the model already computes - not by
            # the wall cell height, so refining the mesh changes only how
            # accurately the integral is evaluated.
            layer = mk(VoidTransition(measure = BubblyLayerAverage(cap = 0.6e-3)))
            @test void_layer_thickness_for(layer.film_boiling, 0.2e-3) ≈ 0.2e-3
            @test void_layer_thickness_for(layer.film_boiling, 0.6e-3) ≈ 0.6e-3
            @test void_layer_thickness_for(layer.film_boiling, 1.11e-3) ≈ 0.6e-3

            # The cap matters: on the LH2 pipe the Fritz departure diameter is
            # 1.11 mm against a 3 mm radius, so an uncapped layer would reach
            # 37% of the way to the axis and stop being a near-wall measure.
            @test void_layer_thickness_for(layer.film_boiling, 1.11e-3) < 1.11e-3

            # Measures that do not average return zero, and nothing reads it.
            @test void_layer_thickness_for(
                mk(VoidTransition(measure = NearWallCell())).film_boiling, 1e-3) == 0.0
            @test void_layer_thickness_for(
                mk(SuperheatTransition()).film_boiling, 1e-3) == 0.0
        end

        @testset "partition still closes under a void transition" begin
            rpi = mk(VoidTransition(alpha_1 = 0.8, alpha_2 = 0.95))
            fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)
            for av in (0.0, 0.85, 0.9, 1.0), dT in (0.5, 2.0, 10.0)
                p = wall_heat_partition(rpi, fb_state(T_w = FB_T_SAT + dT), h_c, fc, av)
                tot = p.q_c + p.q_q + p.q_e + p.q_f
                @test all(≥(0), (p.q_c, p.q_q, p.q_e, p.q_f))
                @test isfinite(tot)
                # Nucleate side is still capped at CHF whichever driver is used -
                # the cap is about what nucleate boiling can deliver, not about
                # how the transition is located.
                @test p.q_c + p.q_q + p.q_e <= fc.q_chf*(1 + 1e-9)
            end
        end
    end

    @testset "transient wall solve integrates through DNB" begin
        # The whole architectural argument: with a wall capacity the wall FOLLOWS
        # the curve, so a flux above CHF drives it past the turnover and up onto
        # the film branch instead of the solver having to choose a root.
        rpi = RPI(patches = (:wall,), wall_capacity = 80.0,
                  film_boiling = FilmBoiling(
                      chf = Zuber(),
                      minimum_film = HomogeneousNucleation(T_crit = FB_T_CRIT)))
        s = fb_state(T_w = FB_T_SAT)
        h_c = 1.0e4
        fc = film_closure(rpi, rpi.film_boiling, s, h_c, 40.0, 0.3)

        q_chf = critical_heat_flux(Zuber(), s)
        dt = 1.0e-4

        # Below CHF the wall settles on the nucleate branch and stays there.
        T_w = FB_T_SAT + 0.1
        for _ in 1:4000
            T_w, _ = solve_wall_temperature_transient(
                rpi, fb_state(T_w = T_w), 0.5*q_chf, h_c, T_w, dt, fc)
        end
        @test T_w - FB_T_SAT < fc.dT_lo
        @test film_boiling_fraction(fc, T_w - FB_T_SAT, 0.0) == 0.0

        # Above CHF it cannot balance on the nucleate branch, so it must climb
        # through the transition onto the film branch.
        T_hot = FB_T_SAT + 0.1
        for _ in 1:4000
            T_hot, _ = solve_wall_temperature_transient(
                rpi, fb_state(T_w = T_hot), 1.5*q_chf, h_c, T_hot, dt, fc)
        end
        @test T_hot - FB_T_SAT > fc.dT_hi
        @test film_boiling_fraction(fc, T_hot - FB_T_SAT, 0.0) == 1.0

        # It converged to a steady state rather than running away: one more step
        # must not move it.
        T_next, p = solve_wall_temperature_transient(
            rpi, fb_state(T_w = T_hot), 1.5*q_chf, h_c, T_hot, dt, fc)
        @test T_next ≈ T_hot rtol = 1e-6

        # And that steady state balances the applied flux through the FILM.
        @test p.q_c + p.q_q + p.q_e + p.q_f ≈ 1.5*q_chf rtol = 1e-3
        @test p.q_f ≈ 1.5*q_chf rtol = 1e-3
    end
end
