using XCALibre
using Test

# =============================================================================
#  RPI wall nucleate boiling - sub-models, flux partition, wall temperature
# =============================================================================
#
#  Pure-physics tests: no mesh, no solver. Everything here is a scalar function
#  of a `BoilingState`, which is what makes the sub-models independently
#  testable and independently replaceable.
# =============================================================================

# Saturated LH2 / GH2 at 0.7 MPa, from the Helmholtz H2 EOS.
const WB_T_SAT = 29.15
const WB_RHO_L = 56.75
const WB_RHO_V = 8.84
const WB_CP_L  = 24604.0
const WB_K_L   = 0.0953
const WB_MU_L  = 6.99e-6
const WB_SIGMA = 1.1e-3
const WB_H_FG  = 327.4e3
const WB_G_MAG = 9.81

wb_make_state(; T_w, T_l = WB_T_SAT) = BoilingState(
    T_w = T_w, T_l = T_l, T_sat = WB_T_SAT,
    rho_l = WB_RHO_L, rho_v = WB_RHO_V, cp_l = WB_CP_L, k_l = WB_K_L, mu_l = WB_MU_L,
    sigma = WB_SIGMA, h_fg = WB_H_FG, g = WB_G_MAG)

@testset "RPI wall boiling" begin

    @testset "BoilingState derives the driving temperature differences" begin
        s = wb_make_state(T_w = WB_T_SAT + 3.0, T_l = WB_T_SAT - 2.0)
        @test s.dT_sup ≈ 3.0
        @test s.dT_sub ≈ 2.0
    end

    @testset "Lemmert-Chawla site density" begin
        m = LemmertChawla()
        @test m.m == 210.0
        @test m.n == 1.805

        # The defining correlation, N_a = (210 dT_sup)^1.805.
        for dT in (0.5, 2.0, 5.0)
            s = wb_make_state(T_w = WB_T_SAT + dT)
            @test nucleation_site_density(m, s) ≈ (210.0*dT)^1.805
        end

        # No sites at or below saturation - this is what makes the partition
        # collapse to pure convection below boiling onset.
        @test nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT)) == 0.0
        @test nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT - 1.0)) == 0.0

        # Strongly superlinear in superheat: doubling dT_sup multiplies N_a by
        # 2^1.805 ~ 3.5. This steepness is why the wall temperature is found by
        # bisection rather than by Newton.
        n1 = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 1.0))
        n2 = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 2.0))
        @test n2/n1 ≈ 2.0^1.805 rtol=1e-10

        # Coefficients are user-settable.
        m2 = LemmertChawla(m = 100.0, n = 2.0)
        @test nucleation_site_density(m2, wb_make_state(T_w = WB_T_SAT + 1.0)) ≈ 100.0^2
    end

    @testset "Hibiki-Ishii site density" begin
        m = HibikiIshii()
        @test nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT)) == 0.0

        # Strictly increasing in superheat while below the cap. For LH2 the cap
        # binds by ~1 K of superheat (the model is well outside its fitted range
        # here - see the docstring warning), so monotonicity is checked in the
        # millikelvin regime where the exponential is still resolved.
        n1 = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 0.005))
        n2 = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 0.010))
        n3 = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 0.020))
        @test 0.0 < n1 < n2 < n3

        # Finite and capped at large superheat rather than overflowing.
        n_big = nucleation_site_density(m, wb_make_state(T_w = WB_T_SAT + 500.0))
        @test isfinite(n_big)
        @test n_big == m.N_max

        @test nucleation_site_density(HibikiIshii(N_max = 1.0e6),
                                      wb_make_state(T_w = WB_T_SAT + 500.0)) == 1.0e6
    end

    @testset "Tolubinsky-Kostanchuk departure diameter" begin
        m = TolubinskyKostanchuk()

        # No subcooling: the reference diameter, unmodified.
        @test bubble_departure_diameter(m, wb_make_state(T_w = WB_T_SAT + 1.0, T_l = WB_T_SAT)) ≈ 0.6e-3

        # Subcooling shrinks the bubble, exponentially.
        d_sub = bubble_departure_diameter(
            m, wb_make_state(T_w = WB_T_SAT + 1.0, T_l = WB_T_SAT - 45.0))
        @test d_sub ≈ 0.6e-3*exp(-1.0) rtol=1e-10

        # Capped both ways.
        m_big = TolubinskyKostanchuk(d_ref = 1.0, d_max = 1.4e-3)
        @test bubble_departure_diameter(m_big, wb_make_state(T_w = WB_T_SAT + 1.0)) ≈ 1.4e-3
        @test bubble_departure_diameter(
            m, wb_make_state(T_w = WB_T_SAT + 1.0, T_l = WB_T_SAT - 1e4)) ≈ m.d_min

        # A superheated bulk must not INFLATE the diameter: negative subcooling
        # is clamped, or exp(+x) would run away.
        d_superheated = bubble_departure_diameter(
            m, wb_make_state(T_w = WB_T_SAT + 5.0, T_l = WB_T_SAT + 3.0))
        @test d_superheated ≈ 0.6e-3
    end

    @testset "Kocamustafaogullari-Ishii departure diameter" begin
        m = KocamustafaogullariIshii()
        s = wb_make_state(T_w = WB_T_SAT + 2.0)
        d = bubble_departure_diameter(m, s)

        # Must be strictly inside the clamps: hitting a clamp means the
        # correlation is not being exercised at all. This caught the contact
        # angle being supplied in radians, which is 57x too small and pinned the
        # result at `d_min`.
        @test m.d_min < d < m.d_max

        # Against the correlation written out longhand.
        drho = WB_RHO_L - WB_RHO_V
        d_fritz = 0.0208*41.37*sqrt(WB_SIGMA/(WB_G_MAG*drho))
        @test d ≈ 0.0012*(drho/WB_RHO_V)^0.9*d_fritz rtol=1e-10

        # Order of magnitude sanity: microns to tens of microns for LH2 at this
        # pressure, not millimetres and not nanometres.
        @test 1.0e-7 < d < 1.0e-4

        # Scales with the capillary length, so it responds to surface tension -
        # the reason to prefer it near the critical point, where sigma -> 0.
        s_low_sigma = BoilingState(
            T_w = WB_T_SAT + 2.0, T_l = WB_T_SAT, T_sat = WB_T_SAT,
            rho_l = WB_RHO_L, rho_v = WB_RHO_V, cp_l = WB_CP_L, k_l = WB_K_L, mu_l = WB_MU_L,
            sigma = WB_SIGMA/4, h_fg = WB_H_FG, g = WB_G_MAG)
        @test bubble_departure_diameter(m, s_low_sigma) ≈ d/2 rtol=1e-10
    end

    @testset "Cole departure frequency" begin
        m = Cole()
        s = wb_make_state(T_w = WB_T_SAT + 2.0)
        D_d = 0.6e-3
        expected = sqrt(4*WB_G_MAG*(WB_RHO_L - WB_RHO_V)/(3*1.0*WB_RHO_L*D_d))
        @test bubble_departure_frequency(m, s, D_d) ≈ expected

        # Diverges as D_d -> 0, so it must be capped; without the cap the
        # quenching flux would be unbounded at high subcooling.
        @test bubble_departure_frequency(m, s, 1e-12) == m.f_max
        @test bubble_departure_frequency(m, s, 0.0) == 0.0
    end

    @testset "Influence area is always a fraction" begin
        s = wb_make_state(T_w = WB_T_SAT + 2.0)

        for m in (DelValleKenning(), ConstantInfluenceArea())
            @test 0.0 <= bubble_influence_fraction(m, s, 1e4, 0.6e-3) <= 1.0
            # The uncapped expression exceeds one at moderate superheat; if the
            # cap were missing the convective term would go NEGATIVE.
            @test bubble_influence_fraction(m, s, 1e9, 0.6e-3) == 1.0
            @test bubble_influence_fraction(m, s, 0.0, 0.6e-3) == 0.0
        end

        # Del Valle-Kenning shrinks with subcooling; the constant model does not.
        dvk = DelValleKenning()
        a_sat = bubble_influence_fraction(dvk, wb_make_state(T_w = WB_T_SAT + 2.0), 1e4, 0.6e-3)
        a_sub = bubble_influence_fraction(
            dvk, wb_make_state(T_w = WB_T_SAT + 2.0, T_l = WB_T_SAT - 40.0), 1e4, 0.6e-3)
        @test a_sub < a_sat
    end

    @testset "Influence area saturation: clamp versus exponential" begin
        s = wb_make_state(T_w = WB_T_SAT + 2.0)

        @test DelValleKenning().saturation === Val(:clamp)          # unchanged default
        @test ConstantInfluenceArea().saturation === Val(:clamp)
        @test_throws ArgumentError DelValleKenning(saturation = :nonsense)
        @test_throws ArgumentError ConstantInfluenceArea(saturation = :nonsense)

        for (clamped, smooth) in (
                (DelValleKenning(), DelValleKenning(saturation = :exponential)),
                (ConstantInfluenceArea(), ConstantInfluenceArea(saturation = :exponential)))

            # Both agree in the dilute limit, where overlap cannot matter.
            @test bubble_influence_fraction(clamped, s, 1.0, 0.6e-3) ≈
                  bubble_influence_fraction(smooth,  s, 1.0, 0.6e-3) rtol = 1e-6

            # Both stay a fraction, and both vanish with no sites.
            @test bubble_influence_fraction(smooth, s, 0.0, 0.6e-3) == 0.0
            @test 0.0 <= bubble_influence_fraction(smooth, s, 1e9, 0.6e-3) <= 1.0

            # THE POINT: at a coverage the clamp has already flattened at, the
            # exponential form is still strictly below 1, so
            # q_conv = h_c*dT*(1 - A_b) survives and keeps responding to the wall
            # temperature. Under the clamp it is identically zero from here on.
            #
            # N_a = 1e7 puts the raw coverage past 1 for both `K = 4.8`
            # (Del Valle-Kenning) and `K = 2` (the constant model), so the clamp
            # has certainly bound. Far beyond it `1 - exp(-x)` does round to
            # exactly 1 in Float64 (around x ~ 40), which is a floating point
            # limit rather than a modelling one and is not a regime any case
            # reaches: on the LH2 calibration A_b tops out at 0.835 at CHF.
            @test bubble_influence_fraction(clamped, s, 1e7, 0.6e-3) == 1.0
            @test bubble_influence_fraction(smooth,  s, 1e7, 0.6e-3) < 1.0

            # Strictly increasing across that whole range, including where the
            # clamp has flattened. This is what removes the slope discontinuity
            # in the boiling curve: on the LH2 calibration the clamped model
            # jumps from d(ln q)/d(ln dT) = 7.9 to 18.1 between 36 and 38 kW/m^2.
            Ns = 10 .^ range(3, 7, length = 40)
            as = [bubble_influence_fraction(smooth, s, N, 0.6e-3) for N in Ns]
            @test issorted(as)
            @test allunique(as)
            @test all(<(1.0), as)

            # The clamp, by contrast, is constant over the upper part of that
            # range - which is exactly the lost sensitivity.
            ac = [bubble_influence_fraction(clamped, s, N, 0.6e-3) for N in Ns]
            @test !allunique(ac)
        end
    end

    @testset "Single-phase heat transfer coefficient" begin
        u_tau = 0.23
        h = single_phase_htc(40.0, u_tau, WB_RHO_L, WB_CP_L, WB_MU_L, WB_K_L, 0.85)
        @test h > 0
        @test isfinite(h)

        # Falls with distance from the wall (T+ grows logarithmically).
        @test single_phase_htc(60.0, u_tau, WB_RHO_L, WB_CP_L, WB_MU_L, WB_K_L, 0.85) < h

        # Grows with friction velocity.
        @test single_phase_htc(40.0, 2*u_tau, WB_RHO_L, WB_CP_L, WB_MU_L, WB_K_L, 0.85) > h

        # Degenerate inputs must not produce Inf/NaN.
        @test single_phase_htc(40.0, 0.0, WB_RHO_L, WB_CP_L, WB_MU_L, WB_K_L, 0.85) == 0.0
        @test single_phase_htc(40.0, u_tau, WB_RHO_L, WB_CP_L, WB_MU_L, 0.0, 0.85) == 0.0
    end

    # -------------------------------------------------------------------------
    rpi = RPI(patches = (:pipeWall,))
    h_c = single_phase_htc(40.0, 0.23, WB_RHO_L, WB_CP_L, WB_MU_L, WB_K_L, 0.85)

    @testset "RPI construction" begin
        @test rpi.site_density isa LemmertChawla
        @test rpi.departure_diameter isa TolubinskyKostanchuk
        @test rpi.departure_frequency isa Cole
        @test rpi.influence_area isa DelValleKenning
        @test rpi.patches == (:pipeWall,)

        # A single symbol is accepted and normalised to a tuple.
        @test RPI(patches = :pipeWall).patches == (:pipeWall,)

        @test_throws ArgumentError RPI(patches = ())
        @test_throws ArgumentError RPI(patches = ("pipeWall",))

        # Sub-models really are swappable through the constructor.
        alt = RPI(patches = (:w,), site_density = HibikiIshii(),
                  departure_diameter = KocamustafaogullariIshii(),
                  influence_area = ConstantInfluenceArea(K = 2.0))
        @test alt.site_density isa HibikiIshii
        @test alt.influence_area isa ConstantInfluenceArea
    end

    @testset "Flux partition" begin
        # At saturation there is no boiling: the partition must collapse
        # EXACTLY to single-phase convection. This is the limit that lets the
        # same routine be used on both sides of onset.
        s0 = wb_make_state(T_w = WB_T_SAT, T_l = WB_T_SAT - 2.0)
        p0 = wall_heat_partition(rpi, s0, h_c)
        @test p0.q_e == 0.0
        @test p0.N_a == 0.0
        @test p0.A_b == 0.0
        @test p0.q_q == 0.0
        @test p0.q_c ≈ h_c*(s0.T_w - s0.T_l)

        # Above onset all three channels are open and non-negative.
        s = wb_make_state(T_w = WB_T_SAT + 3.0, T_l = WB_T_SAT - 1.0)
        p = wall_heat_partition(rpi, s, h_c)
        @test p.q_e > 0
        @test p.q_q > 0
        @test p.q_c >= 0
        @test 0 < p.A_b <= 1

        # The evaporative flux is exactly the latent heat carried by the
        # departing bubbles - the identity the solver relies on when it converts
        # q_e into a vapour mass source.
        @test p.q_e ≈ p.N_a*p.f*(pi/6)*p.D_d^3*WB_RHO_V*WB_H_FG

        # Evaporation grows steeply with superheat.
        p2 = wall_heat_partition(rpi, wb_make_state(T_w = WB_T_SAT + 6.0, T_l = WB_T_SAT - 1.0), h_c)
        @test p2.q_e > p.q_e
    end

    @testset "Wall temperature solve" begin
        T_l = WB_T_SAT - 1.0

        # Below onset the answer is the pure-convection wall temperature, and
        # it must be returned exactly rather than bisected towards.
        q_small = 0.5*h_c*(WB_T_SAT - T_l)
        s = wb_make_state(T_w = T_l, T_l = T_l)
        T_w, _ = solve_wall_temperature(rpi, s, q_small, h_c)
        @test T_w ≈ T_l + q_small/h_c
        @test T_w < WB_T_SAT

        # ONSET FLUX: the flux pure convection can carry with the wall exactly at
        # saturation. Below it there is no boiling; above it there must be.
        #
        # Test fluxes are expressed as multiples of this rather than as absolute
        # numbers. They were absolute (1e4, 3e4, 1e5) and went stale when
        # `single_phase_htc` was corrected: `h_c` rose 3.3x, the onset flux rose
        # with it, and 1e4 W/m^2 stopped being a boiling condition at all - so
        # `T_w > T_sat` failed for the entirely correct reason that the wall was
        # no longer boiling.
        q_onset = h_c*(WB_T_SAT - T_l)

        # Above onset the three components must sum to the imposed flux. This is
        # the whole point of the inversion.
        for mult in (1.5, 3.0, 8.0)
            q_w = mult*q_onset
            T_w, part = solve_wall_temperature(rpi, s, q_w, h_c)
            total = part.q_c + part.q_q + part.q_e
            @test total ≈ q_w rtol=1e-6
            @test T_w > WB_T_SAT              # boiling implies wall superheat
        end

        # REGRESSION: closure must hold where `A_b` is large.
        #
        # `T_l + q_w/h_c` is NOT a valid upper bracket once bubbles cover an
        # appreciable share of the wall, because `q_conv = h_c*dT*(1 - A_b)` is
        # then well short of `q_w` at that temperature. Bisection used to saturate
        # at the top of the bracket and return a partition summing to `q_w*(1-A_b)`
        # - silently, with the wall temperature simply too low. `solve_wall_temperature`
        # now expands the bound until the partition actually reaches `q_w`.
        for mult in (15.0, 40.0)
            q_w = mult*q_onset
            T_w, part = solve_wall_temperature(rpi, s, q_w, h_c)
            @test (part.q_c + part.q_q + part.q_e) ≈ q_w rtol=1e-6
            @test part.A_b > 0.1              # the regime the bracket bug lived in
        end

        # Monotonic in the imposed flux.
        Tw_lo, _ = solve_wall_temperature(rpi, s, 1.5*q_onset, h_c)
        Tw_hi, _ = solve_wall_temperature(rpi, s, 8.0*q_onset, h_c)
        @test Tw_hi > Tw_lo

        # Boiling holds the wall below where pure convection would put it - the
        # physical effect the model exists to capture.
        #
        # Only true while `A_b` is modest. Once bubbles blanket the wall,
        # `q_conv = h_c*dT*(1 - A_b)` is SUPPRESSED and the wall can end up hotter
        # than pure convection would leave it - which is the dryout regime, where
        # RPI is not valid anyway. This is therefore checked at a few multiples of
        # onset (nucleate boiling), not at an arbitrary large flux: the previous
        # 1e5 W/m^2 landed past that crossover once `h_c` was corrected.
        q_w = 3.0*q_onset
        Tw_boil, part_boil = solve_wall_temperature(rpi, s, q_w, h_c)
        @test Tw_boil < T_l + q_w/h_c
        @test part_boil.A_b < 0.5             # confirms we are still in nucleate boiling

        # Zero and negative flux must not blow up.
        Tw0, _ = solve_wall_temperature(rpi, s, 0.0, h_c)
        @test Tw0 ≈ T_l
        @test isfinite(solve_wall_temperature(rpi, s, -1.0e3, h_c)[1])
    end

    @testset "Dryout ramp" begin
        r = RPI(patches = (:w,), alpha_min = 0.1)
        f = XCALibre.ModelPhysics.wall_boiling_liquid_factor

        @test f(r, 1.0) == 1.0          # fully wetted
        @test f(r, 0.25) == 1.0         # above 2*alpha_min
        @test f(r, 0.05) == 0.0         # below alpha_min - source switched off
        @test f(r, 0.15) ≈ 0.5          # linear in between
        @test 0.0 <= f(r, 0.0) <= 1.0
    end

    @testset "Alternative sub-models give a self-consistent partition" begin
        # The point of the abstraction: any combination must still invert.
        for sd in (LemmertChawla(), HibikiIshii()),
            dd in (TolubinskyKostanchuk(), KocamustafaogullariIshii()),
            ia in (DelValleKenning(), ConstantInfluenceArea())

            m = RPI(patches = (:w,), site_density = sd,
                    departure_diameter = dd, influence_area = ia)
            s = wb_make_state(T_w = WB_T_SAT, T_l = WB_T_SAT - 1.0)
            T_w, part = solve_wall_temperature(m, s, 3.0e4, h_c)
            total = part.q_c + part.q_q + part.q_e
            @test isfinite(T_w)
            @test total ≈ 3.0e4 rtol=1e-5
        end
    end
end
