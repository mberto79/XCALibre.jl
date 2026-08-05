using XCALibre
using Test

const MP = XCALibre.ModelPhysics

# =============================================================================
#  Tabulated real-fluid properties
# =============================================================================
#
#  The tables are the only way the real equation of state reaches the solver, so
#  what matters is (a) that they reproduce the EOS they came from, and (b) that
#  a lookup outside the tabulated range degrades in a known way rather than
#  extrapolating into nonsense.
# =============================================================================

@testset "Tabulated real-fluid properties" begin

    @testset "PropertyGrid" begin
        g = PropertyGrid(p_min=1.0e5, p_max=1.0e6, np=10, T_min=20.0, T_max=40.0, nT=21)
        @test g.p_min == 1.0e5
        @test g.dp ≈ (1.0e6 - 1.0e5)/9
        @test g.np == 10
        @test g.T_min == 20.0
        @test g.dT ≈ 1.0
        @test g.nT == 21

        # The extents must come back out. This is what the shadowed positional
        # constructor got wrong: `p_max` was taken as the spacing, giving a grid
        # that ran to 9x the requested pressure while looking entirely plausible.
        @test MP.grid_p_max(g) ≈ 1.0e6
        @test MP.grid_T_max(g) ≈ 40.0

        @test_throws ArgumentError PropertyGrid(
            p_min=1.0e5, p_max=1.0e6, np=1, T_min=20.0, T_max=40.0, nT=21)
        @test_throws ArgumentError PropertyGrid(
            p_min=1.0e6, p_max=1.0e5, np=10, T_min=20.0, T_max=40.0, nT=21)
        @test_throws ArgumentError PropertyGrid(
            p_min=1.0e5, p_max=1.0e6, np=10, T_min=40.0, T_max=20.0, nT=21)
    end

    @testset "Bilinear lookup" begin
        # A table holding f(p, T) = p + 2T exactly; bilinear interpolation is
        # exact for anything linear, so any error here is an indexing error.
        g = PropertyGrid(p_min=0.0, p_max=10.0, np=11, T_min=0.0, T_max=20.0, nT=11)
        vals = [p + 2T for T in MP.grid_temperatures(g), p in MP.grid_pressures(g)]
        tbl = PropertyTable(g, vals)

        # Exact at nodes.
        @test table_lookup(tbl, 0.0, 0.0) ≈ 0.0
        @test table_lookup(tbl, 10.0, 20.0) ≈ 50.0
        @test table_lookup(tbl, 4.0, 6.0) ≈ 16.0

        # Exact between nodes, in both directions and diagonally.
        @test table_lookup(tbl, 2.5, 0.0) ≈ 2.5
        @test table_lookup(tbl, 0.0, 3.0) ≈ 6.0
        @test table_lookup(tbl, 2.5, 3.0) ≈ 8.5

        # Outside the grid the value FREEZES at the boundary rather than
        # extrapolating - the deliberate choice, since an extrapolated Helmholtz
        # EOS is confidently wrong.
        @test table_lookup(tbl, -100.0, 0.0) ≈ 0.0
        @test table_lookup(tbl, 1.0e6, 20.0) ≈ 50.0
        @test table_lookup(tbl, 5.0, -50.0) ≈ 5.0
        @test table_lookup(tbl, 5.0, 1.0e6) ≈ 45.0
    end

    # -------------------------------------------------------------------------
    # Small tables over the Tatsumoto operating envelope. Kept coarse so the
    # test stays quick; the accuracy check below accounts for that.
    p_range = (0.35e6, 1.15e6)
    T_range = (24.0, 33.0)
    lh2 = RealFluid(H2(), :liquid, p=p_range, T=T_range, np=25, nT=25, verbose=false)
    gh2 = RealFluid(H2(), :vapour, p=p_range, T=T_range, np=25, nT=25, verbose=false)

    @testset "RealFluid produces the five property models" begin
        @test lh2.rho isa TabulatedEos
        @test lh2.mu isa TabulatedMu
        @test lh2.k isa TabulatedK
        @test lh2.cp isa TabulatedCp
        @test lh2.beta isa TabulatedBeta

        @test_throws ArgumentError RealFluid(H2(), :solid, p=p_range, T=T_range)
    end

    @testset "Specific gas constant" begin
        # R_u/M for hydrogen. Carried so the Lee/Schrage kinetic prefactor works
        # with a real-fluid vapour, which the previous ideal-gas-only check
        # rejected outright.
        @test specific_gas_constant(gh2.rho) ≈ 8.314472/2.01588e-3 rtol=1e-6
        @test specific_gas_constant(IdealGas(M=2.01588e-3)) ≈ 8.314462618/2.01588e-3 rtol=1e-6
        @test specific_gas_constant(ConstEos(rho=70.8)) === nothing
    end

    @testset "Tables reproduce the equation of state" begin
        constants = MP.helmholtz_constants(H2(), Float64)
        fluid = H2()

        # Compare against a direct EOS evaluation at saturation, where both
        # branches are unambiguous.
        for p in (0.4e6, 0.7e6, 1.1e6)
            T_sat = MP.find_saturation_temperature(p, constants, fluid)
            (_, _, rho_l_mol, rho_v_mol) = MP.find_saturation_properties(
                T_sat, p, constants, fluid)

            rho_l_exact = rho_l_mol*constants.M
            rho_v_exact = rho_v_mol*constants.M

            @test table_lookup(lh2.rho.rho, p, T_sat) ≈ rho_l_exact rtol=0.02
            @test table_lookup(gh2.rho.rho, p, T_sat) ≈ rho_v_exact rtol=0.05

            tau = constants.T_c/T_sat
            cp_l_exact = MP.c_p(rho_l_mol/constants.rho_c, tau, constants, fluid)/constants.M
            @test table_lookup(lh2.cp.cp, p, T_sat) ≈ cp_l_exact rtol=0.10

            # psi = (1/rho)(drho/dp)|_T, the coefficient the pressure equation
            # carries. Must be positive and finite everywhere.
            psi_v = table_lookup(gh2.rho.psi, p, T_sat)
            @test psi_v > 0
            @test isfinite(psi_v)
        end
    end

    @testset "The vapour is materially non-ideal" begin
        # The whole reason for this machinery. If the real vapour were within a
        # few percent of ideal, `IdealGas` would do and the tables would be
        # unnecessary complexity - so assert that it is not.
        for (p, min_ratio) in ((0.4e6, 1.2), (0.7e6, 1.5), (1.1e6, 2.5))
            T_sat = saturation_temperature(
                build_saturation_curve(H2(), p=p_range, T=(24.0, 33.0),
                                       np=41, nT=41, verbose=false), p)
            psi_real = table_lookup(gh2.rho.psi, p, T_sat)
            psi_ideal = 1/p
            @test psi_real/psi_ideal > min_ratio
        end
    end

    @testset "Physically ordered properties" begin
        for p in (0.4e6, 0.7e6, 1.1e6), T in (25.0, 28.0, 31.0)
            rho_l = table_lookup(lh2.rho.rho, p, T)
            rho_v = table_lookup(gh2.rho.rho, p, T)
            @test rho_l > rho_v > 0                     # liquid is the dense branch
            @test table_lookup(lh2.cp.cp, p, T) > 0
            @test table_lookup(lh2.k.k, p, T) > 0
            @test table_lookup(lh2.mu.mu, p, T) > 0
            @test table_lookup(lh2.beta.beta, p, T) > 0 # hydrogen expands on heating
            @test table_lookup(lh2.mu.mu, p, T) > table_lookup(gh2.mu.mu, p, T)
        end
    end

    @testset "SaturationCurve" begin
        sat = build_saturation_curve(H2(), p=p_range, T=(22.0, 33.0),
                                     np=101, nT=101, verbose=false)

        # Paper's stated saturation temperatures (Results and discussion).
        @test saturation_temperature(sat, 0.4e6) ≈ 26.0 atol=0.3
        @test saturation_temperature(sat, 0.7e6) ≈ 29.0 atol=0.3
        @test saturation_temperature(sat, 1.1e6) ≈ 31.9 atol=0.3

        # Round trip: p_sat(T_sat(p)) must return p.
        for p in (0.4e6, 0.7e6, 1.0e6)
            @test saturation_pressure(sat, saturation_temperature(sat, p)) ≈ p rtol=0.02
        end

        # Monotonic in pressure.
        @test saturation_temperature(sat, 0.4e6) <
              saturation_temperature(sat, 0.7e6) <
              saturation_temperature(sat, 1.1e6)

        # Latent heat collapses towards the critical point - the reason a scalar
        # h_fg is not usable across the paper's pressure sweep.
        L_low = latent_heat(sat, 0.4e6, 0.0)
        L_high = latent_heat(sat, 1.1e6, 0.0)
        @test L_low > L_high > 0
        @test L_high < 0.7*L_low

        # `Antoine` carries no latent heat and must pass the reference through
        # unchanged, which is what keeps the bulk source and the energy sink
        # using one and the same value.
        @test latent_heat(Antoine(), 0.7e6, 446.0e3) == 446.0e3
    end

    @testset "Antoine still agrees with the EOS where it is valid" begin
        # The Antoine fit is stated valid over 21.01-32.27 K, which brackets the
        # paper's operating range; a large disagreement here would mean one of
        # the two saturation paths is wrong.
        sat_tab = build_saturation_curve(H2(), p=p_range, T=(22.0, 33.0),
                                         np=101, nT=101, verbose=false)
        ant = Antoine()
        for p in (0.4e6, 0.7e6, 1.1e6)
            @test saturation_temperature(ant, p) ≈
                  saturation_temperature(sat_tab, p) atol=0.6
        end
    end

    @testset "Pressure-locked tables (p_ref)" begin
        p_ref = 0.7e6
        lk = RealFluid(H2(), :liquid, p=p_range, T=T_range,
                       np=25, nT=25, p_ref=p_ref, verbose=false)

        # Properties become functions of temperature alone: the same T at any
        # pressure must give the same value.
        for T in (25.0, 28.0, 31.0)
            ref = table_lookup(lk.rho.rho, p_ref, T)
            for p in (0.35e6, 0.5e6, 0.9e6, 1.15e6)
                @test table_lookup(lk.rho.rho, p, T) ≈ ref
                @test table_lookup(lk.cp.cp, p, T) ≈ table_lookup(lk.cp.cp, p_ref, T)
                @test table_lookup(lk.mu.mu, p, T) ≈ table_lookup(lk.mu.mu, p_ref, T)
            end
            # ...and equal to the unlocked table evaluated at p_ref, to within
            # that table's own interpolation error. The locked table is in fact
            # slightly MORE accurate here: it evaluates the EOS at p_ref exactly,
            # whereas the unlocked one interpolates between its pressure nodes.
            @test ref ≈ table_lookup(lh2.rho.rho, p_ref, T) rtol=1e-3
        end

        # Temperature dependence is retained - that is the whole point of
        # locking pressure rather than going constant-density.
        @test table_lookup(lk.rho.rho, p_ref, 25.0) != table_lookup(lk.rho.rho, p_ref, 31.0)
        @test table_lookup(lk.beta.beta, p_ref, 25.0) > 0

        # psi must be EXACTLY zero. A density that does not depend on pressure
        # has no compressibility to report, and reporting the real value while
        # holding rho fixed would leave the pressure equation carrying a
        # compressibility the density does not have.
        for p in (0.35e6, 0.7e6, 1.15e6), T in (25.0, 30.0)
            @test phase_compressibility(lk.rho, p, T) == 0.0
        end
        @test phase_compressibility(lh2.rho, 0.7e6, 29.0) > 0    # unlocked is not

        # The specific gas constant survives, so Lee/Schrage still work.
        @test specific_gas_constant(lk.rho) ≈ specific_gas_constant(lh2.rho)

        @test_throws ArgumentError RealFluid(H2(), :liquid, p=p_range, T=T_range,
                                             np=9, nT=9, p_ref=5.0e6, verbose=false)
    end

    @testset "Phase built from RealFluid" begin
        ph = Phase(lh2)
        @test ph.rho isa TabulatedEos
        @test ph.mu isa TabulatedMu
        @test ph.k isa TabulatedK
        @test ph.cp isa TabulatedCp
        @test ph.beta isa TabulatedBeta
    end

    @testset "Table range diagnostics" begin
        mesh_file = joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv")
        if isfile(mesh_file)
            mesh = UNV2D_mesh(mesh_file, scale=1.0)
            p_abs = ScalarField(mesh); initialise!(p_abs, 0.7e6)
            T = ScalarField(mesh); initialise!(T, 29.0)

            inside = table_range_report(lh2.rho, p_abs, T)
            @test inside.outside == 0
            @test inside.fraction == 0.0

            initialise!(T, 100.0)               # far above the table
            outside = table_range_report(lh2.rho, p_abs, T)
            @test outside.outside == outside.total
            @test outside.fraction == 1.0
        else
            @info "skipping table_range_report test: quad40.unv not found"
        end
    end
end
