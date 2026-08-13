# Tests for the three interfacial phase change models — step 6 of the LH2 tank
# work. Formulations from Fernandes et al. (2026) Eqs. 9–15.
#
# The model-level tests are exact: at equilibrium every model must return
# identically zero, and the sign of the flux must follow the superheat. Those
# catch formulation and sign errors without needing a solver run.

using XCALibre
using Test

using XCALibre.ModelPhysics: interfacial_mass_flux, _kinetic_prefactor

M_H2 = 2.01588e-3
R_H2 = XCALibre.ModelPhysics.R_UNIVERSAL/M_H2      # ~4124.2 J/kg/K
L_H2 = 446.0e3                                     # [J/kg]
RHO_L = 70.8
RHO_V = 1.34
SAT = Antoine()                                    # paper Eq. (15), hydrogen

@testset "Antoine saturation curve" begin
    # Round-trip: T -> p -> T must be exact to floating point
    for T in (21.0, 22.5, 25.0, 28.0, 32.0)
        p = saturation_pressure(SAT, T)
        @test saturation_temperature(SAT, p) ≈ T rtol=1e-12
    end
    for p in (0.8e5, 1.03e5, 1.5e5, 2.0e5, 3.0e5)
        T = saturation_temperature(SAT, p)
        @test saturation_pressure(SAT, T) ≈ p rtol=1e-12
    end

    # The K-Site operating point: 103 kPa -> ~20.43 K (the value the case uses)
    @test saturation_temperature(SAT, 103.0e3) ≈ 20.43 atol=0.01

    # Monotonic increasing, as a saturation curve must be
    ps = [saturation_pressure(SAT, T) for T in 21.0:0.5:32.0]
    @test all(diff(ps) .> 0)

    # Paper's stated validity window is carried for reference
    @test SAT.Tmin == 21.01
    @test SAT.Tmax == 32.27
end

@testset "Model construction and defaults" begin
    @test Schrage().sigma == 1.0e-3               # paper baseline
    @test Lee().sigma == 1.0e-6                   # paper baseline
    @test ModifiedEnergyJump(h=1.0).h == 1.0      # paper baseline
    @test Schrage(sigma=1e-5).sigma == 1e-5

    for m in (Schrage(), Lee(), ModifiedEnergyJump(h=1.0), SAT)
        @test isbits(m)                           # GPU dispatch safety
    end
    @test Schrage() isa AbstractPhaseChangeModel
    @test SAT isa AbstractSaturationModel
end

@testset "Kinetic prefactor equals sqrt(M/(2 pi R_u T))" begin
    # sqrt(1/(2 pi R_sp T)) must equal sqrt(M/(2 pi R_u T)) since R_sp = R_u/M
    T = 20.43
    Ru = XCALibre.ModelPhysics.R_UNIVERSAL
    @test _kinetic_prefactor(T, R_H2) ≈ sqrt(M_H2/(2*pi*Ru*T)) rtol=1e-12
end

@testset "Equilibrium gives exactly zero mass flux" begin
    # At T = T_sat(p) every model must return zero: no superheat, no phase change.
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    alpha = 0.5

    for pc in (Schrage(sigma=1e-3), Lee(sigma=1e-6), ModifiedEnergyJump(h=10.0))
        flux = interfacial_mass_flux(pc, alpha, T_sat, p, T_sat,
                                     RHO_L, RHO_V, SAT, L_H2, R_H2)
        @test abs(flux) < 1e-9
    end
end

@testset "Sign convention: superheat evaporates, subcool condenses" begin
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    alpha = 0.5

    for pc in (Schrage(sigma=1e-3), Lee(sigma=1e-6), ModifiedEnergyJump(h=10.0))
        hot  = interfacial_mass_flux(pc, alpha, T_sat + 0.5, p, T_sat,
                                     RHO_L, RHO_V, SAT, L_H2, R_H2)
        cold = interfacial_mass_flux(pc, alpha, T_sat - 0.5, p, T_sat,
                                     RHO_L, RHO_V, SAT, L_H2, R_H2)
        @test hot > 0     # evaporation
        @test cold < 0    # condensation
    end
end

@testset "MeJ is exactly h*(T - T_sat)/L" begin
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    dT = 0.3
    for h in (1.0, 10.0, 100.0)          # paper's three values
        pc = ModifiedEnergyJump(h=h)
        flux = interfacial_mass_flux(pc, 0.5, T_sat + dT, p, T_sat,
                                     RHO_L, RHO_V, SAT, L_H2, R_H2)
        @test flux ≈ h*dT/L_H2 rtol=1e-12
    end

    # exactly linear in h — the paper's sensitivity finding
    f1  = interfacial_mass_flux(ModifiedEnergyJump(h=1.0), 0.5, T_sat+dT, p, T_sat,
                                RHO_L, RHO_V, SAT, L_H2, R_H2)
    f10 = interfacial_mass_flux(ModifiedEnergyJump(h=10.0), 0.5, T_sat+dT, p, T_sat,
                                RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test f10 ≈ 10*f1 rtol=1e-12
end

@testset "Schrage is exactly the Eq. 14 expression" begin
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    T = T_sat + 0.4
    s = 1.0e-3
    expected = (2*s/(2 - s))*sqrt(1/(2*pi*R_H2*T_sat))*(saturation_pressure(SAT, T) - p)

    flux = interfacial_mass_flux(Schrage(sigma=s), 0.5, T, p, T_sat,
                                 RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test flux ≈ expected rtol=1e-12

    # For sigma << 1 the prefactor 2s/(2-s) -> s, so the flux is near-linear in
    # sigma across the paper's range.
    f3 = interfacial_mass_flux(Schrage(sigma=1e-3), 0.5, T, p, T_sat,
                               RHO_L, RHO_V, SAT, L_H2, R_H2)
    f5 = interfacial_mass_flux(Schrage(sigma=1e-5), 0.5, T, p, T_sat,
                               RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test f3/f5 ≈ 100.0 rtol=1e-3
end

@testset "Lee uses the sigma-derived beta and switches phase weighting" begin
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    s = 1.0e-6
    alpha = 0.4
    beta = s*sqrt(1/(2*pi*R_H2*T_sat))*L_H2*RHO_L/(RHO_L - RHO_V)

    # evaporation branch weights by alpha_l * rho_l
    dT = 0.3
    got = interfacial_mass_flux(Lee(sigma=s), alpha, T_sat + dT, p, T_sat,
                                RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test got ≈ beta*alpha*RHO_L*(dT/T_sat) rtol=1e-12

    # condensation branch weights by alpha_v * rho_v
    got_c = interfacial_mass_flux(Lee(sigma=s), alpha, T_sat - dT, p, T_sat,
                                  RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test got_c ≈ beta*(1 - alpha)*RHO_V*(-dT/T_sat) rtol=1e-12

    # exactly linear in sigma
    f6 = interfacial_mass_flux(Lee(sigma=1e-6), alpha, T_sat+dT, p, T_sat,
                               RHO_L, RHO_V, SAT, L_H2, R_H2)
    f8 = interfacial_mass_flux(Lee(sigma=1e-8), alpha, T_sat+dT, p, T_sat,
                               RHO_L, RHO_V, SAT, L_H2, R_H2)
    @test f6 ≈ 100*f8 rtol=1e-12
end

@testset "Relative magnitudes match the paper's findings" begin
    # The paper reports Lee's boil-off an order of magnitude below Schrage/MeJ at
    # their baseline coefficients (Figs. 9c, 9f). Check the same ordering holds
    # here, which is a sanity check that the coefficient scalings are right.
    p = 103.0e3
    T_sat = saturation_temperature(SAT, p)
    T = T_sat + 0.5

    f_schrage = interfacial_mass_flux(Schrage(sigma=1e-3), 0.5, T, p, T_sat,
                                      RHO_L, RHO_V, SAT, L_H2, R_H2)
    f_lee     = interfacial_mass_flux(Lee(sigma=1e-6), 0.5, T, p, T_sat,
                                      RHO_L, RHO_V, SAT, L_H2, R_H2)
    @info "baseline fluxes [kg/m^2/s]" f_schrage f_lee ratio=f_schrage/f_lee

    @test f_schrage > 0 && f_lee > 0
    @test f_schrage > f_lee      # Lee under-predicts, as the paper reports
end

# -----------------------------------------------------------------------------
# Solver-level: the sources are wired and mass is conserved
# -----------------------------------------------------------------------------

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)
backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
mesh_dev = adapt(backend, mesh)
noSlip = [0.0, 0.0, 0.0]
P0 = 103.0e3

function build_pc(pc; iterations, dt, superheat=0.0)
    T_sat = saturation_temperature(SAT, P0)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(rho=RHO_L, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164),
                      Phase(rho=IdealGas(M=M_H2), mu=1.11e-6, k=0.0169, cp=12200.0)),
            phase_change = pc,
            saturation = SAT,
            h_fg = L_H2,
            p_operating = P0,
            gravity = Gravity([0.0, 0.0, 0.0])
        ),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T_sat),
        domain = mesh_dev
    )
    BCs = assign(region=mesh_dev, (
        U = [Wall(:inlet, noSlip), Wall(:outlet, noSlip),
             Wall(:bottom, noSlip), Wall(:top, noSlip)],
        p_rgh = [Zerogradient(:inlet), Zerogradient(:outlet),
                 Zerogradient(:bottom), Zerogradient(:top)],
        alpha = [Zerogradient(:inlet), Zerogradient(:outlet),
                 Zerogradient(:bottom), Zerogradient(:top)],
        T = [Zerogradient(:inlet), Zerogradient(:outlet),
             Zerogradient(:bottom), Zerogradient(:top)]))

    s = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss)
    b() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                      convergence=1e-10, relax=1.0, rtol=0.0, atol=1e-14)
    config = Configuration(
        solvers=(U=b(), p_rgh=b(), alpha=b(), T=b()),
        schemes=(U=s, p=s, p_rgh=s, alpha=s, T=s),
        runtime=Runtime(iterations=iterations, time_step=dt, write_interval=-1),
        hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, 0.0)
    setField_Box!(mesh=mesh, field=model.fluid.alpha, value=1.0,
                  min_corner=[0.0, 0.0, -0.5], max_corner=[1.0, 0.5, 0.5])
    initialise!(model.energy.T, T_sat + superheat)
    return model, config
end

total_mass(m) = sum(m.fluid.rho.values[i]*m.domain.cells[i].volume
                    for i in eachindex(m.domain.cells))

@testset "Solver: all three models run and conserve mass" begin
    for pc in (Schrage(sigma=1e-3), Lee(sigma=1e-6), ModifiedEnergyJump(h=10.0))
        model, cfg1 = build_pc(pc, iterations=1, dt=1e-3, superheat=0.2)
        run!(model, cfg1)
        m0 = total_mass(model)

        _, cfgN = build_pc(pc, iterations=50, dt=1e-3, superheat=0.2)
        run!(model, cfgN)
        m1 = total_mass(model)

        @info "phase change mass conservation" model=typeof(pc).name.wrapper m0 m1 drift=abs(m1-m0)/m0
        # Sealed tank: phase change moves mass between phases, it does not create
        # it, so the mixture mass must hold.
        @test abs(m1 - m0)/m0 < 1e-4
        @test all(0.0 .<= model.fluid.alpha.values .<= 1.0)
        @test all(isfinite, model.energy.T.values)
        @test all(isfinite, model.momentum.p.values)
    end
end

@testset "Solver: phase change requires its prerequisites" begin
    # Schrage/Lee need the vapour specific gas constant, so a constant-density
    # vapour must be refused with a message naming the reason.
    bad = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(rho=RHO_L, mu=13.2e-6, k=0.1, cp=9660.0),
                      Phase(rho=1.34, mu=1.11e-6, k=0.0169, cp=12200.0)),
            phase_change = Schrage(), saturation = SAT, h_fg = L_H2,
            p_operating = P0, gravity = Gravity([0.0, 0.0, 0.0])),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=20.43),
        domain = mesh_dev)

    _, cfg = build_pc(Schrage(), iterations=1, dt=1e-3)
    err = try; run!(bad, cfg); nothing; catch e; e; end
    @test err isa ArgumentError
    # constant-density vapour trips the compressibility requirement first
    @test occursin("compressible", err.msg) || occursin("IdealGas", err.msg)
end

# =============================================================================
#  Interfacial area density
# =============================================================================
#
#  The phase change models return a mass flux PER UNIT INTERFACE AREA, so the
#  closure that turns it into a volumetric rate decides how the source responds
#  to the shape of the `alpha` field - not merely its magnitude.
# =============================================================================

@testset "Interfacial area density" begin
    D_B = 0.5e-3                # dispersed bubble diameter [m]
    dispersed = DispersedBubbles(diameter = D_B)
    resolved  = ResolvedInterface()

    @testset "resolved interface is unchanged" begin
        # VOF behaviour must be bit-identical - this is the pre-existing path.
        for gm in (0.0, 1.0, 1234.0, 1e9)
            @test interfacial_area_density(resolved, 0.5, gm) == gm
        end
    end

    @testset "dispersed closure is 6 a(1-a)/d" begin
        for a in (0.1, 0.5, 0.9, 0.998)
            @test interfacial_area_density(dispersed, a, 999.0) ≈ 6*a*(1 - a)/D_B
        end

        # Symmetric under phase inversion, so it does not matter which phase
        # `alpha` tracks.
        @test interfacial_area_density(dispersed, 0.3, 0.0) ≈
              interfacial_area_density(dispersed, 0.7, 0.0)

        # Zero where there is no dispersed phase to have an interface with.
        @test interfacial_area_density(dispersed, 0.0, 1e9) == 0.0
        @test interfacial_area_density(dispersed, 1.0, 1e9) == 0.0

        # Bounded by the value at alpha = 0.5, unlike |grad alpha|.
        peak = 6*0.25/D_B
        for a in range(0.0, 1.0, length = 51)
            @test interfacial_area_density(dispersed, a, 1e9) <= peak + 1e-9
        end

        @test_throws ArgumentError DispersedBubbles(diameter = 0.0)
        @test_throws ArgumentError DispersedBubbles(diameter = -1.0)
    end

    @testset "dispersed closure cannot amplify a checkerboard" begin
        # THE POINT OF THE CHANGE. A 2*dx oscillation in `alpha` is the field
        # that MAXIMISES |grad alpha| for a given amplitude, so with the resolved
        # closure the phase change source grows with exactly the noise it should
        # be indifferent to - a feedback loop with a checkerboard eigenmode.
        #
        # Values are the LH2 pipe: 86 um wall cells, alpha_liquid ~ 0.998.
        h_cell = 8.585e-5
        a_bulk = 0.998

        base = interfacial_area_density(dispersed, a_bulk, 0.0)
        for amplitude in (0.0, 0.01, 0.05, 0.1)
            gm = amplitude/h_cell               # |grad alpha| of that oscillation
            @test interfacial_area_density(dispersed, a_bulk, gm) == base
            # ... while the resolved closure tracks it one-for-one.
            @test interfacial_area_density(resolved, a_bulk, gm) == gm
        end

        # Quantify the gap that motivated the change: at amplitude 0.1 the
        # resolved closure is ~49x the physical dispersed value.
        gm_cb = 0.1/h_cell
        @test interfacial_area_density(resolved, a_bulk, gm_cb)/base > 40

        # And the resolved closure is LARGEST next to pure liquid, where there is
        # least interface, and zero in a uniformly bubbly region, where there is
        # most. The dispersed closure has neither pathology.
        @test interfacial_area_density(resolved, 1.0, gm_cb) > 0
        @test interfacial_area_density(dispersed, 1.0, gm_cb) == 0.0
        @test interfacial_area_density(dispersed, 0.9, 0.0) > 0
    end
end

