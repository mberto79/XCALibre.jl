using XCALibre
using Test

# =============================================================================
#  Under-relaxation of the two vapour sources
# =============================================================================
#
#  `phase_change_relax` damps the bulk interfacial rate, `wall_boiling_relax`
#  the RPI wall rate. They are independent because the two sources are stiff for
#  different reasons: the bulk models respond to (T - T_sat) across the
#  interface, the wall model to the wall superheat through N_a ~ dT_sup^1.805.
#
#  The property that matters most is that relaxation must NOT bias the answer.
#  `relax_source!` blends against the previous step rather than scaling the rate,
#  so a converged solution is identical to the unrelaxed one - it only limits how
#  fast the source may change. The tests below pin that down.
# =============================================================================

mesh_file = joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv")
mesh = UNV2D_mesh(mesh_file, scale=1.0)
config = (hardware = Hardware(backend=CPU(), workgroup=AutoTune()),)

new_field(v) = (f = ScalarField(mesh); initialise!(f, v); f)

@testset "Phase change under-relaxation" begin

    @testset "relax = 1 is a no-op" begin
        field = new_field(7.0)
        prev = new_field(2.0)
        relax_source!(field, prev, 1.0, config)
        @test all(≈(7.0), field.values)
        # `prev` must still be updated, or the next relaxed step would blend
        # against a stale value.
        @test all(≈(7.0), prev.values)
    end

    @testset "blending, not scaling" begin
        field = new_field(10.0)
        prev = new_field(0.0)
        relax_source!(field, prev, 0.25, config)

        # 0.75*0 + 0.25*10 = 2.5. Scaling would also give 2.5 here, which is why
        # the steady-state test below is the one that actually distinguishes them.
        @test all(≈(2.5), field.values)
        @test all(≈(2.5), prev.values)
    end

    @testset "no steady-state bias" begin
        # THE point of blending. Hold the raw rate fixed and iterate: the relaxed
        # value must converge to the raw value, not to relax*raw. A scaling
        # implementation would sit at 0.3*4.0 = 1.2 forever and silently
        # evaporate less mass than the model asks for.
        raw = 4.0
        relax = 0.3
        prev = new_field(0.0)

        local field
        for _ in 1:200
            field = new_field(raw)
            relax_source!(field, prev, relax, config)
        end

        @test all(v -> isapprox(v, raw, rtol=1e-8), field.values)
        @test !isapprox(field.values[1], relax*raw, rtol=1e-3)   # not scaled
    end

    @testset "geometric approach to the new value" begin
        # After n steps at a fixed raw value the error is (1-relax)^n of the
        # initial gap - the standard first-order lag.
        raw, relax, start = 5.0, 0.5, 1.0
        prev = new_field(start)

        local field
        for _ in 1:4
            field = new_field(raw)
            relax_source!(field, prev, relax, config)
        end
        @test field.values[1] ≈ raw - (1 - relax)^4*(raw - start)
    end

    @testset "damps an oscillating source" begin
        # A stiff source that flips sign each step is exactly what relaxation is
        # for. Unrelaxed the field tracks it fully; relaxed the swing shrinks.
        prev_un = new_field(0.0)
        prev_rl = new_field(0.0)

        swing_un = 0.0
        swing_rl = 0.0
        for i in 1:20
            raw = iseven(i) ? 1.0 : -1.0

            f_un = new_field(raw); relax_source!(f_un, prev_un, 1.0, config)
            f_rl = new_field(raw); relax_source!(f_rl, prev_rl, 0.2, config)

            if i > 10                       # after the initial transient
                swing_un = max(swing_un, abs(f_un.values[1]))
                swing_rl = max(swing_rl, abs(f_rl.values[1]))
            end
        end

        @test swing_un ≈ 1.0
        @test swing_rl < 0.5*swing_un
    end

    @testset "negative rates (condensation) are relaxed the same way" begin
        # `mdot` is signed: positive evaporation, negative condensation. The
        # blend must be sign-agnostic.
        field = new_field(-8.0)
        prev = new_field(0.0)
        relax_source!(field, prev, 0.5, config)
        @test all(≈(-4.0), field.values)
    end

    @testset "factors are validated at setup" begin
        v = XCALibre.Solvers._validate_relax
        @test v(:phase_change_relax, 1.0) == 1.0
        @test v(:wall_boiling_relax, 0.5) == 0.5
        @test v(:phase_change_relax, 1) == 1.0            # Int accepted

        # Zero is VALID and means the source is switched off entirely: the blend
        # is against a `prev` that starts at zero, so it stays there. That makes
        # "disable" and "damp" a single keyword.
        @test v(:pressure_work_relax, 0.0) == 0.0
        @test v(:expansion_relax, 0) == 0.0

        @test_throws ArgumentError v(:phase_change_relax, -0.5)
        @test_throws ArgumentError v(:wall_boiling_relax, 1.5)
    end

    @testset "relax = 0 switches a source off permanently" begin
        # The thermo-acoustic loop between the pressure equation's expansion
        # source and the energy equation's pressure work is closed only when
        # BOTH are live, so being able to zero them is what breaks it.
        prev = new_field(0.0)
        local field
        for _ in 1:50
            field = new_field(1.0e8)      # a large raw source, every step
            relax_source!(field, prev, 0.0, config)
        end
        @test all(iszero, field.values)
        @test all(iszero, prev.values)
    end

    @testset "expansion relaxation defaults to off (1.0)" begin
        fluid = (physics_properties = (a = 1,),)
        @test XCALibre.Solvers.multiphase_expansion_relax(fluid) == 1.0

        fluid2 = (physics_properties = (expansion_relax = 0.0,),)
        @test XCALibre.Solvers.multiphase_expansion_relax(fluid2) == 0.0
        # Independent of the pressure-work factor.
        @test XCALibre.Solvers.multiphase_pressure_work_relax(fluid2) == 0.5
    end

    @testset "pressure-work relaxation is ON by default" begin
        # Unlike the phase-change factors, this one is damped out of the box:
        # `Dp/Dt` is the source whose raw value is routinely dominated by the
        # numerics of establishing a pressure field rather than by physics.
        fluid = (physics_properties = (a = 1,),)
        @test XCALibre.Solvers.multiphase_pressure_work_relax(fluid) == 0.5

        # ...and is user-overridable, including switching it off entirely.
        fluid_off = (physics_properties = (pressure_work_relax = 1.0,),)
        @test XCALibre.Solvers.multiphase_pressure_work_relax(fluid_off) == 1.0

        fluid_hard = (physics_properties = (pressure_work_relax = 0.05,),)
        @test XCALibre.Solvers.multiphase_pressure_work_relax(fluid_hard) == 0.05

        # Independent of the phase-change factors.
        @test XCALibre.Solvers.multiphase_phase_change_relax(fluid_hard) == 1.0
        @test XCALibre.Solvers.multiphase_wall_boiling_relax(fluid_hard) == 1.0

        # A slowly-varying dp/dt must still be reproduced: this is what makes a
        # non-unity default safe for the sealed-tank cases, where dp/dt is
        # essentially constant over a step.
        raw = 510.0                       # ~ the K-Site dp/dt, Pa/s
        prev = new_field(0.0)
        local field
        for _ in 1:10
            field = new_field(raw)
            relax_source!(field, prev, 0.5, config)
        end
        @test field.values[1] ≈ raw*(1 - 0.5^10) rtol=1e-10
        @test abs(field.values[1] - raw)/raw < 1e-3      # 0.1% after ten steps
    end

    @testset "accessors default to no relaxation" begin
        # Absent keywords must reproduce the previous behaviour exactly.
        fluid = (physics_properties = (a = 1,),)
        @test XCALibre.Solvers.multiphase_phase_change_relax(fluid) == 1.0
        @test XCALibre.Solvers.multiphase_wall_boiling_relax(fluid) == 1.0

        fluid2 = (physics_properties = (phase_change_relax = 0.4,
                                        wall_boiling_relax = 0.2),)
        @test XCALibre.Solvers.multiphase_phase_change_relax(fluid2) == 0.4
        @test XCALibre.Solvers.multiphase_wall_boiling_relax(fluid2) == 0.2

        # Independent: setting one must not move the other.
        fluid3 = (physics_properties = (wall_boiling_relax = 0.2,),)
        @test XCALibre.Solvers.multiphase_phase_change_relax(fluid3) == 1.0
        @test XCALibre.Solvers.multiphase_wall_boiling_relax(fluid3) == 0.2
    end
end
