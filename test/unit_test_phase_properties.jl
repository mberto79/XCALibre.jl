# Unit tests for the optional thermal properties on `Phase` (k, cp, beta).
#
# These back step 2 of the LH2 tank work: PhaseState carried k/cp/beta fields
# that nothing populated and Phase() would not accept. The tests below pin the
# behaviour that was added:
#   - backwards compatibility: Phase(rho, mu) still works, thermal props nothing
#   - float promotion to the Const* model types
#   - constant models collapse to ConstantScalar (no per-cell storage)
#   - a missing property raises a descriptive error rather than reading as zero
#   - PhaseState stays isbits-adaptable (GPU dispatch safety)

using XCALibre
using Test

using XCALibre.ModelPhysics: build_phase, update_phase_property!

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)

@testset "Phase: backwards compatibility (no thermal properties)" begin
    phase = Phase(rho=1000.0, mu=1.0e-3)

    @test phase.rho isa ConstEos
    @test phase.mu isa ConstMu
    @test phase.k === nothing
    @test phase.cp === nothing
    @test phase.beta === nothing

    state = build_phase(phase, mesh)

    # rho/mu still behave exactly as the solver expects: scalar indexing
    @test state.rho[1] == 1000.0
    @test state.mu[1] == 1.0e-3

    # unsupplied thermal properties are absent, not silently zero
    @test state.k === nothing
    @test state.cp === nothing
    @test state.beta === nothing
end

@testset "Phase: float promotion to Const* models" begin
    phase = Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164)

    @test phase.k isa ConstK
    @test phase.cp isa ConstCp
    @test phase.beta isa ConstBeta

    @test phase.k.k == 0.100
    @test phase.cp.cp == 9660.0
    @test phase.beta.beta == 0.0164
end

@testset "Phase: constant models collapse to ConstantScalar" begin
    phase = Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164)
    state = build_phase(phase, mesh)

    # ConstantScalar stores one number and returns it for any index, so a
    # constant property costs no per-cell memory but still indexes in kernels.
    for (field, value) in ((state.k, 0.100), (state.cp, 9660.0), (state.beta, 0.0164))
        @test field isa XCALibre.Fields.ConstantScalar
        @test field[1] == value
        @test field[12345] == value      # index-independent by construction
    end
end

@testset "Phase: model objects are retained on PhaseState" begin
    phase = Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0)
    state = build_phase(phase, mesh)

    @test state.k_model isa ConstK
    @test state.cp_model isa ConstCp
    @test state.beta_model === nothing
end

@testset "Phase: update_phase_property! is a safe no-op for constants" begin
    # A missing property is reported by `_assert_phase_thermal_properties` at
    # solver setup; that behaviour is covered in unit_test_twophase_energy.jl.
    state = build_phase(
        Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164), mesh)

    T = ScalarField(mesh); initialise!(T, 20.3)
    p = ScalarField(mesh); initialise!(p, 117.2e3)

    backend = CPU(); workgroup = AutoTune()
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = (hardware=hardware,)

    # constant models fall through to the no-op and leave values untouched
    for (field, model) in ((state.rho, state.rho_model), (state.k, state.k_model),
                           (state.cp, state.cp_model), (state.beta, state.beta_model))
        @test update_phase_property!(field, model, p, T, config) === nothing
    end
    @test state.k[1] == 0.100
    @test state.cp[1] == 9660.0

    # and it is equally safe when the property was never supplied (`nothing`)
    bare = build_phase(Phase(rho=1000.0, mu=1.0e-3), mesh)
    @test update_phase_property!(bare.cp, bare.cp_model, p, T, config) === nothing
end

@testset "Phase: PhaseState survives adapt (GPU dispatch safety)" begin
    # A non-isbits property model would break kernel dispatch on GPU. Adapting
    # to the CPU backend exercises the same code path used for device transfer.
    state = build_phase(
        Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164), mesh)
    adapted = adapt(CPU(), state)

    @test adapted.k[1] == 0.100
    @test adapted.cp[1] == 9660.0
    @test adapted.beta[1] == 0.0164
    @test isbits(adapted.k_model)
    @test isbits(adapted.cp_model)
end
