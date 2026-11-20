using XCALibre
using Test
import XCALibre.Solvers: TimeStepControl, ConvectiveCo, CourantCriterion, TimeStepLimit,
    courant_rate!, max_time_step, target_courant, update_time_step!, time_step, cell_time_step,
    courant_numbers, update_dt!, set_time_step!

function ts_square_mesh(n=10)
    P, line = XCALibre.Point, XCALibre.line!
    pts = [P(0.0,0.0,0.0), P(1.0,0.0,0.0), P(0.0,1.0,0.0), P(1.0,1.0,0.0)]
    edges = [line(pts,1,2,n), line(pts,3,4,n), line(pts,1,3,n), line(pts,2,4,n)]
    patches = [XCALibre.Patch(:left,[3]), XCALibre.Patch(:right,[4]), XCALibre.Patch(:bottom,[1]),
        XCALibre.Patch(:top,[2])]
    XCALibre.UNV2.update_mesh_format(XCALibre.generate!(XCALibre.MeshBuilder2D(pts, edges,
        patches, [XCALibre.quad(edges,[1,2,3,4])])), Int64, Float64)
end

# criterion with a uniform rate and its own Courant target, and a fixed dt bound: the contract
struct UniformRate <: CourantCriterion
    rate::Float64
end
courant_rate!(rate, c::UniformRate, model, config) = (rate .= c.rate; nothing)
target_courant(::UniformRate, adaptive) = adaptive.maxAlphaCo

struct FixedBound <: TimeStepLimit
    dt::Float64
end
max_time_step(l::FixedBound, model, config) = l.dt

# mode: halves dt on even iterations
struct HalveEven <: XCALibre.Solve.AbstractTimeStepping end
update_dt!(ctrl, ::HalveEven, model, config, iteration) =
    iseven(iteration) && set_time_step!(ctrl, time_step(ctrl)/2, config)

# 10×10 unit square, Δx = 0.1, U = (2, 0, 0): convective rate |U|/Δx = 20 in every cell
function ts_setup(dt, adaptive)
    mesh = ts_square_mesh()
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    initialise!(model.momentum.U, [2.0, 0.0, 0.0])
    config = Configuration(solvers=(;), schemes=(;), boundaries=(;),
        hardware=Hardware(backend=CPU(), workgroup=1024),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=dt, adaptive=adaptive))
    return model, config
end

@testset "TimeStepControl" begin
    @testset "fixed dt" begin
        model, config = ts_setup(0.01, nothing)
        ctrl = TimeStepControl((ConvectiveCo(),), model, config)
        update_time_step!(ctrl, model, config, 2)
        @test time_step(ctrl) == 0.01 && cell_time_step(ctrl) == 0.01
        @test only(courant_numbers(ctrl)) ≈ 0.2
        @test config.runtime.dt[1] == 0.01
        quiet = TimeStepControl((ConvectiveCo(),), model, config; monitor=false)
        update_time_step!(quiet, model, config, 2)
        @test only(courant_numbers(quiet)) == 0 # not evaluated
    end

    @testset "adaptive, implicit: grow and shrink are clamped" begin
        model, config = ts_setup(0.01, AdaptiveTimeStepping(maxCo=1.0, minShrink=0.5, maxGrow=1.2))
        ctrl = TimeStepControl((ConvectiveCo(),), model, config)
        update_time_step!(ctrl, model, config, 2) # Co = 0.2: wants ×5, grows ×1.2
        @test time_step(ctrl) ≈ 0.012 && config.runtime.dt[1] ≈ 0.012
        initialise!(model.momentum.U, [200.0, 0.0, 0.0]) # Co = 24: wants ×1/24, shrinks ×0.5
        update_time_step!(ctrl, model, config, 2)
        @test time_step(ctrl) ≈ 0.006
    end

    @testset "adaptive, explicit: never above the stability limit" begin
        model, config = ts_setup(0.01, AdaptiveTimeStepping(maxCo=1.0, minShrink=0.5, maxGrow=1.2))
        initialise!(model.momentum.U, [200.0, 0.0, 0.0])
        ctrl = TimeStepControl((ConvectiveCo(),), model, config; explicit=true)
        update_time_step!(ctrl, model, config, 2)
        @test time_step(ctrl) ≈ 1.0 / 2000 # maxCo / λ, below minShrink × dt
    end

    @testset "several criteria: the most restrictive wins, each at its own target" begin
        model, config = ts_setup(0.01, AdaptiveTimeStepping(maxCo=1.0, maxAlphaCo=0.1, maxGrow=10.0))
        ctrl = TimeStepControl((ConvectiveCo(), UniformRate(5.0)), model, config)
        update_time_step!(ctrl, model, config, 2)
        @test all(courant_numbers(ctrl) .≈ (0.2, 0.05))
        @test time_step(ctrl) ≈ 0.02 # UniformRate: 0.1 / 5 (ConvectiveCo allows 0.05)
    end

    @testset "limits apply in every mode" begin
        model, config = ts_setup(0.01, nothing)
        ctrl = TimeStepControl((ConvectiveCo(),), model, config; limits=(FixedBound(0.004),))
        @test time_step(ctrl) == 0.004 && config.runtime.dt[1] == 0.004
        model, config = ts_setup(0.01, AdaptiveTimeStepping(maxCo=1.0, maxGrow=10.0))
        ctrl = TimeStepControl((ConvectiveCo(),), model, config; limits=(FixedBound(0.03),))
        update_time_step!(ctrl, model, config, 2)
        @test time_step(ctrl) ≈ 0.03
    end

    @testset "user-defined mode" begin
        model, config = ts_setup(0.01, HalveEven())
        ctrl = TimeStepControl((ConvectiveCo(),), model, config)
        update_time_step!(ctrl, model, config, 3)
        @test time_step(ctrl) == 0.01
        update_time_step!(ctrl, model, config, 4)
        @test time_step(ctrl) == 0.005 && config.runtime.dt[1] == 0.005
        @test only(courant_numbers(ctrl)) ≈ 0.2 # at the dt before the update
    end

    @testset "local time stepping" begin
        lts = LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=4)
        model, config = ts_setup(0.01, lts)
        @test_throws ArgumentError TimeStepControl((ConvectiveCo(),), model, config)
        @test_throws ArgumentError TimeStepControl((ConvectiveCo(), UniformRate(1.0)), model, config; explicit=true)
        ctrl = TimeStepControl((ConvectiveCo(),), model, config; explicit=true)
        update_time_step!(ctrl, model, config, 3) # Co ramped to 0.1 + 0.4·2/4 = 0.3
        @test all(cell_time_step(ctrl) .≈ 0.3 / 20) && time_step(ctrl) ≈ 0.3 / 20
        @test config.runtime.dt[1] ≈ 0.3 / 20
    end

    @testset "update does not allocate per cell" begin
        model, config = ts_setup(0.01, AdaptiveTimeStepping())
        ctrl = TimeStepControl((ConvectiveCo(),), model, config)
        update_time_step!(ctrl, model, config, 2)
        @test (@inferred update_time_step!(ctrl, model, config, 2)) === ctrl
        @test (@allocated update_time_step!(ctrl, model, config, 2)) < 2_048
    end
end
