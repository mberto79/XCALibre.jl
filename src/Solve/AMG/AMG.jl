# reference/ = current production AMG impl (CPU track + existing GPU path), kept intact as
# the benchmark/regression oracle. device/ = new greenfield GPU pipeline (Decision 0),
# inert until Phase 5d. See AMG_plan.md.
include("reference/0_AMG_types.jl")
include("reference/1_AMG_setup.jl")
include("reference/2_AMG_coarsening.jl")
include("reference/3_AMG_transfer.jl")
include("reference/4_AMG_smoothers.jl")
include("reference/5_AMG_cycle.jl")
include("reference/6_AMG_cg.jl")
include("reference/7_AMG_update.jl")

include("device/0_AMG_device.jl")
include("device/1_AMG_macro.jl")
include("device/2_AMG_fused.jl")
include("device/3_AMG_fused_cycle.jl")
include("device/4_AMG_ml_cycle.jl")
include("device/5_AMG_refresh.jl")
include("device/6_AMG_fused_zone.jl")
include("device/7_AMG_integration.jl")
