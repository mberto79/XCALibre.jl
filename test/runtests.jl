using XCALibre
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using StaticArrays 
using Statistics
using Random
using Interpolations: linear_interpolation
using FFTW
using Test

workgroupsize(mesh) = length(mesh.cells) ÷ Threads.nthreads()

TEST_CASES_DIR = pkgdir(XCALibre, "test", "cases")

# NEW SECTION: test registry (tier => ordered files relative to test/)
const TIERS = [
    "unit" => [
        "unit/Mesh/test_mesh_conversion.jl",
        "unit/Mesh/test_foam_padded_neighbour.jl",
        "unit/Mesh/test_mesh_reorder.jl",
        "unit/Mesh/test_face_orientation.jl",
        "unit/Mesh/test_bounding_box.jl",
        "unit/IOFormats/test_foam_writer_unset_bcs.jl",
        "unit/Multithread/test_kernel_index_pattern.jl",
        "unit/Preprocess/test_setFields.jl",
        "unit/Calculate/test_wall_distance.jl",
        "unit/Calculate/test_least_squares.jl",
        "unit/Calculate/test_venkat_limiter.jl",
        "unit/Discretise/test_physical_boundary_conditions.jl",
        "unit/Discretise/test_laplace.jl",
        "unit/Discretise/test_neumann_sign.jl",
        "unit/Discretise/test_robin.jl",
        "unit/Discretise/test_nonorthogonal_flux.jl",
        "unit/Discretise/test_boundary_accumulation.jl",
        "unit/Discretise/test_linear_upwind.jl",
        "unit/Discretise/test_lustdc.jl",
        "unit/Discretise/test_lustlu.jl",
        "unit/Discretise/test_crank_nicolson.jl",
        "unit/Solve/test_smoothers.jl",
        "unit/Solve/test_DILU.jl",
        "unit/Solve/test_amg_freeze.jl",
        "unit/Solve/test_AMG.jl",
        "unit/Solve/test_AMG_matrices.jl",
        "unit/Solve/test_xvector.jl",
        "unit/Solve/test_bicgstab_shadow.jl",
        "unit/Solve/test_krylov_breakdown.jl",
        "unit/Solve/test_solve_warning.jl",
        "unit/Solve/test_mixed_precision.jl",
        "unit/Solvers/test_reconstruct.jl",
        "unit/Solvers/test_potential_flow.jl",
        "unit/Solvers/test_run_continuation.jl",
        "unit/Solvers/test_compressible_mass_flux.jl",
        "unit/Solvers/test_consistent_coupling.jl",
        "unit/Solvers/test_dt_consistent_flux.jl",
        "unit/Solvers/test_periodic_mass_flux.jl",
        "unit/Solvers/test_film_dt_consistency.jl",
        "unit/ModelPhysics/test_wall_function_empty_patch.jl",
        "unit/ModelPhysics/test_wall_production_density.jl",
        "unit/ModelPhysics/test_wall_function_averaging.jl",
        "unit/ModelPhysics/test_parallel_cell_updates.jl",
        "unit/ModelPhysics/test_lke_diffusion.jl",
        "unit/ModelPhysics/test_lke_production.jl",
        "unit/ModelPhysics/test_transpose_stress.jl",
        "unit/ModelPhysics/test_thermo_psi_time.jl",
        "unit/ModelPhysics/test_viscous_work.jl",
        "unit/ModelPhysics/test_kinetic_energy_flux.jl",
        "unit/ModelPhysics/test_viscosity_models.jl",
        "unit/ModelPhysics/test_fluidProperties.jl",
        "unit/ModelPhysics/test_turbulence_float_types.jl",
        "unit/Postprocess/test_field_average.jl",
        "unit/Postprocess/test_field_rms.jl",
        "unit/Postprocess/test_reynolds_stress.jl",
        "unit/Postprocess/test_qcriterion_window.jl",
        "unit/Postprocess/test_post_process_transient.jl",
        "unit/Postprocess/test_post_process_steady.jl",
        "unit/Postprocess/test_viscous_force_bcs.jl",
        "unit/Postprocess/test_probe.jl",
    ],
    "cases" => [
        "cases/test_2d_laplace_steady.jl",
        "cases/test_2d_laplace_unsteady.jl",
        "cases/test_transient_verification_examples.jl",
        "cases/test_adaptive_dt.jl",
        "cases/test_2d_incompressible_flatplate_KOmega_lowRe.jl",
        "cases/test_2d_incompressible_flatplate_KOmega_HighRe.jl",
        "cases/test_2d_incompressible_laminar_BFS.jl",
        "cases/test_2d_incompressible_laminar_rotatingFlatplate_MRF.jl",
        "cases/test_2d_incompressible_transient_KOmega_BFS_lowRe.jl",
        "cases/test_2d_incompressible_transient_laminar_BFS.jl",
        "cases/test_2d_incompressible_transient_cylinder_oscillating.jl",
        "cases/test_3d_incompressible_laminar_BFS.jl",
        "cases/test_3d_incompressible_laminar_cascade_periodic.jl",
        "cases/test_2d_incompressible_pitzdaily_LES.jl",
        "cases/test_3d_incompressible_isotropic_turbulence_WALE.jl",
        "cases/test_2d_taylor_couette_laminar.jl",
        "cases/test_2d_compressible_KOmega_flatplate.jl",
        "cases/test_2d_compressible_laminar_flatplate.jl",
        "cases/test_2d_compressible_transient_laminar_heated_cylinder.jl",
        "cases/test_2d_compressible_transient_cylinder_energy_models.jl",
        "cases/test_2d_compressible_supersonic_compression_corner.jl",
        "cases/test_2d_godunov_supersonic_cylinder.jl",
        "cases/test_2d_godunov_supersonic_cylinder_RANS.jl",
        "cases/test_2d_multiphase_gravity.jl",
        "cases/test_2d_multiphase_hydrostatic.jl",
        "cases/test_2d_multiphase_mixture.jl",
        "cases/test_2d_EFM.jl",
    ],
    "distributed" => [
        "distributed/gate.jl",
    ],
]

# NEW SECTION: tier selection (ARGS or XCAL_TEST_TIER, comma/space separated; default all)
valid_tiers = first.(TIERS)
requested = String[]
for s in vcat(ARGS, split(get(ENV, "XCAL_TEST_TIER", ""), ','))
    append!(requested, filter(!isempty, split(strip(s), r"[,\s]+")))
end
isempty(requested) && (requested = copy(valid_tiers))
bad = setdiff(requested, valid_tiers)
isempty(bad) || error("Unknown test tier(s) $(bad). Valid tiers: $(join(valid_tiers, ", "))")
selected = filter(t -> first(t) in requested, TIERS)

# NEW SECTION: run
Base.cumulative_compile_timing(true)
timings = Tuple{String,Float64,Float64}[]

@testset verbose = true "Functionality tests" begin
    for (tier, files) in selected
        @testset "$tier" verbose = true begin
            for file in files
                @testset "$file" begin
                    c0 = Base.cumulative_compile_time_ns()[1]
                    t = @elapsed include(joinpath(@__DIR__, file))
                    tc = (Base.cumulative_compile_time_ns()[1] - c0) / 1e9
                    push!(timings, (file, t, tc))
                    println("[$tier] $file: $(round(t, digits=1)) s ($(round(tc, digits=1)) s compiling)")
                end
            end
        end
    end

    foreach(rm, filter(endswith(".vtk"), readdir(pwd(), join=true)))
    foreach(rm, filter(endswith(".vtu"), readdir(pwd(), join=true)))
    foreach(x->rm(x, recursive=true), filter(contains("0."), readdir(pwd(), join=true)))
    foreach(x->rm(x, recursive=true), filter(contains("constant"), readdir(pwd(), join=true)))
end

# NEW SECTION: timing summary
println("\nTest timing summary (slowest first), total $(round(sum(x -> x[2], timings), digits=1)) s, compiling $(round(sum(x -> x[3], timings), digits=1)) s")
for (file, t, tc) in sort(timings; by=x -> x[2], rev=true)
    println(rpad(round(t, digits=1), 8), rpad(round(tc, digits=1), 8), file)
end
