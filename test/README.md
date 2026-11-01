# XCALibre.jl tests

## Layout
- `runtests.jl`: tier selector + explicit ordered file registry (`TIERS`), the single index of what runs
- `unit/<Module>/test_<topic>.jl`: tiny-mesh checks; folder = `src/` module under test
- `cases/test_<case>.jl`: short end-to-end simulations with physics assertions
- `distributed/`: MPI tier, entry point `gate.jl`
- `gpu/test_<topic>.jl`: CPU/GPU comparisons, not in a tier (need a GPU): `julia --project=<env with XCALibre and CUDA> test/gpu/test_<topic>.jl`
- `grids/`, `CBC_Data/`, `Fluids_NIST_Data/`: test data

## Naming
`test_<topic>.jl`; a file not listed in `TIERS` never runs.

## Running
- All tiers (default): `] test XCALibre`
- Select: `Pkg.test("XCALibre"; test_args=["unit"])` or `["cases", "distributed"]`
- Env: `XCAL_TEST_TIER=unit,cases`; direct: `julia --project=test test/runtests.jl unit`
- Unknown tier names error with the valid list; per-file times and a sorted summary print at the end

## Adding a test
1. Create `unit/<Module>/test_<topic>.jl` (or `cases/test_<case>.jl`)
2. Add its path to the tier in `runtests.jl`
3. Load grids via `pkgdir(XCALibre, ...)`; `workgroupsize(mesh)` comes from `runtests.jl`

## Budget and conventions
- CI, no coverage, one parallel job per tier (unit and cases on Julia 1.10 and the latest release, distributed on the latest release only). Job times, 1.10 / latest: unit ~13 / 15 min, cases ~10 / 13 min, distributed ~11 min; the slowest job sets the wall-clock. About 70% of test time is compilation, so cutting steps in a file rarely helps: cut distinct specialisations instead
- Most unit files take seconds (solver-based checks such as test_crank_nicolson up to ~1 min); each case under ~15 s on a desktop CPU
- Unit: tiny meshes, assert the property (exactness, conservation, sign, types)
- Cases: assert physics (convergence, reference values), not just "it ran"

## Index (tier -> folder -> file)
unit/Mesh: test_mesh_conversion (readers, writers), test_foam_padded_neighbour, test_mesh_reorder, test_face_orientation, test_bounding_box, test_boundary_distance (boundary face delta), test_unv2d_geometry (centroids, face geometry consistency)
unit/IOFormats: test_foam_writer_unset_bcs
unit/Multithread: test_kernel_index_pattern (source scan: `@index` bound on its own)
unit/Preprocess: test_setFields (region initialisation)
unit/Calculate: test_wall_distance, test_least_squares, test_venkat_limiter
unit/Discretise: test_physical_boundary_conditions, test_laplace (analytic), test_neumann_sign, test_robin, test_nonorthogonal_flux (flux continuity), test_boundary_accumulation (deterministic sums), test_linear_upwind, test_lustdc, test_lustlu, test_crank_nicolson (temporal order)
unit/Solve: test_smoothers, test_DILU, test_amg_freeze, test_AMG, test_AMG_matrices, test_xvector, test_bicgstab_shadow, test_krylov_breakdown, test_solve_warning, test_mixed_precision, test_stop_criterion (one stopping rule on every path), test_residual_parity (same outer residual on every linear-solve path)
unit/Solvers: test_reconstruct (face fluxes), test_potential_flow, test_run_continuation, test_compressible_mass_flux, test_consistent_coupling (SIMPLEC), test_dt_consistent_flux, test_periodic_mass_flux, test_film_dt_consistency
unit/ModelPhysics: test_wall_function_* and test_wall_production_density (wall functions), test_viscosity_models, test_parallel_cell_updates, test_lke_diffusion, test_lke_production, test_transpose_stress, test_viscous_work, test_kinetic_energy_flux, test_thermo_psi_time, test_fluidProperties (NIST), test_turbulence_float_types
unit/Postprocess: test_field_average, test_field_rms, test_reynolds_stress, test_qcriterion_window, test_post_process_transient, test_post_process_steady, test_viscous_force_bcs, test_probe
cases: test_2d_laplace_*, test_transient_verification_examples (runs examples/transient_verification/heat_decay.jl), test_adaptive_dt, test_*incompressible* (pitzdaily_LES: KEquation+Smagorinsky; transient_laminar_BFS: Euler+CrankNicolson), test_2d_taylor_couette_laminar, test_*compressible* (flatplate files hold viscosity/wall-BC variants), test_2d_godunov_supersonic_cylinder (+ _RANS), test_2d_multiphase_*, test_2d_EFM
distributed: gate.jl (MPI gate: one 2-rank gate_session.jl including each file in its own module; test_halo, test_halo_subcomm and test_turbulence_sst_wallfn at 3 ranks, skipped unless `XCALIBRE_TEST_MAX_RANKS` >= 3, which CI sets). Other files and rank counts: `runtests_mpi.jl`

## Not registered (known issues)
- unit/ModelPhysics/test_mixture_multiphase.jl: needs `examples/0_GRIDS/unit_test_stage2.unv` (missing)
