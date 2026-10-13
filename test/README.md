# XCALibre.jl tests

## Layout
- `runtests.jl`: tier selector + explicit ordered file registry (`TIERS`), the single index of what runs
- `unit/<Module>/test_<topic>.jl`: tiny-mesh checks; folder = `src/` module under test
- `cases/test_<case>.jl`: short end-to-end simulations with physics assertions
- `distributed/`: MPI tier, entry point `gate.jl`
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
- Whole suite under 10 min per Julia version on CI, no coverage
- Unit files take seconds; each case under ~15 s (cases tier ~2.5 min)
- Unit: tiny meshes, assert the property (exactness, conservation, sign, types)
- Cases: assert physics (convergence, reference values), not just "it ran"

## Index (tier -> folder -> file)
unit/Mesh: test_mesh_conversion (readers, writers), test_foam_padded_neighbour, test_mesh_reorder, test_face_orientation
unit/Preprocess: test_setFields (region initialisation)
unit/Calculate: test_wall_distance
unit/Discretise: test_physical_boundary_conditions, test_laplace (analytic), test_neumann_sign, test_robin, test_nonorthogonal_flux (flux continuity)
unit/Solve: test_smoothers, test_DILU, test_amg_freeze, test_AMG, test_AMG_matrices, test_xvector, test_bicgstab_shadow, test_mixed_precision
unit/Solvers: test_reconstruct (face fluxes), test_potential_flow, test_run_continuation, test_compressible_mass_flux, test_consistent_coupling (SIMPLEC)
unit/ModelPhysics: test_wall_function_* and test_wall_production_density (wall functions), test_viscosity_models, test_parallel_cell_updates, test_lke_diffusion, test_lke_production, test_transpose_stress, test_viscous_work, test_thermo_psi_time, test_fluidProperties (NIST), test_turbulence_float_types
unit/Postprocess: test_field_average, test_field_rms, test_reynolds_stress, test_post_process_transient, test_post_process_steady
cases: test_2d_laplace_*, test_adaptive_dt, test_*incompressible* (pitzdaily_LES: KEquation+Smagorinsky; transient_laminar_BFS: Euler+CrankNicolson), test_2d_taylor_couette_laminar, test_*compressible* (flatplate files hold viscosity/wall-BC variants), test_2d_godunov_supersonic_cylinder, test_2d_multiphase_*, test_2d_EFM
distributed: gate.jl (MPI gate: test_partition in-process, then one 3-rank gate_session.jl including each file in its own module)

## Not registered (known issues)
- unit/ModelPhysics/test_mixture_multiphase.jl: needs `examples/0_GRIDS/unit_test_stage2.unv` (missing)
