#!/bin/bash
# Full serial suite as five groups, one process each at -t 4 (each under the D101 cap); usage: suite_groups.sh <outdir>
T=~/Julia/XCALibre.jl/test; C=$T/0_TEST_CASES; O=$1; mkdir -p $O; cd $O
G1="$T/test_mesh_conversion.jl $T/test_reconstruct.jl $T/test_physical_boundary_conditions.jl $T/test_potential_flow.jl $T/unit_test_wall_function_empty_patch.jl $T/test_smoothers.jl $T/test_DILU.jl $T/unit_test_laplace.jl $T/test_AMG.jl $T/test_AMG_matrices.jl $T/unit_test_xvector.jl $T/unit_test_wall_distance.jl $T/unit_test_wall_production_density.jl $T/unit_test_wall_function_averaging.jl $T/unit_test_fluidProperties.jl $T/unit_test_field_average.jl $T/unit_test_field_rms.jl $T/unit_test_reynolds_stress.jl $T/unit_test_post-process_transient.jl $T/unit_test_post-process_steady.jl $C/2d_laplace_steady.jl $C/2d_laplace_unsteady.jl $C/adaptive_dt.jl"
G2="$C/2d_incompressible_flatplate_KOmega_lowRe.jl $C/2d_incompressible_flatplate_KOmega_HighRe.jl $C/2d_incompressible_laminar_BFS.jl $C/2d_incompressible_laminar_rotatingFlatplate_MRF.jl $C/2d_incompressible_transient_KOmega_BFS_lowRe.jl $C/2d_incompressible_transient_laminar_BFS.jl $C/2d_incompressible_transient_laminar_BFS_CrankNicolson.jl"
G3="$C/2d_incompressible_transient_cylinder_oscillating.jl $C/3d_incompressible_laminar_BFS.jl $C/3d_incompressible_laminar_cascade_periodic.jl $C/2d_incompressible_pitzdaily_KEquation.jl $C/2d_incompressible_pitzdaily_Smagorinsky.jl $C/2d_taylor_couette_laminar.jl"
G4="$C/2d_compressible_KOmega_flatplate_fixedT.jl $C/2d_compressible_laminar_flatplate_fixedT.jl $C/2d_compressible_transient_laminar_heated_cylinder.jl $C/2d_compressible_transient_cylinder_energy_models.jl $C/2d_compressible_supersonic_compression_corner.jl"
G5="$C/2d_godunov_supersonic_cylinder.jl $C/2d_multiphase_gravity.jl $C/2d_multiphase_hydrostatic.jl $C/2d_multiphase_mixture.jl $C/2d_EFM.jl"
for g in G1 G2 G3 G4 G5; do
  t0=$(date +%s)
  timeout 295 julia --project=$HOME/.cache/xcal_m28/env_test --startup-file=no -t 4 ~/Julia/XCALibre.jl/dev/scripts/suite_file.jl ${!g} > $g.log 2>&1
  echo "$g exit=$? s=$(( $(date +%s) - t0 ))" >> status
done
grep -h "^SUITE" G*.log > summary; echo DONE >> status
