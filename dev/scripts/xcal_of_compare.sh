#!/bin/bash
# Like-for-like strong scaling, XCALibre against OpenFOAM, on the same mesh with the same
# solver settings and the same core binding (P1-M3). Both report (t100 - t3) / 97 s/iter.
#   dev/scripts/xcal_of_compare.sh
set -uo pipefail
MESH=/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_BFS_laminar/XCALibre/bfs_tet_5mm.unv
RANKS="1 2 4 8"

echo "== OpenFOAM (PCG+diagonal, PBiCGStab+diagonal) =="
env -i HOME="$HOME" PATH=/usr/bin:/bin TERM=dumb bash -lc \
  "source \$HOME/OpenFOAM/OpenFOAM-12/etc/bashrc; cd $PWD; dev/scripts/openfoam_scaling.sh $RANKS"

echo "== XCALibre (Cg+Jacobi, Bicgstab+Jacobi) =="
julia --startup-file=no --project=dev/petscenv_stock \
  dev/scripts/scaling_probe.jl drive "$MESH" $RANKS 100
