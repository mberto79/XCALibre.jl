#!/bin/bash
# OpenFOAM strong-scaling reference for the same backward-facing-step case (P1-M3).
# Copies the benchmark case to a scratch directory, runs PCG+diagonal / PBiCGStab+diagonal at
# each rank count with the same core binding XCALibre is measured under, and reports the
# per-iteration cost as (T100 - T3) / 97 so start-up and mesh loading cancel.
#   dev/scripts/openfoam_scaling.sh <n>...
# the case is OpenFOAM 12 (foamRun, constant/momentumTransport); the shell here usually has
# the ESI build sourced, so re-enter from a clean environment before using this script:
#   env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc 'source $HOME/OpenFOAM/OpenFOAM-12/etc/bashrc; dev/scripts/openfoam_scaling.sh 1 2 4 8'
set -euo pipefail
SRC=/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_BFS_laminar/OpenFOAM
WORK=$HOME/.cache/xcal_of_scaling

# /tmp is memory-backed on this machine, so the case never goes there
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$WORK/"
cd "$WORK"
foamDictionary system/controlDict -entry endTime -set 110 > /dev/null
foamDictionary system/controlDict -entry writeInterval -set 1000 > /dev/null
foamDictionary system/fvSolution -entry solvers/p/solver -set PCG > /dev/null
foamDictionary system/fvSolution -entry solvers/p/preconditioner -set diagonal > /dev/null
foamDictionary system/fvSolution -entry solvers/U/solver -set PBiCGStab > /dev/null
foamDictionary system/fvSolution -entry solvers/U/preconditioner -set diagonal > /dev/null

per_iter() { awk '
    /^Time = / { t = $3 }
    /^ExecutionTime = / { if (t == 3) t3 = $3; if (t == 100) t100 = $3 }
    END { printf "%.4f %s %s", (t100 - t3) / 97, t3, t100 }' "$1"; }
max_mhz() { awk -F: '/cpu MHz/ {if ($2+0 > m) m = $2+0} END {printf "%.0f", m}' /proc/cpuinfo; }

for n in "$@"; do
    rm -rf processor* [1-9]* 0.*
    if [ "$n" -eq 1 ]; then
        foamRun > "log.n$n" 2>&1
    else
        foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$n" > /dev/null
        decomposePar > /dev/null 2>&1
        mpirun -np "$n" --bind-to core --map-by core foamRun -parallel > "log.n$n" 2>&1
    fi
    echo "OF nranks=$n per_iter=$(per_iter "log.n$n") mhz=$(max_mhz)"
done
