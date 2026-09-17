#!/bin/bash
# OpenFOAM strong-scaling reference for the backward-facing-step case; see dev/scripts/INDEX.md.
# Per-iteration cost is (T100 - T3) / 97, so start-up and mesh loading cancel.
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
        # pinned to one physical core, matching what mpiexec gives every other rank count
        taskset -c 0-1 foamRun > "log.n$n" 2>&1
    else
        foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$n" > /dev/null
        decomposePar > /dev/null 2>&1
        mpirun -np "$n" --bind-to core --map-by core foamRun -parallel > "log.n$n" 2>&1
    fi
    echo "OF nranks=$n per_iter=$(per_iter "log.n$n") mhz=$(max_mhz)"
done
