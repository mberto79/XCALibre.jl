#!/bin/bash
# Heated single-rank control: run one rank on core 0 while the other five
# performance cores are loaded, so rank count and thermal state stop co-varying.
#   dev/scripts/heated_control.sh <partdir_n1> <iters> <env>
set -u
PARTDIR=$1; ITERS=$2; ENVDIR=${3:-dev/petscenv_stock}
SAMPLE=$(mktemp)

for c in 2 4 6 8 10; do
    taskset -c "$c-$((c+1))" sh -c 'while :; do :; done' &
done
HEATERS=$(jobs -p)
( while :; do grep 'cpu MHz' /proc/cpuinfo | awk -F: 'NR<=16 {print $2}' >> "$SAMPLE"; sleep 2; done ) &
SAMPLER=$!

sleep 20 # let the package hit its sustained power limit before the timed run
julia --startup-file=no --project="$ENVDIR" -e "using MPI; run(\`\$(MPI.mpiexec()) -n 1 --bind-to core --map-by core \$(Base.julia_cmd()) --startup-file=no --project=$ENVDIR dev/scripts/scaling_probe.jl worker $PARTDIR $ITERS\`)" 2>&1 | grep -E 'PROBE|ERROR'

kill $SAMPLER $HEATERS 2>/dev/null
awk '{s+=$1; n++} END {printf "HEATED mean_pcore_mhz=%.0f samples=%d\n", s/n, n}' "$SAMPLE"
rm -f "$SAMPLE"
