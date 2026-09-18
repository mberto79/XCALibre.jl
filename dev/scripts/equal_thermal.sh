#!/bin/bash
# Strong scaling at CONSTANT package power: at rank count n the solver takes physical cores
# 0..n-1 and spin-loops occupy the remaining P-cores, so every rank count runs the machine at
# the same sustained power limit. Spin loops are pure ALU: they equalise clocks without
# competing for memory bandwidth. Removes throttling as a variable without needing root.
set -u
REPO=/home/humberto/Julia/XCALibre.jl
ENVDIR=$REPO/dev/petscenv_stock
CACHE=$HOME/.cache/xcal_scaling_probe
TAG=${1:-bfs_tet_5mm}
ITERS=${2:-100}
cd $REPO

for n in 1 2 4 8; do
    PARTDIR=$CACHE/${TAG}_n$n
    [ -d "$PARTDIR" ] || { echo "SKIP n=$n (no $PARTDIR)"; continue; }
    HEAT=()
    for (( c=2*n; c<16; c+=2 )); do
        taskset -c "$c-$((c+1))" sh -c 'while :; do :; done' &
        HEAT+=($!)
    done
    SAMPLE=$(mktemp)
    ( while :; do grep 'cpu MHz' /proc/cpuinfo | awk -F: 'NR<=16 {print $2}' >> "$SAMPLE"; sleep 2; done ) &
    SAMPLER=$!
    sleep 25   # reach the sustained power limit before timing
    OUT=$($HOME/.julia/bin/mpiexecjl -n $n --bind-to core --map-by core \
          julia --startup-file=no --project=$ENVDIR \
          $REPO/dev/scripts/scaling_probe.jl worker "$PARTDIR" $ITERS 2>&1 | grep '^PROBE')
    kill $SAMPLER 2>/dev/null
    [ ${#HEAT[@]} -gt 0 ] && kill ${HEAT[@]} 2>/dev/null
    MEAN=$(awk '{s+=$1; c++} END {if(c) printf "%.0f", s/c; else print 0}' "$SAMPLE")
    rm -f "$SAMPLE"
    echo "EQT n=$n heaters=$(( 8 - n )) mean_pcore_mhz=$MEAN $OUT"
    sleep 15   # let the package settle between points
done
echo EQT_DONE
