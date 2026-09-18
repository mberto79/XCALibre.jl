#!/bin/bash
# Strong scaling with the clock pinned by hardware (no_turbo=1, min=max=100%), so rank count
# and clock are decoupled without spin-loop heaters. A sampler records the clock actually held
# during each timed run, so the pin is verified rather than assumed.
set -u
REPO=/home/humberto/Julia/XCALibre.jl
ENVDIR=$REPO/dev/petscenv_stock
CACHE=$HOME/.cache/xcal_scaling_probe
cd $REPO
run_one() {
    local tag=$1 n=$2 iters=$3
    local partdir=$CACHE/${tag}_n${n}
    [ -d "$partdir" ] || { echo "SKIP $tag n=$n (no partdir)"; return; }
    local sample; sample=$(mktemp)
    ( while :; do grep 'cpu MHz' /proc/cpuinfo | awk -F: 'NR<=16 {print $2}' >> "$sample"; sleep 2; done ) &
    local sampler=$!
    local out; out=$($HOME/.julia/bin/mpiexecjl -n $n --bind-to core --map-by core \
        julia --startup-file=no --project=$ENVDIR \
        $REPO/dev/scripts/scaling_probe.jl worker "$partdir" $iters 2>&1 | grep '^PROBE')
    kill $sampler 2>/dev/null
    local stats; stats=$(awk '{s+=$1;c++; if($1>mx)mx=$1} END {printf "%.0f %.0f", s/c, mx}' "$sample")
    rm -f "$sample"
    echo "FIXED tag=$tag n=$n mean_mhz=${stats% *} max_mhz=${stats#* } $out"
}
for n in 1 2 4 8; do run_one bfs_tet_5mm  $n 100; done
for n in 1 2 4 6; do run_one bfs_unv_tet_4mm $n 100; done
echo FIXED_DONE
