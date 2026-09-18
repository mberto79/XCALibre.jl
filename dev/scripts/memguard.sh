#!/bin/bash
# Run a command, killing its process group if MemAvailable drops below MIN_MB (default 1500).
MIN_MB=${MIN_MB:-1500}
setsid "$@" &
PID=$!
while kill -0 $PID 2>/dev/null; do
    AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$AVAIL" -lt "$MIN_MB" ]; then
        echo "MEMGUARD: MemAvailable ${AVAIL} MB < ${MIN_MB} MB, killing run" >&2
        kill -TERM -- -$PID; sleep 2; kill -KILL -- -$PID 2>/dev/null
        pkill -f scaling_probe.jl
        exit 137
    fi
    sleep 1
done
wait $PID
