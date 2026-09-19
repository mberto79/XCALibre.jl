#!/usr/bin/env bash
# Usage: pin_clock.sh pin|revert  (pin = 2200 MHz base clock on every core; needs sudo)
set -e
case "$1" in
pin)
  powerprofilesctl set performance
  echo 1   | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor ;;
revert)
  echo 0  | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  echo 15 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
  echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  echo balance_performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
  powerprofilesctl set balanced ;;
*) echo "usage: $0 pin|revert" >&2; exit 1 ;;
esac
