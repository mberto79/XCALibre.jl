#!/usr/bin/env bash
# amg_loop.sh — AMG optimization loop
#
# Alternates between: julia benchmark → headless Claude analysis+edit.
# Stops when ratio < TARGET_RATIO, STATUS: EXHAUSTED appears, or MAX_ITER is reached.
#
# Run from XCALibre.jl project root:
#   ./amg_loop.sh
#
# Environment variables (all optional):
#   MAX_ITER=10          max optimization iterations (default 10)
#   TARGET_RATIO=0.60    stop when AMG/CG ratio drops below this (default 0.60)
#   SMOKE=1              3-iteration benchmark for pipeline testing (default 0)
#   CLAUDE_MODEL=sonnet  claude model to use (default sonnet)

set -euo pipefail

MAX_ITER=${MAX_ITER:-3}
TARGET_RATIO=${TARGET_RATIO:-0.60}
SMOKE=${SMOKE:-0}
CLAUDE_MODEL=${CLAUDE_MODEL:-sonnet}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/amg_loop_logs"
mkdir -p "$LOG_DIR"

if [ "$SMOKE" = "1" ]; then
    export SMOKE_TEST=3
    echo "SMOKE mode: 3-iteration benchmark"
else
    unset SMOKE_TEST 2>/dev/null || true
fi

CLAUDE_CMD="claude --model ${CLAUDE_MODEL} \
    --allowedTools Edit,Read,Write,Bash,Grep,Glob \
    --max-turns 40 \
    --permission-mode acceptEdits"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AMG Optimization Loop                                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Target  : ratio < ${TARGET_RATIO}                                  ║"
echo "║  Max iter: ${MAX_ITER}                                              ║"
echo "║  Model   : ${CLAUDE_MODEL}                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Verify we're in the right place ───────────────────────────────────────────
if [ ! -f "CLAUDE.md" ] || [ ! -f "amg_loop_prompt.md" ]; then
    echo "ERROR: Run from XCALibre.jl project root (must contain CLAUDE.md and amg_loop_prompt.md)"
    exit 1
fi

# ── Verify claude is available ─────────────────────────────────────────────────
if ! command -v claude &>/dev/null; then
    echo "ERROR: 'claude' not found in PATH. Install Claude Code CLI first."
    exit 1
fi

LAST_RATIO="n/a"

# ── Pre-launch state check ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Pre-launch: state check + recovery"
echo "══════════════════════════════════════════════════════════════"
PRE_LOG="${LOG_DIR}/pre_launch_claude.txt"
echo "[$(date '+%H:%M:%S')] Invoking Claude for state check (model=${CLAUDE_MODEL})..."
echo "  Log: ${PRE_LOG}"

PRE_PROMPT="$(cat amg_loop_prompt.md)

---
MODE: PRE-LAUNCH STATE CHECK
See ## Pre-launch Mode section for instructions.
BENCHMARK LOG: F1-fetchCFD_Minimal/amg_loop_results.txt"

set +e
claude --model "${CLAUDE_MODEL}" \
    --allowedTools Edit,Read,Write,Bash,Grep,Glob \
    --max-turns 15 \
    --permission-mode acceptEdits \
    -p "$PRE_PROMPT" \
    2>&1 | tee "$PRE_LOG"
PRE_EXIT=${PIPESTATUS[0]}
set -e

if [ "$PRE_EXIT" -ne 0 ]; then
    echo "WARNING: Pre-launch Claude exited with code ${PRE_EXIT}. Check ${PRE_LOG}."
fi
echo "[$(date '+%H:%M:%S')] Pre-launch check complete."
echo ""

for iter in $(seq 1 "$MAX_ITER"); do
    TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Iteration ${iter}/${MAX_ITER}   [${TIMESTAMP}]"
    echo "══════════════════════════════════════════════════════════════"

    # ── Run benchmark ──────────────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')] Running Julia benchmark..."
    BENCH_LOG="${LOG_DIR}/iter_${iter}_benchmark.txt"

    set +e
    julia --project \
        F1-fetchCFD_Minimal/amg_loop_profile.jl \
        2>&1 | tee "$BENCH_LOG"
    BENCH_EXIT=${PIPESTATUS[0]}
    set -e

    if [ "$BENCH_EXIT" -ne 0 ]; then
        echo ""
        echo "ERROR: Julia benchmark exited with code ${BENCH_EXIT}."
        echo "       Check ${BENCH_LOG} for details."
        exit 1
    fi

    # ── Parse parseable output ─────────────────────────────────────────────────
    RATIO=$(grep '^BENCHMARK_RATIO=' "$BENCH_LOG" | tail -1 | cut -d= -f2)
    CG_TIME=$(grep '^BENCHMARK_CG_TIME=' "$BENCH_LOG" | tail -1 | cut -d= -f2)
    AMG_TIME=$(grep '^BENCHMARK_AMG_TIME=' "$BENCH_LOG" | tail -1 | cut -d= -f2)

    if [ -z "$RATIO" ]; then
        echo ""
        echo "ERROR: BENCHMARK_RATIO not found in benchmark output."
        echo "       The benchmark script may have crashed before printing results."
        echo "       Check ${BENCH_LOG}."
        exit 1
    fi

    LAST_RATIO="$RATIO"
    echo ""
    echo "  Cg+Jacobi : ${CG_TIME} s"
    echo "  AMG       : ${AMG_TIME} s"
    echo "  Ratio     : ${RATIO}  (target < ${TARGET_RATIO})"

    # ── Check if target achieved ───────────────────────────────────────────────
    if awk "BEGIN { exit !( ${RATIO} + 0 < ${TARGET_RATIO} + 0 ) }"; then
        echo ""
        echo "  ✓ TARGET ACHIEVED: ratio ${RATIO} < ${TARGET_RATIO}"
        echo ""
        echo "Loop complete — target reached after ${iter} iteration(s)."
        echo "Final ratio: ${RATIO}"
        exit 0
    fi

    # ── Check if agent reported exhaustion ────────────────────────────────────
    if grep -q "STATUS: EXHAUSTED" amg_loop_state.md 2>/dev/null; then
        echo ""
        echo "  Agent reports exhausted — no further improvements found."
        echo ""
        echo "Loop complete — agent exhausted after ${iter} iteration(s)."
        echo "Best ratio achieved: ${RATIO}"
        exit 0
    fi

    # ── Check if last iteration (don't invoke Claude after final run) ──────────
    if [ "$iter" -eq "$MAX_ITER" ]; then
        echo ""
        echo "  MAX_ITER (${MAX_ITER}) reached. Best ratio: ${RATIO}"
        echo "  Increase MAX_ITER or review amg_loop_state.md for next steps."
        exit 0
    fi

    # ── Invoke headless Claude ────────────────────────────────────────────────
    CLAUDE_LOG="${LOG_DIR}/iter_${iter}_claude.txt"
    echo "[$(date '+%H:%M:%S')] Invoking Claude (model=${CLAUDE_MODEL}, max-turns=40)..."
    echo "  Log: ${CLAUDE_LOG}"

    PROMPT="$(cat amg_loop_prompt.md)

---
CURRENT ITERATION: ${iter}
CURRENT RATIO: ${RATIO}
TARGET RATIO: ${TARGET_RATIO}
PREVIOUS RATIO: ${LAST_RATIO}
BENCHMARK LOG: F1-fetchCFD_Minimal/amg_loop_results.txt
"

    set +e
    echo "$PROMPT" | $CLAUDE_CMD -p "$(cat amg_loop_prompt.md)

CURRENT ITERATION: ${iter}
CURRENT RATIO: ${RATIO}
TARGET RATIO: ${TARGET_RATIO}
BENCHMARK LOG: F1-fetchCFD_Minimal/amg_loop_results.txt" \
        2>&1 | tee "$CLAUDE_LOG"
    CLAUDE_EXIT=${PIPESTATUS[0]}
    set -e

    if [ "$CLAUDE_EXIT" -ne 0 ]; then
        echo ""
        echo "WARNING: Claude exited with code ${CLAUDE_EXIT}. Continuing loop."
        echo "         Check ${CLAUDE_LOG} for details."
    fi

    echo "[$(date '+%H:%M:%S')] Claude analysis complete."

    # ── Re-check exhaustion (Claude may have just set it) ─────────────────────
    if grep -q "STATUS: EXHAUSTED" amg_loop_state.md 2>/dev/null; then
        echo ""
        echo "  Agent set STATUS: EXHAUSTED. Stopping loop."
        exit 0
    fi

    echo ""
done

echo "Loop finished. Best ratio: ${LAST_RATIO}"
