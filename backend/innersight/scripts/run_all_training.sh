#!/usr/bin/env bash
#
# run_all_training.sh — train all four ladder steps end-to-end (Phase 6).
#
# Runs, in order: compute baselines → XGBoost → MLP → Temporal CNN →
# Temporal+Graph → Full fusion. Each step has a 45-minute timeout, failures are
# logged and do not abort the run, and steps whose results JSON already exists
# are skipped (idempotent resume) unless --force is given.
#
# Designed to be portable to bash 3.2 (macOS default).

set -o pipefail   # NOTE: not `set -e` — we continue past a failing step.

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR=""
VERSION="r4.2"
STORE_DIR="feature_store"
OUTPUT_DIR="training_results"
QUICK=0
FORCE=0
TIMEOUT=2700   # default 45 minutes per step (override with --timeout; heavy graph/
               # fusion steps may legitimately need much longer)

usage() {
    cat <<'EOF'
Usage: bash run_all_training.sh --data-dir PATH [options]

Train the full progressive ladder (XGBoost → MLP → Temporal → Temporal+Graph → Fusion).

Required:
  --data-dir PATH     CERT dataset directory (raw logs, LDAP, psychometric, answers/).

Options:
  --version VER       CERT version (default: r4.2).
  --store-dir PATH    Feature store directory (default: feature_store).
  --output-dir PATH   Where results JSONs, checkpoints and logs go (default: training_results).
  --quick             Smoke test: 1 seed, 2 folds, 3 epochs (fast).
  --force             Re-run every step even if its results JSON already exists.
  --timeout SECONDS   Per-step time limit (default: 2700 = 45m; 0 = no limit).
                      The graph/fusion steps often need more — raise it for full runs.
  -h, --help          Show this help and exit.

Examples:
  # Full training (GPU session)
  bash run_all_training.sh --data-dir /data/cert_r4.2 --version r4.2

  # Quick smoke test (CI / local)
  bash run_all_training.sh --data-dir /data/cert_r4.2 --version r4.2 --quick
EOF
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --version)    VERSION="$2"; shift 2 ;;
        --store-dir)  STORE_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --quick)      QUICK=1; shift ;;
        --force)      FORCE=1; shift ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "ERROR: --data-dir is required." >&2
    usage >&2
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # innersight/scripts -> innersight -> backend
cd "$BACKEND_DIR" || { echo "ERROR: cannot cd to $BACKEND_DIR" >&2; exit 1; }

# Prefer the project venv's interpreter when present.
PYTHON="python"
if [ -x "$BACKEND_DIR/.venv/bin/python" ]; then
    PYTHON="$BACKEND_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
MAIN_LOG="$OUTPUT_DIR/run_all_training.log"
START_TIME=$(date +%s)
STEP_RESULTS=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# Resolve a timeout binary (GNU coreutils `timeout`, or `gtimeout` on macOS).
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
fi

run_with_timeout() {
    if [ -n "$TIMEOUT_BIN" ] && [ "$TIMEOUT" != "0" ]; then
        "$TIMEOUT_BIN" "$TIMEOUT" "$@"
    else
        "$@"
    fi
}

# run_step NAME RESULTS_JSON CMD...
#   RESULTS_JSON empty → never skip on existing results (e.g. compute_baselines,
#   which decides freshness itself).
run_step() {
    name="$1"
    results_json="$2"
    shift 2
    logf="$LOG_DIR/${name}.log"

    if [ -n "$results_json" ] && [ -f "$results_json" ] && [ "$FORCE" -ne 1 ]; then
        log "SKIP  $name — results exist ($results_json). Use --force to rerun."
        STEP_RESULTS="${STEP_RESULTS} ${name}:SKIP:0"
        return 0
    fi

    log "START $name  ->  $logf"
    t0=$(date +%s)
    if run_with_timeout "$@" >"$logf" 2>&1; then
        dt=$(( $(date +%s) - t0 ))
        log "PASS  $name (${dt}s)"
        STEP_RESULTS="${STEP_RESULTS} ${name}:PASS:${dt}"
    else
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        if [ "$rc" -eq 124 ]; then
            log "FAIL  $name — TIMEOUT after ${TIMEOUT}s (see $logf)"
            STEP_RESULTS="${STEP_RESULTS} ${name}:TIMEOUT:${dt}"
        else
            log "FAIL  $name — exit $rc (see $logf)"
            STEP_RESULTS="${STEP_RESULTS} ${name}:FAIL:${dt}"
        fi
    fi
}

# ── Header ────────────────────────────────────────────────────────────────────
log "================ run_all_training ================"
log "version=$VERSION  data-dir=$DATA_DIR  store-dir=$STORE_DIR  output-dir=$OUTPUT_DIR"
log "quick=$QUICK  force=$FORCE  python=$PYTHON"
log "per-step timeout=${TIMEOUT}s ($((TIMEOUT / 60))m; 0 = unlimited)"
if [ -z "$TIMEOUT_BIN" ] && [ "$TIMEOUT" != "0" ]; then
    log "WARN  no 'timeout'/'gtimeout' binary found — the per-step limit CANNOT be"
    log "WARN  enforced, so a hung/runaway step (e.g. the graph step) will run"
    log "WARN  unbounded. Install GNU coreutils ('apt-get install -y coreutils') to"
    log "WARN  enable it, or watch the run manually."
fi

# In --quick mode, write a minimal config the config-driven scripts merge over
# their defaults (1 seed, 2 folds, 3 epochs). XGBoost/MLP take --n-folds/--seeds
# directly instead.
QUICK_CFG="$OUTPUT_DIR/quick_config.yaml"
if [ "$QUICK" -eq 1 ]; then
    cat > "$QUICK_CFG" <<'EOF'
training:
  max_epochs: 3
  warmup_epochs: 1
  patience: 2
evaluation:
  n_folds: 2
  seeds: [42]
EOF
    log "quick mode: wrote $QUICK_CFG (max_epochs=3, n_folds=2, seeds=[42])"
fi

# ── Step 1: compute baselines (deviations) ───────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.compute_baselines \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR")
[ "$FORCE" -eq 1 ] && cmd=("${cmd[@]}" --force)
run_step "01_compute_baselines" "" "${cmd[@]}"

# ── Step 2: XGBoost baseline ─────────────────────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.xgboost_baseline \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$OUTPUT_DIR/xgboost_results.json")
[ "$QUICK" -eq 1 ] && cmd=("${cmd[@]}" --n-folds 2 --seeds 42)
run_step "02_xgboost" "$OUTPUT_DIR/xgboost_results.json" "${cmd[@]}"

# ── Step 3: MLP baseline ─────────────────────────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.train_mlp_baseline \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$OUTPUT_DIR/mlp_results.json")
[ "$QUICK" -eq 1 ] && cmd=("${cmd[@]}" --n-folds 2 --seeds 42)
run_step "03_mlp" "$OUTPUT_DIR/mlp_results.json" "${cmd[@]}"

# ── Step 4: Temporal CNN ─────────────────────────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.train_temporal \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$OUTPUT_DIR/temporal_results.json" --checkpoint-dir "$CKPT_DIR")
[ "$QUICK" -eq 1 ] && cmd=("${cmd[@]}" --config "$QUICK_CFG")
run_step "04_temporal" "$OUTPUT_DIR/temporal_results.json" "${cmd[@]}"

# ── Step 5: Temporal + Graph ─────────────────────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.train_temporal_graph \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$OUTPUT_DIR/temporal_graph_results.json" --checkpoint-dir "$CKPT_DIR")
[ "$QUICK" -eq 1 ] && cmd=("${cmd[@]}" --config "$QUICK_CFG")
run_step "05_temporal_graph" "$OUTPUT_DIR/temporal_graph_results.json" "${cmd[@]}"

# ── Step 6: Full fusion ──────────────────────────────────────────────────────
cmd=("$PYTHON" -m innersight.scripts.train_fusion \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$OUTPUT_DIR/fusion_results.json" --checkpoint-dir "$CKPT_DIR" \
     --baseline-results-dir "$OUTPUT_DIR")
[ "$QUICK" -eq 1 ] && cmd=("${cmd[@]}" --config "$QUICK_CFG")
run_step "06_fusion" "$OUTPUT_DIR/fusion_results.json" "${cmd[@]}"

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - START_TIME ))
log "================ SUMMARY ================"
for entry in $STEP_RESULTS; do
    step="${entry%%:*}"
    rest="${entry#*:}"
    status="${rest%%:*}"
    secs="${rest##*:}"
    log "$(printf '  %-8s %-22s (%ss)' "$status" "$step" "$secs")"
done
log "Total time: ${TOTAL}s ($((TOTAL / 60))m)"

# Exit non-zero if any step failed or timed out (skips/passes are OK).
case "$STEP_RESULTS" in
    *":FAIL:"*|*":TIMEOUT:"*) exit 1 ;;
    *) exit 0 ;;
esac
