#!/usr/bin/env bash
#
# reproduce.sh — re-run one experiment fast and check it reproduces (Phase 6).
#
# Runs a single ladder experiment with seed=42 and a minimal fold count, then
# compares its AUPRC against a saved reference from a previous reproduce run.
# The first run establishes the reference; later runs verify reproducibility.
#
# Determinism: seed_everything(42) plus pinned dependencies make results
# bit-for-bit reproducible on the same hardware. Cross-hardware (GPU vs CPU,
# different CUDA versions) is approximate.
#
# Note: stratified k-fold needs >= 2 splits, so the "fast" run uses 2 folds
# (the minimum), not 1.

set -o pipefail

EXPERIMENT=""
DATA_DIR=""
VERSION="r4.2"
STORE_DIR="feature_store"
REPRO_DIR="reproduce_results"
TOLERANCE="0.001"

usage() {
    cat <<'EOF'
Usage: bash reproduce.sh --experiment NAME --data-dir PATH [options]

Re-run one experiment (seed=42, 2 folds) and compare its AUPRC to a saved reference.

Required:
  --experiment NAME   One of: xgboost | mlp | temporal | temporal_graph | fusion
  --data-dir PATH     CERT dataset directory.

Options:
  --version VER       CERT version (default: r4.2).
  --store-dir PATH    Feature store directory (default: feature_store).
  --output-dir PATH   Where reproduce runs/references live (default: reproduce_results).
  -h, --help          Show this help and exit.

Outcomes:
  REPRODUCIBLE ✓   AUPRC matches the reference within +/- 0.001.
  MISMATCH ✗       AUPRC differs from the reference (both values printed).
  (first run)        No reference yet -> this run is saved as the reference.

Examples:
  bash reproduce.sh --experiment xgboost  --data-dir /data/cert_r4.2 --version r4.2
  bash reproduce.sh --experiment temporal --data-dir /data/cert_r4.2 --version r4.2
  bash reproduce.sh --experiment fusion   --data-dir /data/cert_r4.2 --version r4.2
EOF
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --version)    VERSION="$2"; shift 2 ;;
        --store-dir)  STORE_DIR="$2"; shift 2 ;;
        --output-dir) REPRO_DIR="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "$EXPERIMENT" ] || [ -z "$DATA_DIR" ]; then
    echo "ERROR: --experiment and --data-dir are required." >&2
    usage >&2
    exit 1
fi

# ── Map experiment -> module + how folds/seeds are passed ─────────────────────
case "$EXPERIMENT" in
    xgboost)        MODULE="innersight.scripts.xgboost_baseline";   KIND="cli" ;;
    mlp)            MODULE="innersight.scripts.train_mlp_baseline"; KIND="cli" ;;
    temporal)       MODULE="innersight.scripts.train_temporal";     KIND="config" ;;
    temporal_graph) MODULE="innersight.scripts.train_temporal_graph"; KIND="config" ;;
    fusion)         MODULE="innersight.scripts.train_fusion";       KIND="fusion" ;;
    *)
        echo "ERROR: unknown experiment '$EXPERIMENT'." >&2
        echo "Choose one of: xgboost | mlp | temporal | temporal_graph | fusion" >&2
        exit 1 ;;
esac

# ── Setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # innersight/scripts -> innersight -> backend
cd "$BACKEND_DIR" || { echo "ERROR: cannot cd to $BACKEND_DIR" >&2; exit 1; }

PYTHON="python"
if [ -x "$BACKEND_DIR/.venv/bin/python" ]; then
    PYTHON="$BACKEND_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
fi
export PYTHONHASHSEED=0   # extra determinism for hash-ordered structures

mkdir -p "$REPRO_DIR" "$REPRO_DIR/checkpoints"
RUN_JSON="$REPRO_DIR/${EXPERIMENT}_run.json"
REF_JSON="$REPRO_DIR/${EXPERIMENT}_reference.json"

# Minimal, fixed config for the config-driven scripts (seed=42, 2 folds).
REPRO_CFG="$REPRO_DIR/reproduce_config.yaml"
cat > "$REPRO_CFG" <<'EOF'
training:
  max_epochs: 5
  warmup_epochs: 1
  patience: 3
evaluation:
  n_folds: 2
  seeds: [42]
EOF

# ── Build the run command ─────────────────────────────────────────────────────
CMD=("$PYTHON" -m "$MODULE" \
     --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$STORE_DIR" \
     --output "$RUN_JSON")
case "$KIND" in
    cli)
        CMD=("${CMD[@]}" --n-folds 2 --seeds 42) ;;
    config)
        CMD=("${CMD[@]}" --checkpoint-dir "$REPRO_DIR/checkpoints" --config "$REPRO_CFG") ;;
    fusion)
        CMD=("${CMD[@]}" --checkpoint-dir "$REPRO_DIR/checkpoints" --config "$REPRO_CFG" \
             --baseline-results-dir "$REPRO_DIR") ;;
esac

extract_auprc() {
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['cross_validation']['mean']['auprc'])" "$1"
}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Reproducing '$EXPERIMENT' ($VERSION, seed=42, 2 folds) ..."
if ! "${CMD[@]}"; then
    echo "ERROR: training run failed for '$EXPERIMENT'." >&2
    exit 1
fi
if [ ! -f "$RUN_JSON" ]; then
    echo "ERROR: no results JSON produced at $RUN_JSON." >&2
    exit 1
fi

RUN_AUPRC="$(extract_auprc "$RUN_JSON")"
if [ -z "$RUN_AUPRC" ]; then
    echo "ERROR: could not read AUPRC from $RUN_JSON." >&2
    exit 1
fi

# ── Compare to reference ──────────────────────────────────────────────────────
if [ ! -f "$REF_JSON" ]; then
    cp "$RUN_JSON" "$REF_JSON"
    echo "No reference — saving this run as the reference (AUPRC=$RUN_AUPRC) -> $REF_JSON"
    exit 0
fi

REF_AUPRC="$(extract_auprc "$REF_JSON")"
MATCH="$("$PYTHON" -c "import sys; r,f,t=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); print(1 if abs(r-f)<=t else 0)" "$RUN_AUPRC" "$REF_AUPRC" "$TOLERANCE")"

if [ "$MATCH" = "1" ]; then
    echo "REPRODUCIBLE ✓  $EXPERIMENT AUPRC=$RUN_AUPRC (reference=$REF_AUPRC, tol=+/-$TOLERANCE)"
    exit 0
else
    echo "MISMATCH ✗  $EXPERIMENT this run AUPRC=$RUN_AUPRC vs reference=$REF_AUPRC (tol=+/-$TOLERANCE)"
    exit 1
fi
