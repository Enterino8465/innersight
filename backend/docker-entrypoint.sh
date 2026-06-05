#!/usr/bin/env bash
#
# Container entrypoint. Prepares state, then execs the given command (Flask).
#
#   * If checkpoints / cached deviations already exist  -> production mode (skip).
#   * Otherwise                                         -> compute baselines on the
#     bundled synthetic demo data so the UI works out of the box.
#
set -e

MODEL_DIR="${INNERSIGHT_MODEL_DIR:-/models}"
DATA_DIR="${INNERSIGHT_DATA_DIR:-/app/demo_data}"
VERSION="${INNERSIGHT_DEMO_VERSION:-r4.2}"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/best_model.pt" ] || [ -f "$MODEL_DIR/$VERSION/deviations.parquet" ]; then
    echo "[entrypoint] Checkpoints found in $MODEL_DIR — starting in PRODUCTION mode."
else
    echo "[entrypoint] No checkpoints — computing baselines on demo data ($DATA_DIR) ..."
    python -m innersight.scripts.compute_baselines \
        --version "$VERSION" --data-dir "$DATA_DIR" --store-dir "$MODEL_DIR" \
        || echo "[entrypoint] WARN: baseline computation failed; the API will retry in demo mode."
fi

echo "[entrypoint] Starting: $*"
exec "$@"
