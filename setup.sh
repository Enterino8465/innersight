#!/usr/bin/env bash
# One command sets Stage 0 up:  bash setup.sh
# Creates .venv, installs the pinned dependencies, runs the acceptance test.
# Pass --full to also count every row in every log.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version="$("$candidate" -c 'import sys; print("%d %d" % sys.version_info[:2])')"
    major="${version%% *}"
    minor="${version##* }"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "No Python >= 3.11 found on PATH." >&2
    echo "Tried: python3.13, python3.12, python3.11, python3." >&2
    echo "Install one (brew install python@3.13) and re-run this script." >&2
    exit 1
fi

echo "==> interpreter: $PY ($("$PY" --version 2>&1))"

if [ ! -d .venv ]; then
    echo "==> creating .venv"
    "$PY" -m venv .venv
else
    echo "==> .venv already exists, reusing it"
fi

echo "==> installing dependencies (torch is large; this takes a few minutes)"
./.venv/bin/python -m pip install --upgrade pip --quiet
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "==> running verify_setup.py"
./.venv/bin/python verify_setup.py "$@"
