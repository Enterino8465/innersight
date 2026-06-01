# InnerSight UEBA — Phase 0: Remaining Infrastructure Tasks

**Task Guide with Claude Code Prompts**

---

## Overview

Phase 0 Tasks 1–9 (codebase cleanup) are already complete and committed. However, **six critical infrastructure tasks** from the Implementation Plan v4 were never executed:

- **Proper Python packaging** (pyproject.toml) — eliminates the sys.path hacking plague (30+ occurrences)
- **Docker setup** — teacher runs the system via `docker-compose up`, must work from day one
- **Reproducibility infrastructure** — seed fixing, pinned deps, deterministic results
- **CI/CD fix** — current workflows reference non-existent directories and fail on every push
- **Checkpoint cleanup** — 441MB of `.pt` files tracked in git, should be in `.gitignore`
- **Literature review** — collect AUPRC target numbers BEFORE training

This guide covers these **7 remaining tasks**. Each is atomic and ends with a git commit.

**Execution method:** Copy the prompt from each task and paste it into Claude Code. After each task, verify the commit was made before moving to the next.

**Important:** Execute tasks IN ORDER. Task 1 (packaging) must be done first because it changes how imports work. Task 2 (sys.path cleanup) depends on Task 1.

---

## Task Summary

| # | Task | Git Commit |
|---|------|-----------|
| 1 | Python Package Setup (pyproject.toml) | Phase 0.10 |
| 2 | Eliminate sys.path Hacking | Phase 0.11 |
| 3 | Fix CI/CD Workflows | Phase 0.12 |
| 4 | Docker + docker-compose Setup | Phase 0.13 |
| 5 | Reproducibility Infrastructure | Phase 0.14 |
| 6 | Remove Checkpoints from Git Tracking | Phase 0.15 |
| 7 | Literature Review + Target Numbers | Phase 0.16 |

---

## Task 1: Python Package Setup (pyproject.toml)

### The Problem

The backend has no `pyproject.toml` or `setup.py`. This means the package cannot be installed with `pip install -e .`, so every script resorts to `sys.path` hacking to find `innersight.backend` modules. There are **30+ `sys.path.insert()` calls** scattered across `models/`, `scripts/`, `tests/`, and `scoring/`. This is fragile, ugly, and causes IDE confusion. The senior-python skill explicitly forbids this pattern.

The fix is simple: add a `pyproject.toml` that declares `innersight` as an installable package. After `pip install -e .` in the venv, all imports resolve naturally without any path manipulation.

### Steps

1. Create `backend/pyproject.toml` with build metadata, dependencies from requirements.txt, and the innersight package declaration
2. Ensure the proper package structure exists: `innersight/__init__.py` and `innersight/backend/__init__.py` at the right levels (they already exist)
3. Verify the package installs: `cd backend && pip install -e .`
4. Verify imports work: `python -c "from innersight.backend.config import DATA_DIR; print('OK')"`
5. Run the full test suite to confirm nothing broke
6. Git commit: `Phase 0.10: Add pyproject.toml for proper package installation`

### How This Achieves the Goal

A single `pip install -e .` makes the entire `innersight` package importable from anywhere. This eliminates the root cause of sys.path hacking and enables Task 2 (cleanup of all those hacks). It also enables proper tooling: mypy, ruff, pytest all discover the package correctly.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Create a pyproject.toml so the backend is a proper installable Python package.
This is Phase 0, Task 1 (continuing from the existing Phase 0.9 commits).

CONTEXT:
- The project root has innersight/ with __init__.py already
- innersight/backend/ has __init__.py already
- All imports use 'from innersight.backend.X import Y'
- requirements.txt exists at backend/requirements.txt with pinned versions
- There is NO pyproject.toml, setup.py, or setup.cfg anywhere
- Python 3.13 is being used

CREATE backend/pyproject.toml:

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "innersight"
version = "0.1.0"
description = "InnerSight UEBA - Insider Threat Detection"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "torch>=2.0",
    "pandas>=2.0",
    "scikit-learn>=1.5",
    "flask>=3.0",
    "flask-cors>=5.0",
    "pyyaml",
    "matplotlib",
    "networkx",
    "torch-geometric",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.10"]

[tool.setuptools.packages.find]
where = [".."]  # look one level up from pyproject.toml
include = ["innersight*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".."]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

VERIFY:
cd backend && pip install -e . 2>&1 | tail -5
python -c "from innersight.backend.config import DATA_DIR; print('Package OK')"
python -c "from innersight.backend.models.mlp import InsiderThreatMLP; print('Models OK')"
python -m pytest tests/ -x --tb=short

Git commit: git add -A && git commit -m "Phase 0.10: Add pyproject.toml for proper package installation"
```

---

## Task 2: Eliminate sys.path Hacking

### The Problem

With `pyproject.toml` in place (Task 1), `sys.path` manipulation is no longer needed. But 30+ files still have it. Every `models/*.py` file, every `scripts/*.py` file, and even some test files include 3-6 lines of boilerplate computing `_FILE_DIR`, `_BACKEND`, `_PKG_ROOT` and inserting them into `sys.path`. This clutters every file, violates the senior-python skill's "no global mutable state" rule, and confuses IDEs.

### Steps

1. Remove all sys.path hacking blocks from `backend/models/*.py` (graphsage.py, graph_builder.py, gnn_trainer.py, node2vec_trainer.py, embeddings.py, graph_loader.py, dataset.py, mlp.py)
2. Remove all sys.path hacking from `backend/scripts/*.py` (train.py, train_embeddings.py, extract_embeddings.py, visualize_embeddings.py, compare_models.py, validate_graphs.py)
3. Remove sys.path hacking from `backend/scoring/scoring.py`
4. Remove sys.path hacking from `backend/tests/test_api.py` and `smoke_test_api.py`
5. Leave `backend/learning/*.py` alone (archived, not production code)
6. Remove any now-unused `import sys` and `import os` statements that only existed for path manipulation
7. Run the full test suite: `python -m pytest tests/ -x`
8. Git commit: `Phase 0.11: Eliminate sys.path hacking (package is pip-installed)`

### How This Achieves the Goal

Every production `.py` file is cleaned of the 3-6 lines of `sys.path` boilerplate. Imports work because the package is installed, not because of runtime path manipulation. The codebase looks professional and matches senior-python standards.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Remove ALL sys.path hacking from production code. This is Phase 0, Task 2.

PREREQUISITE: pyproject.toml exists and 'pip install -e .' was run (Phase 0.10).
All 'from innersight.backend.X import Y' imports now resolve without path tricks.

Find all affected files:
grep -rn 'sys.path' backend/ --include='*.py' | grep -v learning/

For EACH file found (except backend/learning/*.py which are archived):
1. Delete the entire sys.path manipulation block. This typically looks like:

   _FILE_DIR = os.path.abspath(os.path.dirname(__file__))
   _BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
   _PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
   for _p in (_PKG_ROOT, _BACKEND):
       if _p not in sys.path:
           sys.path.insert(0, _p)

   Or simpler variants like:
   _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
   if _root not in sys.path:
       sys.path.insert(0, _root)

2. Remove 'import sys' if it's ONLY used for sys.path (check if sys is used elsewhere first)
3. Remove 'import os' ONLY if it's solely used for path computation AND nothing else uses os
4. Keep all actual imports (from innersight.backend.X import Y) untouched

FILES TO CLEAN (non-exhaustive, grep will find them all):
- backend/models/graphsage.py
- backend/models/graph_builder.py
- backend/models/gnn_trainer.py
- backend/models/node2vec_trainer.py
- backend/models/embeddings.py
- backend/models/graph_loader.py
- backend/models/dataset.py
- backend/models/mlp.py
- backend/scripts/train.py
- backend/scripts/train_embeddings.py
- backend/scripts/extract_embeddings.py
- backend/scripts/visualize_embeddings.py
- backend/scripts/compare_models.py
- backend/scripts/validate_graphs.py
- backend/scoring/scoring.py
- backend/tests/test_api.py
- backend/tests/smoke_test_api.py

DO NOT touch backend/learning/*.py (archived learning scripts).

After cleanup, verify no sys.path remains in production code:
grep -rn 'sys.path' backend/ --include='*.py' | grep -v learning/
(should return nothing)

Run tests: cd backend && python -m pytest tests/ -x --tb=short

Git commit: git commit -am "Phase 0.11: Eliminate sys.path hacking (package is pip-installed)"
```

---

## Task 3: Fix CI/CD Workflows

### The Problem

The `.github/workflows/ci.yml` references `apps/api` and `apps/worker` directories that **DO NOT EXIST**. It uses `uv` (not pip), references `innersight_api` and `innersight_worker` packages that were never created, and tries to build Dockerfiles that don't exist yet. The `security.yml` is an empty stub (only comments, no actual jobs). Every push to GitHub triggers a failing CI run.

### Steps

1. Rewrite `.github/workflows/ci.yml` to match the actual project structure (`backend/` with pytest, ruff)
2. Add lint job: `ruff check` + `ruff format --check` on `backend/`
3. Add test job: `pytest backend/tests/` with Python 3.13
4. Remove `.github/workflows/security.yml` (empty stub)
5. Git commit: `Phase 0.12: Fix CI/CD workflows to match actual project structure`

### How This Achieves the Goal

CI actually runs on push. Lint errors are caught. Tests run automatically. The project looks maintained, not abandoned. A teacher seeing green CI badges immediately has confidence in code quality.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Fix the CI/CD workflows. This is Phase 0, Task 3.

PROBLEM: .github/workflows/ci.yml references 'apps/api' and 'apps/worker' which
DON'T EXIST. It uses 'uv' and references 'innersight_api' / 'innersight_worker'
packages that were never created. security.yml is an empty stub (just comments).

REWRITE .github/workflows/ci.yml to:

name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install ruff
        run: pip install ruff
      - name: Check
        run: ruff check backend/ --exclude backend/learning/
      - name: Format
        run: ruff format --check backend/ --exclude backend/learning/

  test:
    name: Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install
        working-directory: backend
        run: |
          pip install -e ".[dev]"
      - name: Test
        working-directory: backend
        run: python -m pytest tests/ -x --tb=short

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build backend
        run: docker build -f backend/Dockerfile .
      - name: Build frontend
        run: docker build -f frontend/Dockerfile frontend/

DELETE .github/workflows/security.yml (it's an empty stub with only comments).

Git commit: git add -A && git commit -m "Phase 0.12: Fix CI/CD workflows to match actual project structure"
```

---

## Task 4: Docker + docker-compose Setup

### The Problem

The teacher grades by running `docker-compose up` on their machine. There is currently **NO Dockerfile** and **NO docker-compose.yml** anywhere in the project. The implementation plan requires Docker from day one to prevent environment drift between phases. The system needs 3 services: backend (Flask API), frontend (React/Vite served by nginx), and qdrant (vector database for Phase 5+).

### Steps

1. Create `backend/Dockerfile` (Python 3.13-slim, install deps, expose 5001)
2. Create `frontend/Dockerfile` (Node 22, npm install, vite build, nginx serve)
3. Create `frontend/nginx.conf` for API proxy and SPA routing
4. Create `docker-compose.yml` at project root with 3 services: backend, frontend, qdrant
5. Create `.dockerignore` files to exclude `.venv`, `node_modules`, `__pycache__`, `.pt` files
6. Create `.env.example` file showing required environment variables
7. Verify docker-compose config is valid: `docker-compose config`
8. Git commit: `Phase 0.13: Add Docker and docker-compose setup`

### How This Achieves the Goal

The teacher can run `docker-compose up` and get the full system. Backend, frontend, and Qdrant all start with health checks and proper ordering. No Python/Node version conflicts. Environment drift between development phases becomes impossible.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Create Docker infrastructure. This is Phase 0, Task 4.

CREATE backend/Dockerfile:

FROM python:3.13-slim

WORKDIR /app

# System deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copy package definition and install
COPY pyproject.toml .
COPY ../innersight/ ./innersight/
RUN pip install --no-cache-dir -e .

# Copy backend code
COPY . .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/status')" || exit 1

CMD ["flask", "--app", "innersight.backend.api", "run", "--host", "0.0.0.0", "--port", "5001"]

---

CREATE frontend/Dockerfile:

# Build stage
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

---

CREATE frontend/nginx.conf:

server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://backend:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}

---

CREATE docker-compose.yml (project root):

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "5001:5001"
    environment:
      - INNERSIGHT_DATA_DIR=/data
      - INNERSIGHT_MODEL_DIR=/models
    volumes:
      - ./backend/checkpoints:/models
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  qdrant_data:

---

CREATE .env.example (project root):

# Required: path to CERT dataset (any version)
INNERSIGHT_DATA_DIR=/path/to/cert/r4.2

# Optional: override model checkpoint directory
INNERSIGHT_MODEL_DIR=backend/checkpoints

---

CREATE backend/.dockerignore:

.venv/
__pycache__/
*.pyc
*.pt
*.pth
.pytest_cache/
learning/
outputs/

---

CREATE frontend/.dockerignore:

node_modules/
dist/
.env

---

VERIFY: docker-compose config (syntax check only, don't try to build yet)

Git commit: git add -A && git commit -m "Phase 0.13: Add Docker and docker-compose setup"
```

---

## Task 5: Reproducibility Infrastructure

### The Problem

The implementation plan requires reproducibility from day one: every experiment must be reproducible with one command. Currently there is no seed-fixing utility, no mechanism to ensure deterministic results, and `requirements.txt` has some unpinned dependencies (`torch-geometric`, `pyyaml`, `matplotlib`, `networkx`, `umap-learn`). The council specifically mandated multiple seeds per experiment (3-5 runs) for statistical significance — this requires a robust seeding mechanism.

### Steps

1. Create `backend/innersight/backend/utils/reproducibility.py` with `seed_everything()` function
2. Create `backend/innersight/backend/utils/__init__.py` exporting the utility
3. Pin ALL dependencies in `requirements.txt`
4. Add `seed: 42` to `DEFAULT_TRAINING_CONFIG` in `config.py`
5. Create a test for the reproducibility utility
6. Run tests to verify the new utility works
7. Git commit: `Phase 0.14: Add reproducibility infrastructure`

### How This Achieves the Goal

Every training run produces identical results given the same seed. Dependencies are fully pinned so the environment is byte-for-byte reproducible. The `seed_everything()` function handles Python, NumPy, PyTorch, and CUDA seeds in one call. This is foundational for the progressive model ladder — without reproducibility, comparing model steps is meaningless.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Add reproducibility infrastructure. This is Phase 0, Task 5.

CREATE backend/innersight/backend/utils/__init__.py:

"""Shared utilities for InnerSight UEBA."""
from innersight.backend.utils.reproducibility import seed_everything

__all__ = ["seed_everything"]

---

CREATE backend/innersight/backend/utils/reproducibility.py:

"""Reproducibility utilities for deterministic training.

Public API:
    seed_everything(seed) - Fix all random seeds for full reproducibility.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Fixes seeds for: Python stdlib random, NumPy, PyTorch (CPU + CUDA).
    Also sets deterministic algorithm flags for CUDA reproducibility.

    Args:
        seed: Integer seed value. Default 42.

    Note:
        Deterministic mode may reduce performance by 10-20% on GPU.
        This is acceptable for research; disable for production inference.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (reproducibility > speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ deterministic mode
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Not all ops have deterministic implementations

    logger.info("Seeded all RNGs with seed=%d", seed)

---

UPDATE backend/config.py DEFAULT_TRAINING_CONFIG:
Add 'seed': 42 to the dict.

---

UPDATE backend/requirements.txt - pin all versions. Replace unpinned entries:
- torch-geometric==2.6.1
- pyyaml==6.0.2
- matplotlib==3.10.3
- networkx==3.5
- umap-learn==0.5.7

---

CREATE backend/tests/test_reproducibility.py:

"""Tests for reproducibility utilities."""

import torch

from innersight.backend.utils.reproducibility import seed_everything


def test_seed_produces_identical_tensors() -> None:
    """Same seed must produce identical random tensors."""
    seed_everything(123)
    a = torch.randn(10)
    seed_everything(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_tensors() -> None:
    """Different seeds must produce different random tensors."""
    seed_everything(1)
    a = torch.randn(10)
    seed_everything(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)

---

Run tests: cd backend && python -m pytest tests/test_reproducibility.py -v
Then full suite: python -m pytest tests/ -x --tb=short

Git commit: git add -A && git commit -m "Phase 0.14: Add reproducibility infrastructure"
```

---

## Task 6: Remove Checkpoints from Git Tracking

### The Problem

The `backend/checkpoints/` directory contains **441MB of `.pt` files** tracked by git. This makes cloning painfully slow and bloats the repo permanently. These are trained model weights that should be distributed via GitHub Releases (with a download script), not tracked in version control. The `.pt` graph files (`train_graph.pt`, `val_graph.pt`, `test_graph.pt`) are also huge and regeneratable from raw data.

### Steps

1. Add `*.pt` and `*.pth` to `.gitignore`
2. Remove all `.pt` files from git tracking: `git rm --cached`
3. Create `.gitkeep` files to preserve directory structure
4. Create `scripts/download_checkpoints.sh` placeholder
5. Git commit: `Phase 0.15: Remove .pt checkpoints from git tracking`

### How This Achieves the Goal

The repo shrinks from ~450MB to ~5MB on next clone. Model checkpoints are managed separately via GitHub Releases. The `.gitignore` prevents accidentally committing large binaries again.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Remove .pt checkpoint files from git tracking. This is Phase 0, Task 6.

CONTEXT: backend/checkpoints/ contains 441MB of .pt files. These are trained model
weights and precomputed graphs. They should NOT be in version control.

STEP 1 - Update .gitignore (root level). Add these lines at the end:

# Model checkpoints (distributed via GitHub Releases)
*.pt
*.pth
*.onnx

# Large data artifacts
*.parquet
*.feather
*.npz

STEP 2 - Remove .pt files from git tracking (keeps files on disk):
Find and remove all tracked .pt files:
git ls-files '*.pt' | xargs git rm --cached

STEP 3 - Create .gitkeep files to preserve directory structure:
touch backend/checkpoints/.gitkeep
touch backend/checkpoints/flat/.gitkeep
touch backend/checkpoints/n2v/.gitkeep
touch backend/checkpoints/graphsage/.gitkeep
touch backend/checkpoints/graphsage/graphs/.gitkeep
touch backend/checkpoints/graphs/.gitkeep
touch backend/checkpoints/m2v/.gitkeep
touch backend/checkpoints/m2v/graphs/.gitkeep
touch backend/innersight/data/graphs/.gitkeep

STEP 4 - Create scripts/download_checkpoints.sh:

#!/usr/bin/env bash
# Download pre-trained checkpoints from GitHub Releases.
# Usage: ./scripts/download_checkpoints.sh
#
# This script will be implemented in Phase 7 when checkpoints are
# uploaded to a GitHub Release. For now it's a placeholder.

set -e
echo "[InnerSight] Checkpoint download script"
echo "TODO: Implement after first real training run (Phase 4+)"
echo "For now, train locally or use the synthetic demo dataset."

---

mkdir -p scripts
chmod +x scripts/download_checkpoints.sh

STEP 5 - Verify .pt files are untracked:
git ls-files '*.pt'
(should return nothing)

Git commit: git add -A && git commit -m "Phase 0.15: Remove .pt checkpoints from git tracking"
```

---

## Task 7: Literature Review + Target Numbers

### The Problem

The council mandated: **collect literature target numbers BEFORE training**. Without knowing what published papers achieve on the CERT dataset, you can't tell if your model is good or broken. The implementation plan says: "Read 3-4 published papers on CERT insider threat detection. Record their AUPRC/AUC numbers and methods. Know your target BEFORE building models."

This task is research, not code. The output is a markdown file that serves as a reference during all subsequent phases.

### Steps

1. Search for published papers on the CERT insider threat dataset (CMU CERT r4.2, r5.2)
2. Find 4-6 papers with reported metrics (AUPRC, AUC-ROC, F1, precision@k)
3. Record: paper title, method, dataset version, metric, score, evaluation methodology
4. Note which papers use honest temporal evaluation vs. random splits (random splits leak future info)
5. Create `docs/literature_targets.md` with a comparison table
6. Git commit: `Phase 0.16: Literature review with target numbers`

### How This Achieves the Goal

When your temporal CNN achieves AUPRC 0.40, you can immediately say whether that's competitive or not. Without this reference, you're flying blind. The literature targets also inform realistic expectations for the thesis presentation.

### Claude Code Prompt

*Copy everything below the line and paste it as a single message to Claude Code:*

---

```
You are working on the InnerSight UEBA project at /Users/michaelkuksov/Developer/innersight.

TASK: Create a literature review of CERT insider threat detection papers.
This is Phase 0, Task 7.

RESEARCH these papers (search the web for each):

1. Lindauer et al. (2020) - "Generating Test Data for Insider Threat Detectors"
   (CMU CERT team's own paper describing the dataset and baselines)

2. Yuan et al. (2019) - "Deep Learning for Insider Threat Detection"
   or Yuan & Wu (2021) "Insider Threat Detection with Deep Neural Network"

3. Le & Zincir-Heywood (2019) - "Evaluating Insider Threat Detection Workflow
   Using Supervised and Unsupervised Learning"

4. Chattopadhyay et al. (2018) - "Scenario-based Insider Threat Detection"

5. Liu et al. (2019/2020) - Any paper using GNN/deep learning on CERT

6. Any additional papers found that report metrics on CERT r4.2 or r5.2

For each paper, extract:
- Full citation (authors, year, title, venue)
- CERT version used (r4.2? r5.2? which scenarios?)
- Method (what model architecture, what features)
- Evaluation methodology (temporal split? random? k-fold? user-based?)
- Key metric: AUPRC if reported, else AUC-ROC, else F1
- Any caveats (e.g., random splits inflate results, only tested on scenario 1)

CREATE docs/literature_targets.md with this structure:

# Literature Review: CERT Insider Threat Detection

## Summary Table

| Paper | Year | Method | Dataset | Split Type | AUPRC | AUC-ROC | Notes |
|-------|------|--------|---------|-----------|-------|---------|-------|
| ... | ... | ... | ... | ... | ... | ... | ... |

## Our Targets (honest temporal evaluation on r4.2)

Based on the literature and architectural constraints:
- MLP baseline on z-scored deviations: AUPRC 0.15-0.25
- Temporal CNN on deviation sequences: AUPRC 0.35-0.50 (biggest jump)
- Temporal CNN + GATv2 graph: AUPRC 0.45-0.55
- Full fusion model: AUPRC 0.50-0.60
- Realistic ceiling: 0.55-0.65 (limited by 70 positive examples, not architecture)

## Key Insight

Many papers report AUC-ROC > 0.95 but use random splits or window-level
metrics that don't reflect real-world detection capability. Our evaluation uses
honest temporal splits where the model never sees future data. This makes our
numbers LOWER but HONEST — which is what a thesis committee respects.

Papers using random train/test splits will show inflated metrics because:
1. Future behavioral patterns leak into training data
2. The same user can appear in both train and test (identity leakage)
3. Window overlap means near-identical samples in train and test

## Detailed Paper Notes

(one section per paper with full analysis)

---

mkdir -p docs

Git commit: git add -A && git commit -m "Phase 0.16: Literature review with target numbers"
```

---

## After Phase 0

When all 7 tasks are complete, the project will have:

- **Proper Python packaging** — `pip install -e .` makes all imports work cleanly
- **Zero sys.path hacking** in production code
- **Working CI/CD** that runs lint + tests on every push
- **Docker infrastructure** ready for `docker-compose up`
- **Reproducibility from day one** — `seed_everything()` called in every training script
- **Clean git without 441MB of binary checkpoints**
- **Literature targets** to benchmark against during Phases 3-6

**Next step:** Phase 1 builds the universal data pipeline with 6 adapters so any CERT version (r1 through r6.2) loads through one interface. The feature store (Parquet persistence) ensures downstream modules never recompute features.
