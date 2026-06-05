# InnerSight UEBA

InnerSight is an insider-threat detection system that learns a per-user behavioral
baseline and flags deviations from it. It combines four signals — statistical
per-user baselines, a temporal CNN over daily activity, a graph neural network over
the user/device/file/URL interaction graph, and vector search for finding
behaviorally similar users — into a single risk score per employee.

## Quick Start

Prerequisites: Docker, 4GB+ RAM.

```bash
make demo
# Open http://localhost:3000
```

Expected startup time: ~30 seconds. The stack runs on bundled synthetic demo data
out of the box — no dataset download required.

## What You Should See

- **Alerts** — threat alerts sorted by risk score, with one-click triage actions.
- **Employees** — directory of all users with their current risk levels.
- **Investigation** — per-user deep dive: 30-day risk history, behavioral deviation
  heatmap, model attention timeline, per-feature breakdown, activity timeline, and
  graph neighborhood.
- **Suspect Discovery** — k-NN search for users behaviorally similar to a suspect,
  backed by Qdrant vector search.
- **Model Comparison** — progressive-ladder results comparing each model tier.
- **Training** — start/stop training runs and watch live metrics.

## Architecture

A four-module pipeline turns raw activity logs into per-user risk scores:

**Module 1 → Module 2 → Module 3 → Module 4**

1. **Baselines** — build per-user statistical behavioral baselines and z-score
   deviations from daily features.
2. **Temporal CNN** — model each user's activity over time to score sequence-level
   anomalies.
3. **Graph Neural Network** — propagate signal across the user/device/file/URL
   interaction graph.
4. **Vector Search** — embed user behavior and run k-NN similarity search (Qdrant)
   to surface look-alike suspects.

## Using Real CERT Data

To run against a real CERT dataset instead of the bundled synthetic data:

1. Edit `docker-compose.yml` and uncomment the volume mount under the `backend`
   service:
   ```yaml
   - /path/to/cert_r4.2:/data  # uncomment for real CERT data
   ```
2. Uncomment the matching environment line to set `INNERSIGHT_DATA_DIR`:
   ```yaml
   - INNERSIGHT_DATA_DIR=/data
   ```
3. Run `make demo`.

## Development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -x
```

## Troubleshooting

- **Port conflicts** — change the host ports in `docker-compose.yml` (defaults:
  3000 frontend, 5001 backend, 6333 Qdrant).
- **Docker memory** — allocate 4GB+ to Docker.
- **Qdrant OOM** — reduce the collection size or increase Docker memory.
- **"No module named innersight"** — run `pip install -e .` inside `backend/`.
