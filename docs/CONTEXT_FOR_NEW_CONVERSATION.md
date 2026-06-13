# InnerSight UEBA — Project Context (Post Phase 7)

> **How to use this file:** Upload this to any new Claude conversation alongside
> the relevant phase doc from `docs/`. This file covers the full project state.

---

## 1. What This Project Is

InnerSight is a cybersecurity platform that detects insider threats using machine learning on the CMU CERT Insider Threat Dataset. It's a **solo university thesis project** graded on code quality, working system, ML results, and presentation.

The teacher runs it via `docker-compose up` — must work perfectly with bundled synthetic demo data.

---

## 2. Current State (Post Phase 7, Pre-Remote-Block)

**Phases 0-7 are COMPLETE.** 79 commits, 202 tests passing, 50 Python source files.

The ML pipeline is fully built but NOT yet trained on real data. The system runs in demo mode with synthetic data (5 users, 1 insider).

### What's done:
- **Phase 0:** Codebase cleanup, Docker, reproducibility, literature review (18 commits)
- **Phase 1:** Universal data pipeline — 6 adapters for all 10 CERT versions (11 commits)
- **Structural cleanup:** Refactored to `backend/innersight/` as package root, `from innersight.X` imports (3 commits)
- **Phase 2:** Per-user EMA baselines, z-scored deviations, 28-day windowed dataset (7 commits)
- **Phase 3:** XGBoost + MLP baselines, FocalLoss, evaluation harness (AUPRC, P@k, per-scenario, detection latency) (7 commits)
- **Phase 4:** Temporal CNN — 4-layer dilated causal Conv1d, attention pooling, 128-dim embeddings (6 commits)
- **Phase 5:** GATv2Conv graph encoder with edge features, Qdrant integration, k-NN API endpoint (8 commits)
- **Phase 6:** Fusion model (272-dim), VM training scripts, cross-version eval, embedding extraction, reproduce.sh (8 commits)
- **Phase 7:** Frontend (SuspectDiscovery, ModelComparison pages, DeviationHeatmap, AttentionTimeline, GraphNeighborhood components), Docker demo mode, Makefile, README, synthetic demo dataset (9 commits)
- **Cleanup:** Removed 18 legacy files (SAGEConv, Node2Vec), added FeatureVector to schema

### What's next:
- **Remote Block:** Rent a GPU on vast.ai, train all models on real CERT r4.2 data (~$4-8, 6-8 hours)
- **Phase 8:** Load real embeddings into Qdrant, verify full system with trained models
- **Phase 9:** Thesis presentation materials (UMAP viz, error analysis, cost model, figures)

---

## 3. Project Structure

```
innersight/                          (repo root)
├── backend/
│   ├── innersight/                  (Python package — all source code)
│   │   ├── config.py, schema.py, api.py
│   │   ├── data/                    (pipeline, adapters, answers, loaders, feature_store)
│   │   ├── features/               (18 daily features + 129 temporal features)
│   │   ├── models/                  (baseline, temporal_encoder, graph_encoder, fusion_model, losses, dataset, mlp, graph_builder, graph_schema)
│   │   ├── training/               (trainer, evaluation, cross_version)
│   │   ├── scoring/                (scoring, suspect_discovery)
│   │   ├── feedback/               (feedback)
│   │   ├── utils/                  (io, reproducibility)
│   │   ├── scripts/                (compute_baselines, validate_baselines, xgboost_baseline, train_mlp_baseline, train_temporal, visualize_attention, evaluate_temporal, train_temporal_graph, evaluate_graph, train_fusion, eval_cross_version, extract_embeddings, sync_embeddings, compare_baselines, run_all_training.sh, reproduce.sh, download_checkpoints.sh)
│   │   └── configs/                (train_baseline_mlp.yaml, train_temporal.yaml, train_temporal_graph.yaml, train_fusion.yaml)
│   ├── tests/                      (29 test files, 213 test functions, 202 passing)
│   ├── pyproject.toml, Dockerfile, docker-entrypoint.sh
│   └── checkpoints/                (gitignored — model weights go here)
├── frontend/
│   └── src/                        (React 19, TypeScript, Redux, styled-components, Recharts, d3-force)
│       ├── pages/                  (Alerts, Employees, Investigation, Training, SuspectDiscovery, ModelComparison, NotFound)
│       └── components/             (Navbar, ErrorBoundary, Spinner, DeviationHeatmap, AttentionTimeline, GraphNeighborhood)
├── data/synthetic_demo/            (bundled demo dataset — 5 users, 1 insider, r4.2 format)
├── docs/                           (phase planning docs, DATASET_CONTEXT.md, literature_targets.md)
├── docker-compose.yml, Makefile, README.md, .github/workflows/ci.yml
└── .gitignore, .env, .env.example
```

---

## 4. Import Convention

All imports use `from innersight.X import Y` (NOT `from innersight.backend.X`).

The package is at `backend/innersight/`. Install with `cd backend && pip install -e .`.

---

## 5. The Neural Network Architecture (4 Modules)

**Module 1: Per-User Behavioral Baseline** (no training)
- EMA(α=0.05), std floor, role-cohort cold-start
- Converts raw features to z-scored deviations per user per day

**Module 2: Temporal Pattern Encoder** (supervised)
- 4-layer dilated causal CNN (dilations 1,2,4,8), kernel=3
- LayerNorm (NOT BatchNorm), residual connections, dropout 0.3
- Attention-weighted pooling → 128-dim embedding
- Input: (batch, 18, 28) → Output: (batch, 128)

**Module 3: Graph Context Encoder** (supervised)
- GATv2Conv with edge features, 4 heads, 2 layers, full-batch
- User node features = temporal embeddings (128-dim)
- Output: 128-dim relational embedding per user

**Module 4: Fusion + Classification Head** (supervised)
- Concat: temporal(128) + graph(128) + static(16) = 272 dims
- Static = OCEAN(5) + role_emb(8) + dept_emb(3)
- MLP: 272→128→64→1 with dropout 0.4
- FocalLoss(α=0.75, γ=2.0), AUPRC primary metric

### Progressive Model Ladder:
1. MLP on z-scored deviations → AUPRC 0.15-0.25
2. Temporal CNN → AUPRC 0.35-0.50 (biggest jump)
3. +GATv2 Graph → AUPRC 0.45-0.55
4. Full Fusion → AUPRC 0.50-0.60

---

## 6. Key Technical Details

- **Python 3.13**, PyTorch, torch-geometric, XGBoost, Qdrant, Flask, React 19
- **Dataset:** CERT r4.2 (1000 users, 70 insiders, 3 attack scenarios, ~16GB)
- **Evaluation:** AUPRC primary, temporal stratified 5-fold CV, 3 seeds, user-level splits
- **Window:** 28 days, stride 7, overlap-ratio labeling (≥50% = positive, 1-49% = excluded)
- **Docker:** 3 services (backend Flask:5001, frontend nginx:3000, qdrant:6333)
- **Demo mode:** No env vars → falls back to bundled synthetic data
- **GitHub:** https://github.com/Enterino8465/innersight.git

---

## 7. What the Remote Block Does

Rent a GPU on vast.ai (~$4-8), train all 6 ladder steps on real CERT r4.2 with 3 seeds each using `run_all_training.sh`. Download checkpoints, embeddings, and results JSONs. Then Phase 8 loads them into the local system.

---

## 8. Coding Standards

- **python-standards:** Type every signature (`X | None`), pathlib over os.path, logging over print, frozen dataclasses, explicit errors
- **ml-training:** seed_everything(), device detection via get_device(), torch.load with weights_only=True, shape comments on tensors, NaN detection
- All new code uses `from __future__ import annotations`

---

## 9. Documents Available in docs/

1. `DATASET_CONTEXT.md` — Complete CERT dataset schemas, quirks, adapter fingerprints
2. `literature_targets.md` — 10 papers, AUPRC targets
3. `Phase1_Universal_Data_Pipeline.html` through `Phase7_Frontend_Docker_Demo.html`
4. `RemoteBlock_GPU_Training.html` — GPU training runbook
5. `Phase0_Remaining_Tasks.html/.md` — historical (Phase 0 planning)
