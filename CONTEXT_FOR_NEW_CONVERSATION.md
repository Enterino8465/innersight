# InnerSight UEBA — Complete Project Context

> **How to use this file:** Upload this to any new Claude conversation alongside
> `CERT_DATASET_REFERENCE.md` (dataset schemas/quirks) and optionally the three
> HTML architecture docs. This file covers everything EXCEPT raw dataset details.

---

## 1. What This Project Is

InnerSight is a cybersecurity platform that detects insider threats (employees stealing data, sabotaging systems, or leaking information) using machine learning on the CERT Insider Threat Dataset from Carnegie Mellon University.

The CERT dataset simulates a fake company (DTAA Corp) with 10 independent simulation runs (r1 through r6.2). Each run generates employees, org charts, and behavioral logs — then injects specific insider attack patterns. We know exactly who the insiders are and when they attacked.

**This is a solo university thesis project.** Graded on code quality, working system, ML results, and presentation. The teacher runs via `docker-compose up` on their own machine — must work perfectly.

---

## 2. The Dataset (Summary — see CERT_DATASET_REFERENCE.md for full details)

- **r4.2** (primary training): ~1,000 users, 70 insiders, 3 attack scenarios, ~16 GB
- **r5.2** (largest trainable): ~2,000 users, 99 insiders, 4 scenarios, ~38 GB
- **r6.2** (test only): ~4,000 users, 5 insiders, 5 scenarios, ~94 GB
- **Location**: External hard drive at `/Volumes/dataset hard drive/dataset/`
- **Log types**: logon.csv, device.csv, email.csv, http.csv, file.csv, psychometric.csv, LDAP/ (monthly snapshots)
- **DO NOT connect to the external drive in sandbox** — it kills the sandbox. Use `CERT_DATASET_REFERENCE.md` instead.

### The 5 Attack Scenarios
1. After-hours USB + WikiLeaks upload (quick, 1-2 days)
2. Job hunting → resignation → USB data theft (3-8 weeks)
3. Sysadmin keylogger on another user's machine (cross-machine access, 1-2 days)
4. Cross-account file snooping + email exfil over months
5. Post-layoff Dropbox exfiltration (minutes)

---

## 3. The Neural Network Architecture (Council-Approved)

**Core idea: Learn what normal looks like for each user, then detect patterns of deviation.**

### 4-Module Pipeline:

**Module 1: Per-User Behavioral Baseline (no training needed)**
- For each user, compute an Exponential Moving Average (EMA, α=0.05) of their 18 daily behavioral features
- Convert every day into z-scored deviations: `z = (today - user_mean) / max(user_std, std_floor)`
- This normalizes per-user: a sysadmin's 50 USB events/day = deviation 0; an accountant's 2 USB events = deviation 8σ
- Cold-start: new hires initialized from role-cohort statistics (hierarchical: role→dept→global)
- Std floor at 10% of global median prevents z-score explosion on ultra-consistent users
- No learned parameters — purely analytical

**Module 2: Temporal Pattern Encoder (supervised)**
- Input: 28-day sliding window of z-scored deviations (shape: num_users × 18 channels × 28 days)
- Architecture: 4-layer 1D dilated causal CNN (dilations 1, 2, 4, 8)
- Uses LayerNorm (NOT BatchNorm — BatchNorm has class-imbalance issues with 0.4% positive rate)
- Residual connections, dropout(0.3)
- Attention-weighted temporal pooling → 128-dim embedding per user
- Causal padding: left-pad only, no future information leakage
- OCEAN scores are NOT in the CNN input — they go in Module 4

**Module 3: Graph Context Encoder (supervised)**
- Heterogeneous graph: 4 node types (user, pc, url, file), 10 edge types (5 forward + 5 reverse)
- User node features = Module 2 temporal embeddings (128-dim)
- Edge features AGGREGATED per 28-day window: count_in_window, frac_after_hours, is_new_connection, max_burst_day
- GATv2Conv with edge_attr (4 attention heads, 2 layers) — NOT SAGEConv (which ignores edge features)
- Full-batch inference (no NeighborLoader — 846 nodes fits trivially in memory)
- Residual connections + LayerNorm
- Output: 128-dim relational embedding per user

**Module 4: Fusion + Classification Head (supervised)**
- Concatenate: temporal(128) + graph(128) + static(16) = 272 dims
- Static context = 5 OCEAN scores + 8-dim role embedding + 3-dim dept embedding
- MLP head: 272→128→64→1 with dropout(0.4) and ReLU
- Focal Loss (α=0.75, γ=2.0) — NOT BCEWithLogitsLoss with pos_weight
- Threshold tuned on validation PR curve (NOT fixed 0.5)
- Primary metric: AUPRC (not F1, not AUROC — those are misleading at 0.4% positive rate)

### Training Details
- AdamW, lr=1e-3, weight_decay=1e-4
- Cosine annealing with 5-epoch warmup, eta_min=1e-5
- Gradient clipping max_norm=1.0
- Early stopping: patience=10 on validation AUPRC
- Temporal stratified 5-fold cross-validation with user-level inductive splits
- Test users REMOVED from training graph (prevents label leakage through message passing)
- Window labeling: ≥50% overlap with attack = positive, 1-49% = excluded from training, 0% = negative
- 3 seeds per experiment for statistical significance

### Progressive Model Ladder
Each step must beat the previous to justify its existence:
1. MLP on z-scored deviations + OCEAN (~5K params) — proves signal exists
2. Temporal CNN on deviation sequences (~50K params) — proves temporal patterns help
3. Temporal CNN + GATv2 graph (~250K params) — proves relational context helps
4. Full Fusion model (~300K params) — proves static context helps

### Realistic Performance Expectations (CERT r4.2, honest temporal eval)
- Step 1 (MLP): AUPRC 0.15–0.25
- Step 2 (Temporal CNN): AUPRC 0.35–0.50 (biggest jump)
- Step 3 (+Graph): AUPRC 0.45–0.55
- Step 4 (Fusion): AUPRC 0.50–0.60
- Ceiling: 0.55–0.65. Limited by 70 positive examples, not architecture.

---

## 4. Implementation Plan (10 Phases)

| Phase | What | Days | Status |
|-------|------|------|--------|
| 0 | Codebase cleanup + Docker + reproducibility + literature review | 3-4 | **DONE** (16 commits) |
| 1 | Data exploration + universal pipeline for all 10 CERT versions | 7-9 | **NEXT** |
| 2 | Per-user baselines + feature store (Module 1) | 5-7 | Pending |
| 3 | XGBoost + MLP baselines as validation gate | 3-4 | Pending |
| 4 | Temporal CNN (Module 2) | 5-6 | Pending |
| 5 | Graph + Qdrant (Module 3) | 8-10 | Pending |
| 6 | Fusion model + evaluation harness (Module 4) | 5-6 | Pending |
| 7 | Frontend + Docker + CI/CD + demo infrastructure | 9-12 | Pending |
| Remote | GPU training, multiple seeds ($4-8) | 1-2 | Pending |
| 8 | Qdrant population + full system verification | 2-3 | Pending |
| 9 | Thesis presentation materials | 4-5 | Pending |

---

## 5. Current Codebase State (Post Phase 0)

**Project folder:** `/Users/michaelkuksov/Developer/innersight`

### 5.1 Git History (18 commits)
```
aa6f292 Phase 0.16: Literature review with target numbers
898e059 Phase 0.15: Remove .pt checkpoints from git tracking
2854c58 Phase 0.14: Add reproducibility infrastructure
9398098 Phase 0.13: Add Docker and docker-compose setup
2944263 Phase 0.12: Fix CI/CD workflows to match actual project structure
d27e2ec Phase 0.11: Eliminate sys.path hacking (package is pip-installed)
cad4f07 Phase 0.10: Add pyproject.toml for proper package installation
f7a897c Phase 0.9: Final verification fixes
b20d593 Phase 0.8: Archive learning scripts with README
3c68b97 Phase 0.7: Apply senior Python engineering standards
9057dd3 Phase 0.6: Remove legacy numpy .npz checkpoint compatibility
9ddf61a Phase 0.5: Fix config.py LDAP handling and checkpoint model_type bugs
47bf2ef Phase 0.4: Remove all hardcoded cert_r4.2 references
2ad6660 Phase 0.3: Fix all imports for new folder structure
0d0d457 Phase 0.2: Rename folders to clean professional structure
c1d5b5d Phase 0.1: Delete dead code and root-level junk
f4a205c Phase 5: Heterogeneous GraphSAGE GNN for insider threat detection
edaec8a Fresh start
```

### 5.2 Complete File Tree (production code only)

```
innersight/
├── __init__.py                          # Root package marker
├── .gitignore                           # Covers *.pt, *.npz, *.parquet, .venv, etc.
├── .env                                 # Empty (use .env.example as template)
├── .env.example                         # INNERSIGHT_DATA_DIR, INNERSIGHT_MODEL_DIR
├── docker-compose.yml                   # 3 services: backend, frontend, qdrant
├── .github/workflows/ci.yml            # Lint (ruff) + Test (pytest) + Docker build
├── scripts/download_checkpoints.sh      # Placeholder for GitHub Releases download
│
├── docs/
│   └── literature_targets.md            # 10 papers, AUPRC targets, evaluation methodology critique
│
├── backend/
│   ├── pyproject.toml                   # Package definition, deps, tool config
│   ├── requirements.txt                 # Pinned versions (torch==2.11.0, etc.)
│   ├── Dockerfile                       # Python 3.13-slim, pip install -e .
│   ├── .dockerignore
│   ├── .gitignore
│   ├── run.sh                           # Legacy dev script (create venv, run tests, start Flask)
│   │
│   ├── config.py                        # Central config: paths, feature cols, training defaults, constants
│   ├── api.py                           # Flask REST API (train SSE, alerts, employees, investigation)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py                  # load_raw_logs(), load_labels(), time_split(), load_data()
│   │                                    #   → Single-version only. Phase 1 replaces with universal pipeline.
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── features.py                  # 18 per-user-per-day features from raw logs
│   │                                    #   → logon(5), device(3), file(3), email(4), http(3)
│   │                                    #   → build_user_day_features(), build_features_for_split()
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                       # InsiderThreatMLP (simple feedforward, [18,64,32,1])
│   │   │                                #   → BCEWithLogitsLoss, get_device(), build_mlp()
│   │   ├── dataset.py                   # Standardizer class + build_dataloaders()
│   │   ├── factory.py                   # Model factory pattern for checkpoint loading
│   │   ├── graphsage.py                 # HeteroGraphSAGE + InsiderThreatGNN
│   │   │                                #   → Uses SAGEConv (NOT GATv2Conv — must change in Phase 5)
│   │   │                                #   → 128-dim hidden, 2 layers, dropout 0.3
│   │   │                                #   → MLP head: [128, 64] → 1
│   │   ├── graph_schema.py              # Node/edge type constants, feature dimensions
│   │   │                                #   → 4 node types, 5 edge types + 5 reverse
│   │   │                                #   → USER_FEATURE_DIM=16, PC=8, URL=8, FILE=6
│   │   │                                #   → Edge dims: LOGON=4, USB=3, EMAIL=5, HTTP=3, FILE=4
│   │   ├── graph_builder.py             # ~1,600 lines: build all node/edge tensors → HeteroData
│   │   │                                #   → Per-event edges (NOT windowed-aggregated — must change)
│   │   │                                #   → build_user_nodes(), build_pc_nodes(), build_url_nodes(),
│   │   │                                #     build_file_nodes(), build_*_edges(), build_hetero_graph()
│   │   ├── graph_loader.py              # Load pre-built .pt graph files
│   │   ├── gnn_trainer.py               # GNN training loop (same event contract as MLP trainer)
│   │   ├── node2vec_trainer.py          # Node2Vec embedding trainer
│   │   └── embeddings.py                # EmbeddingManager for Node2Vec/MetaPath2Vec
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                   # MLP training loop: train(), _evaluate(), _save_model()
│   │                                    #   → BCEWithLogitsLoss + pos_weight (OLD — plan uses Focal Loss)
│   │                                    #   → Early stopping on val F1 (should be AUPRC)
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── scoring.py                   # score_employees(), load_alerts(), update_alert_status()
│   │                                    #   → Loads MLP or GNN checkpoint by model_type tag
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── feedback.py                  # apply_learn(), apply_mute(), apply_block()
│   │                                    #   → Online fine-tuning on false positives
│   │
│   ├── utils/
│   │   ├── __init__.py                  # Exports: safe_json_read, safe_json_write, seed_everything
│   │   ├── io.py                        # safe_json_write() (atomic), safe_json_read() (graceful fallback)
│   │   └── reproducibility.py           # seed_everything(seed=42) — Python, NumPy, PyTorch, CUDA
│   │
│   ├── configs/
│   │   ├── train_graphsage.yaml
│   │   ├── train_metapath2vec.yaml
│   │   ├── train_mlp.yaml
│   │   └── train_node2vec.yaml
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── train.py                     # CLI training entry point
│   │   ├── train_embeddings.py          # Node2Vec/MetaPath2Vec training
│   │   ├── extract_embeddings.py        # Extract embeddings from trained GNN
│   │   ├── compare_models.py            # Compare MLP vs GNN performance
│   │   ├── validate_graphs.py           # Validate graph .pt files
│   │   └── visualize_embeddings.py      # UMAP visualization
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_config.py
│   │   ├── test_dataset.py
│   │   ├── test_gnn_trainer.py
│   │   ├── test_graphsage.py
│   │   ├── test_integration.py
│   │   ├── test_mlp.py
│   │   ├── test_node2vec.py
│   │   ├── test_reproducibility.py
│   │   ├── test_trainer.py
│   │   ├── test_utils.py
│   │   └── smoke_test_api.py
│   │
│   ├── checkpoints/                     # .pt files UNTRACKED (in .gitignore), .gitkeep files only
│   │   ├── .gitkeep
│   │   ├── alerts.json                  # Persisted alerts for the API
│   │   ├── flat/.gitkeep
│   │   ├── n2v/.gitkeep
│   │   ├── graphsage/.gitkeep
│   │   ├── graphsage/graphs/.gitkeep
│   │   ├── graphs/.gitkeep
│   │   └── m2v/.gitkeep
│   │
│   ├── innersight/data/                 # Runtime data store for alerts, corrections, block log
│   │   ├── alerts.json
│   │   ├── block_log.json
│   │   ├── corrections.json
│   │   └── graphs/.gitkeep
│   │
│   ├── outputs/
│   │   └── phase5_benchmark.txt         # Old benchmark (all zeros — no real data was used)
│   │
│   └── learning/                        # ARCHIVED tutorial scripts (not production)
│       ├── README.md
│       └── 01-11 .py files
│
└── frontend/
    ├── Dockerfile                       # Node 22 → npm build → nginx serve
    ├── .dockerignore
    ├── .env                             # VITE_API_URL
    ├── .env.example
    ├── nginx.conf                       # /api/ → backend:5001, SPA routing
    ├── package.json                     # React 19, Redux Toolkit, styled-components, Recharts
    ├── package-lock.json
    ├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
    ├── vite.config.ts
    ├── eslint.config.js
    ├── index.html
    │
    └── src/
        ├── main.tsx                     # React root
        ├── App.tsx                      # Routes: /alerts, /employees, /employee/:userId, /training
        │
        ├── components/
        │   ├── Navbar/                  # Top navigation bar
        │   ├── ErrorBoundary/           # React error boundary
        │   └── Spinner/                 # Loading spinner
        │
        ├── pages/
        │   ├── Alerts/                  # Alert list with severity, user, date, actions
        │   ├── Employees/               # Employee table with risk scores
        │   ├── Investigation/           # Per-user deep dive
        │   │   ├── ActionButtons/       # Learn/Mute/Block actions
        │   │   ├── ActivityTimeline/    # Event timeline
        │   │   ├── ConfirmDialog/       # Confirmation modal
        │   │   └── ScoreHistoryChart/   # Score over time (Recharts)
        │   ├── Training/                # Training control panel
        │   │   ├── ConfusionMatrix/
        │   │   ├── MetricsChart/        # Loss/F1 curves (Recharts)
        │   │   ├── TrainingConfig/      # Hyperparameter form
        │   │   └── TrainingControls/    # Start/stop buttons
        │   └── NotFound/
        │
        └── store/
            ├── store.ts                 # Redux store
            ├── hooks.ts                 # Typed useAppDispatch/useAppSelector
            └── slices/
                ├── alertsSlice.ts
                ├── employeesSlice.ts
                ├── investigationSlice.ts
                ├── trainingSlice.ts
                └── uiSlice.ts
```

### 5.3 Key Configuration Values (config.py)

```python
DATA_DIR  = os.environ.get('INNERSIGHT_DATA_DIR', '')    # Must be set for real data
MODEL_DIR = os.environ.get('INNERSIGHT_MODEL_DIR', 'innersight/data')

BUSINESS_HOURS_START = 7
BUSINESS_HOURS_END   = 19
INTERNAL_DOMAIN      = 'dtaa.com'
JOB_KEYWORDS         = ('job', 'career', 'linkedin', 'indeed')
CLOUD_KEYWORDS       = ('dropbox', 'wikileaks', 'pastebin')

TRAIN_END_DATE = '2010-09-30'
VAL_END_DATE   = '2010-11-30'

FEATURE_COLS = [                        # 18 features per user per day
    'logon_count', 'logoff_count', 'after_hours_logons', 'weekend_logons', 'unique_pcs_used',
    'usb_connect_count', 'usb_disconnect_count', 'after_hours_usb',
    'file_count', 'file_to_removable_count', 'unique_filenames',
    'email_sent_count', 'email_to_external_count', 'large_attachment_count', 'total_email_size',
    'http_request_count', 'job_search_visits', 'cloud_upload_visits',
]

DEFAULT_TRAINING_CONFIG = {
    'epochs': 50, 'batch_size': 64, 'lr': 0.001,
    'layer_sizes': [18, 64, 32, 1],
    'pos_weight': 50.0, 'patience': 5, 'seed': 42,
}
```

### 5.4 API Endpoints (api.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/config | Training configuration |
| POST | /api/train | Start training (SSE stream) |
| GET | /api/events | SSE event stream for training progress |
| GET | /api/status | System status |
| GET | /api/alerts | List alerts (optional status_filter) |
| GET | /api/employees | All employees with risk scores |
| GET | /api/employee/\<user_id\>/activity | User's activity timeline |
| GET | /api/employee/\<user_id\>/score-history | Score history chart data |
| POST | /api/alert/\<alert_id\>/learn | Mark as false positive + fine-tune |
| POST | /api/alert/\<alert_id\>/mute | Suppress alert |
| POST | /api/alert/\<alert_id\>/block | Block user |

### 5.5 Graph Schema (graph_schema.py)

**Node types:** user (16 features), pc (8), url (8), file (6)

**Edge types (forward + reverse = 10 total):**
- (user, logon, pc) — 4 edge features
- (user, usb_connect, pc) — 3 edge features
- (user, email_to, user) — 5 edge features
- (user, http_request, url) — 3 edge features
- (user, file_copy, file) — 4 edge features

### 5.6 Docker Setup

**docker-compose.yml** defines 3 services:
- `backend`: Python 3.13-slim, Flask on port 5001, depends on qdrant
- `frontend`: Node 22 build → nginx on port 3000, proxies /api/ to backend
- `qdrant`: Official image on port 6333, health-checked

**CI/CD** (.github/workflows/ci.yml): lint (ruff), test (pytest), docker build

---

## 6. What Exists vs. What the New Architecture Needs

| Component | Current state | What Phase 1+ needs |
|-----------|--------------|-------------------|
| Data pipeline | Single-version loader (r4.2 only) | Universal pipeline with 6 adapters for all 10 versions |
| Feature extraction | 18 raw daily features | Same 18 features + EMA baselines + z-scored deviations |
| Feature store | None (recomputes every time) | Parquet persistence, never recompute |
| Model: MLP | BCEWithLogitsLoss, pos_weight=50 | Focal Loss (α=0.75, γ=2.0), threshold from PR curve |
| Model: Graph | SAGEConv (ignores edge features) | GATv2Conv with edge_attr (4 attention heads) |
| Graph edges | Per-event edges | Windowed-aggregated: count, frac_after_hours, is_new, max_burst |
| Temporal model | None | 4-layer dilated causal CNN on 28-day windows |
| Evaluation | F1 + confusion matrix | AUPRC, P@k, per-scenario breakdown, detection latency |
| Training | Early stopping on val F1 | Early stopping on val AUPRC, cosine annealing, gradient clipping |
| Qdrant | Listed in docker-compose | Need population script + /users/:id/similar endpoint |
| Demo dataset | None | 50-100 synthetic records bundled in repo |

---

## 7. What Was Approved by the Architecture Council

The architecture was reviewed across 4 rounds by a 5-engineer council (Breaker, Architect, Pragmatist, Veteran, Operator). Final verdict: **SHIP WITH CAVEATS.**

**What the council caught and we fixed:**
- Focal loss α was inverted (fixed to 0.75)
- OCEAN scores were in CNN input (removed — convolution over a constant produces nothing; moved to Module 4)
- BatchNorm was planned (changed to LayerNorm — BatchNorm fails with 0.4% positive rate)
- Z-score std floor was missing (added: 10% of global median)
- Cold-start for new hires was unaddressed (added: role-cohort hierarchical fallback)

**Council caveats acknowledged:**
- Realistic ceiling is 0.55-0.65 AUPRC — limited by 70 positive examples, not architecture
- Graph module may add less than expected on synthetic data — progressive ladder will reveal this
- Per-event edges vs. windowed-aggregated edges is a significant change from current code

---

## 8. Literature Targets (from docs/literature_targets.md)

Most papers report AUC-ROC (which saturates near 1.0 at CERT's imbalance). Our metric is AUPRC.

**Key reference points:**
- Yuan et al. 2018: AUC-ROC 0.9449 on r4.2 (LSTM→CNN hybrid, split type unverified)
- LAN (Cai et al. 2024): AUC-ROC 0.9369-0.9607 on r4.2 (temporal split, best verified baseline)
- Chattopadhyay 2018: Claims 100% P/R/F on r4.2 — heavily inflated (random split, undersampling, best-config cherry-picking)
- Tuor 2017: CR-1000 ≈ 35.6 on r6.2 (temporal, but different dataset version)

**Our honest targets (AUPRC, temporal split, r4.2):**
- MLP: 0.15-0.25
- Temporal CNN: 0.35-0.50
- +Graph: 0.45-0.55
- Full fusion: 0.50-0.60
- Ceiling: 0.55-0.65

---

## 9. Key Constraints

- **Solo university student** — graded on code quality, working system, ML results, presentation
- **Teacher runs via `docker-compose up`** on their own machine — must work perfectly
- **No sustained GPU load on Mac laptop** — smoke tests only, real training on remote VM ($4-8 budget)
- **Dataset on external hard drive** — NEVER mount in sandbox (kills it), use CERT_DATASET_REFERENCE.md
- **Project folder**: `/Users/michaelkuksov/Developer/innersight`
- **Python 3.13** locally, pyproject.toml says `>=3.10`
- **Frontend**: React 19 + TypeScript + Vite + Redux Toolkit + styled-components + Recharts

---

## 10. Senior Python Standards (Enforced)

All code must follow these standards (from the senior-python skill):

- **Type hints** on all public function signatures
- **Module docstrings** describing purpose and public API
- **Import order**: stdlib → third-party → local (blank lines between)
- **No sys.path hacking** — package is installed via `pip install -e .`
- **No global mutable state** — pass state explicitly
- **No magic numbers** — every constant lives in config
- **No bare `except:`** — catch specific exceptions
- **Fail fast and loud** — raise with descriptive messages, don't return silent defaults
- **Use `logging.getLogger(__name__)`** — no `print()` in production code
- **Dataclasses for config** — `frozen=True`, with `__post_init__` validation
- **Single responsibility** — one class does one thing
- **Composition over inheritance**
- **Small functions** — max ~40 lines, extract helpers
- **When to split files**: when one file does more than one job (the "and" test)
- **When NOT to split**: 3 related functions in 80 lines don't need their own file

---

## 11. What to Build Next (Phase 1: Universal Data Pipeline)

Phase 1 has two parts: (a) data exploration and (b) building the adapter layer.

### Phase 1a: Data Exploration (2-3 days)
- Profile all 10 versions: row counts, column schemas, timestamp ranges, user counts, insider counts
- Document every difference between versions (see CERT_DATASET_REFERENCE.md — already done)
- Validate the answers/insiders.csv master file against per-insider detail files

### Phase 1b: Universal Pipeline (5-6 days)
- Canonical schema (Python dataclasses) defining every column name, dtype, required/optional
- 6 version adapters (~15 lines each): r1, r2, r3x, r4x, r5x, r6x
- Header fingerprint validation to auto-detect version
- Answers parser handling both flat CSVs and subdirectory formats
- LDAP + psychometric + decoy loaders (gracefully absent when missing)
- Chunked streaming: `pd.read_csv(chunksize=50000)` for r6.2 http.csv (85GB)
- Provenance manifest (JSON) per run: version, file hashes, adapter, timestamp
- Tests: unit test per adapter, integration test (r4.2 through new pipeline = identical output)

### Validation Gate
All 6 adapters pass unit tests. r4.2 produces identical output through old vs. new pipeline. r5.2 loads with psychometric + decoy. r6.2 http.csv streams without OOM.

---

## 12. Documents Available

These documents provide deep architectural detail and were all council-reviewed:

1. **CONTEXT_FOR_NEW_CONVERSATION.md** — This file
2. **CERT_DATASET_REFERENCE.md** — Complete dataset schemas, quirks, adapter fingerprints
3. **InnerSight_Temporal_Relational_Architecture.html** — 12-chapter neural network architecture guide
4. **InnerSight_Implementation_Plan_v4.html** — Phased implementation plan with validation gates
5. **InnerSight_System_Components.html** — 77-component specification (every file, class, function)
6. **docs/literature_targets.md** — Literature review with 10 papers and AUPRC targets
