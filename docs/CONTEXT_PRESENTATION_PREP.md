# InnerSight — Context for a New Conversation (Presentation Prep)

> **How to use this file:** Upload to a new Claude conversation to continue
> exactly where we left off. This captures what the project *actually does today*,
> the technologies behind it, the neural-network design, and the honest status —
> framed for explaining/defending the work to a university examiner.

---

## 1. Situation

Solo university thesis project. The student is **presenting tomorrow** what has
been built **up to Phase 7**. The goal of recent work was *understanding and
explaining* the system in plain language (not writing new code). This file is the
distilled result of that prep.

A detailed phase-by-phase walkthrough already exists at
`docs/PHASE_WALKTHROUGH_FOR_PRESENTATION.html` (and `.md`). This file is the
shorter "what it actually does + how + honest status" companion.

---

## 2. What InnerSight Actually Does (plain language)

InnerSight detects **insider threats** — employees stealing data, sabotaging
systems, or leaking to competitors — by watching day-to-day computer activity and
flagging when someone behaves **unlike their own normal self**.

It runs as a web app (`make demo` → `http://localhost:3000`) on a bundled
synthetic dataset (5 users, 1 insider) so it works instantly with no setup.

**The five stages it performs, end to end:**

1. **Read activity logs** — per employee: logons, USB usage, file copies, emails,
   web browsing (CSV files).
2. **Turn logs into 18 daily numbers per person** — a daily behavioural
   fingerprint (logon counts, after-hours logons, USB copies, external emails,
   large attachments, job-site/cloud-upload visits, etc.).
3. **Learn each person's personal normal** — track a running average of what's
   typical *for that individual*; express today as a z-score (how many standard
   deviations from their own baseline). New users fall back to a role/department
   cohort average until they build history.
4. **Score risk and raise alerts** — turn deviations into a 0–1 risk score; above
   a threshold, raise an explainable alert listing the top contributing features.
5. **Find look-alikes** — given one suspect, search for other employees behaving
   similarly (vector similarity search).

The web app surfaces this as: **Alerts** (ranked, with triage), **Employees**
(directory with risk levels), **Investigation** (deviation heatmap, per-feature
breakdown, activity timeline, graph neighborhood), **Suspect Discovery**, **Model
Comparison**, and a live **Training** page.

---

## 3. Technologies Behind Each Stage

- **Stage 1 (read logs):** Python + **pandas** (tabular data); per-version adapter
  layer normalizes CERT's differing CSV schemas into one standard layout; results
  cached as **Parquet** via **pyarrow**.
- **Stage 2 (18 features):** **pandas** group-by/aggregate + datetime logic
  (after-hours, weekend). Pure data engineering, no ML.
- **Stage 3 (personal normal):** **NumPy** statistics — **EMA** (exponential
  moving average) baseline + standard deviation → **z-scores**; cohort fallback
  (role → department → global) for cold-start. No training, no GPU.
- **Stage 4 (score + alerts):** Live demo path = simple **NumPy** math on z-scores
  → JSON alerts. The powerful path (built, not yet trained) = **PyTorch** models +
  **XGBoost** baseline, served via **Flask** REST API (+ flask-cors).
- **Stage 5 (look-alikes):** **Qdrant** vector database, k-nearest-neighbor search
  over behavioural embeddings (its own Docker container).
- **Frontend:** **React 19** + **TypeScript**, Redux, styled-components,
  **Recharts** (charts/heatmaps), **d3-force** (interactive graph); served by
  **nginx**.
- **Packaging:** **Docker** + **docker-compose** — 3 services: backend Flask
  (`:5001`), frontend nginx (`:3000`), Qdrant (`:6333`).

---

## 4. The Neural Networks (3 of them) and How They Relate

There are **three neural networks**. Critically, they are **not** all fused into
one vector — the MLP is a separate baseline.

### MLP (Multi-Layer Perceptron) — standalone baseline
- Simplest network; takes the **18 daily numbers**, outputs **one risk score**
  (shape ~`18 → 64 → 32 → 1`).
- Sees **only one day at a time** — no sense of time. Its job is to be the
  **baseline** the fancier models must beat. **Not fused into the final model.**

### Temporal CNN (`TemporalPatternEncoder`) — reads behaviour over time
- Convolution slides filters across a **28-day window** to detect *patterns of
  change* (gradual escalation, bursts).
- **Causal** (only looks at past/present days, never the future → no leakage);
  uses **dilations** so few layers cover the whole window.
- Outputs a **128-number embedding** per user + an **attention map** (which days
  it focused on → interpretability).

### GNN (`GraphContextEncoder`, GATv2) — uses relationships
- Builds a graph: **4 node types** (user, pc, url, file), **5 edge types** +
  reverses = **10** (logon, usb_connect, email_to, http_request, file_copy).
  Edges carry features: counts, after-hours fraction, novelty flags.
- **GATv2 attention** learns *how much* each connection matters (boring logon =
  low weight; new after-hours USB copy = high weight).
- Catches relational signals a per-user model can't: emailing a competitor,
  clusters touching the same exfiltration site/file, contact outside normal
  circles. Outputs a **128-number embedding** per user.

### How CNN and GNN work together (chained, not parallel)
1. **CNN runs first** → 128-dim embedding per user (behaviour over time, in
   isolation).
2. **Those embeddings are injected as the graph's user-node features** (the graph
   builder seeds user nodes with placeholder zeros, then the CNN embeddings
   replace them).
3. **GNN message-passing refines** each user's temporal embedding with relational
   context → a new 128-dim embedding ("how they behaved" + "the company they
   keep"). Example: a borderline CNN signal for Alice gets pushed to "suspicious"
   because the GNN sees she shares a copied file with users emailing an external
   domain.

### Fusion model (`InsiderThreatDetector`) — combines everything
- Concatenates **CNN(128) + GNN(128) + static(16)** = **272-dim** vector.
  - static = OCEAN personality (5) + role embedding (8) + department embedding (3);
    static bypasses the convolutions (a constant-over-time value gets no gradient
    through a temporal conv) and joins only at fusion.
- A **final MLP-style classification head** (`272 → 128 → 64 → 1`) reads the vector
  → risk score. **This head is a different MLP from the standalone baseline.**

**Clean one-liner:** "Two encoders (CNN + GNN) fuse into a 272-dim vector that an
MLP head classifies; a separate MLP is the baseline I compare against."

### The progressive ladder (the results story)
| Step | Model | Target AUPRC |
|---|---|---|
| 1 | MLP baseline | 0.15–0.25 |
| 2 | + Temporal CNN | 0.35–0.50 (biggest jump) |
| 3 | + GATv2 GNN | 0.45–0.55 |
| 4 | Full fusion | 0.50–0.60 |

---

## 5. HONEST STATUS — read this before presenting

- **Working live today (on synthetic demo data):** stages 1–3 and the alerting.
  The app genuinely ingests logs, builds per-user baselines, computes z-score
  deviations, scores risk, raises explainable alerts, and serves the full
  investigation UI. The **live scoring is driven by the statistical baseline
  (Module 1 z-scores)** — `_generate_demo_alerts` in `api.py`.
- **Built and tested but NOT yet trained:** the MLP, CNN, GNN, and fusion model.
  `api.py:_get_model()` logs "No trained model found" if no `.pt` checkpoint
  exists. So **Model Comparison and the attention timeline show illustrative /
  structural output, not real trained numbers, until training happens.**
- **202 tests passing, 79 commits, 50 source files** — the engineering is solid;
  what's missing is the trained weights.

**Framing to use:** "The full system runs end-to-end today on synthetic data; the
deep-learning ladder is fully implemented and tested, and the trained weights come
from the next step (a short GPU run). Right now live scoring uses the statistical
baseline."

---

## 6. The Dataset (CERT)

- **CMU CERT Insider Threat Dataset** — synthetic (computer-generated) data from
  Carnegie Mellon's CERT Insider Threat Center (with ExactData/DARPA). Synthetic
  because real insider data is private/legally radioactive; synthetic gives
  **perfect ground-truth labels** (`insiders.csv`: who, when, which scenario).
- Log types: logon, device (USB), file, email, http, LDAP (org chart),
  psychometric (OCEAN). 3 scripted attack **scenarios** in r4.2.
- Ships in **10 versions** (r1–r6.2) with differing CSV schemas — hence the
  6-adapter pipeline. Primary training version: **r4.2** (1000 users, 70 insiders,
  ~16GB).
- **If asked "is this real data?"** → No, synthetic. Strength: perfect labels,
  shareable. Limitation: scripted insiders may behave more cleanly than real ones.

---

## 7. What's Next (if the examiner asks)

- **Remote Block:** rent a GPU (vast.ai, ~$4–8, ~6–8h), train all ladder steps on
  real CERT r4.2 with 3 seeds via `run_all_training.sh`.
- **Phase 8:** load real trained embeddings into Qdrant, verify the full system
  with trained models.
- **Phase 9:** thesis materials — UMAP visualization, error analysis, cost model,
  figures.

---

## 8. Key Numbers to Memorize

- **18** daily features · **28**-day windows, **stride 7** · ≥**50%** overlap =
  positive label.
- Imbalance ≈ **0.4%** positives → why **AUPRC** (not accuracy/AUROC) + **focal
  loss** (α=0.75, γ=2.0).
- Graph: **4** node types, **10** edge types (5 + reverses).
- Embeddings: temporal **128**, graph **128**, static **16**, fused **272**.
- Architecture: per-user baseline (Module 1, no training) → Temporal CNN (Module
  2) → GNN/GATv2 (Module 3) → Fusion + MLP head (Module 4).

---

## 9. Repo Facts

- Package root: `backend/innersight/`; imports are `from innersight.X import Y`;
  install `cd backend && pip install -e .`.
- Run: `make demo` (Docker). Other targets: `make test`, `make lint`,
  `make train-quick`, `make stop`.
- GitHub: https://github.com/Enterino8465/innersight.git
- Docs: `PHASE_WALKTHROUGH_FOR_PRESENTATION.html/.md`, `DATASET_CONTEXT.md`,
  `literature_targets.md`, `CONTEXT_FOR_NEW_CONVERSATION.md` (full project state).
