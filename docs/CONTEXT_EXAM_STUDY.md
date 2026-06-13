# InnerSight — Exam Study Context

> **How to use this file:** Upload to a new Claude conversation and say "quiz me
> on this" or "explain X deeper." It's a self-contained study companion for the
> InnerSight thesis project — concepts, the *why* behind every design choice, a
> glossary, and practice questions. Goal: be able to **explain and defend** any
> part of the system, not just recite it.

---

## 0. The 30-Second Summary

InnerSight is an **insider-threat detection** system (a UEBA — User & Entity
Behaviour Analytics platform). It learns each employee's **personal behavioural
baseline** from their computer activity, then flags **deviations** from that
normal using a ladder of models: a statistical baseline → an MLP → a temporal CNN
→ a graph neural network → a fusion model. Trained and evaluated on the synthetic
**CMU CERT Insider Threat Dataset**. **Status: MLP, XGBoost, and the Temporal CNN
are now trained on r4.2 with strong results (§6.5); the GNN and fusion are not yet
run.** Primary metric: **AUPRC** (because positives
are ~0.4% of the data).

---

## 1. The Core Idea (understand this cold)

**Problem:** Catch employees who steal data, sabotage, or leak — from their logs
alone. The catch: "anomalous" is personal. 10 logons is normal for a sysadmin,
alarming for a receptionist. And insiders are extremely rare (~0.4% of user-days),
so naive models that predict "everyone is fine" score 99.6% accuracy while
catching nobody.

**Solution in one sentence:** Normalise every person against *their own* normal
(turn raw activity into per-user z-scores), then learn *patterns of deviation*
that indicate an attack — over time (CNN) and across relationships (GNN).

**Why this is clever:** By the time any neural net sees the data, "abnormal for
this person" is already encoded. The models learn what *anomaly patterns* look
like, not who happens to be a sysadmin. This is also what makes it generalise to
new users and even new dataset versions.

---

## 2. The Pipeline — 5 Stages

1. **Ingest logs** — logon, device/USB, file, email, http per user (CSV →
   pandas). An adapter layer normalises 10 differing CERT versions into one schema.
2. **Feature engineering** — aggregate each user's events into **18 numbers per
   day** (a daily behavioural fingerprint).
3. **Per-user baseline (Module 1)** — EMA + std → **z-scored deviation** per
   feature per day. Cohort fallback for new users.
4. **Modelling + scoring** — the model ladder turns deviations into a risk score.
5. **Similarity search** — Qdrant k-NN over behavioural embeddings to find
   look-alike suspects.

### The 18 daily features (know the categories, not necessarily all 18)
- **Logon (5):** logon_count, logoff_count, after_hours_logons, weekend_logons,
  unique_pcs_used
- **Device/USB (3):** usb_connect_count, usb_disconnect_count, after_hours_usb
- **File (3):** file_count, file_to_removable_count, unique_filenames
- **Email (4):** email_sent_count, email_to_external_count,
  large_attachment_count, total_email_size
- **HTTP (3):** http_request_count, job_search_visits, cloud_upload_visits

These map directly to insider behaviours: after-hours access, USB exfiltration,
emailing data out, job-hunting before quitting, cloud uploads.

---

## 3. The Models (the heart of the exam)

### Module 1 — Per-User Behavioural Baseline (NO training)
- **EMA (exponential moving average), α≈0.05:** a running average that weights
  recent days more. Tracks "what's typical for this user."
- **z-score:** `(today − baseline_mean) / baseline_std`. How many standard
  deviations from personal normal. A **std floor** prevents quiet features
  (near-zero variance) from exploding the z-score.
- **Cohort cold-start:** new users have no history → seed from a cohort prior
  (most-specific first: **role → department → global**, min 5 users). Without
  this, every new hire looks anomalous.
- **Why no training?** It's pure statistics — and that's a feature: it works live
  with no GPU, and it does the normalisation so the neural nets can focus on
  pattern recognition.

### The windowing (how data is fed to sequence models)
- **28-day windows, stride 7.** Each sample is a `(18 features × 28 days)` matrix.
- **Overlap-ratio labelling:** a window is **positive** if ≥50% overlaps a known
  attack, **negative** if 0% overlap, and **excluded** (−1) if 1–49%. Why exclude
  the middle? Ambiguous half-attack windows would add label noise; dropping them
  keeps training clean.

### Model 1 — MLP (baseline)
- Multi-Layer Perceptron: stacked layers of neurons.
- **In the ladder evaluation it takes the *flattened* 28-day window (18×28 = 504)
  + 5 OCEAN scores = 509 inputs.** So it sees the *same data* as the CNN but as a
  flat vector — it loses the temporal ordering (which day came before which).
- This is the clean comparison: **CNN 0.997 vs MLP 0.448 on the same input
  isolates the value of modelling temporal structure.** The CNN's convolutions
  exploit *order*; the MLP can't.
- **Role: the baseline to beat. Not part of fusion.** (The app's live/production
  MLP in `trainer.py` is a simpler per-day variant; the ladder baseline is the
  windowed one above.)

### Model 2 — Temporal CNN (`TemporalPatternEncoder`)
- **Convolution over time:** filters slide across the 28-day window to detect
  *patterns of change* (escalation, bursts) — same math as image CNNs, applied to
  a time axis instead of pixels.
- **Causal:** left-padding only, so day *t* only sees days ≤ *t*. **Prevents data
  leakage** (can't use the future to predict the present). Common exam trap.
- **Dilated (1,2,4,8):** dilation skips inputs so the receptive field grows
  exponentially (≈3→7→15→23 days) with few layers — captures both short bursts
  and long trends cheaply.
- **LayerNorm, not BatchNorm:** with ~0.4% positives, most batches are
  all-negative, so batch statistics would be dominated by normal users. LayerNorm
  normalises **per sample**, so it's stable under imbalance.
- **Residual connections:** help gradients flow through depth (1×1 conv to match
  channels).
- **Attention pooling:** collapses 28 days into one **128-dim embedding** and
  exposes *which days* mattered (interpretability).
- Target AUPRC 0.35–0.50 — **biggest single jump** (time-awareness is the key
  signal).

### Model 3 — Graph Neural Network (`GraphContextEncoder`, GATv2)
- **Graph:** 4 node types (user, pc, url, file), 5 edge types + reverses = 10
  (logon, usb_connect, email_to, http_request, file_copy). **Heterogeneous**
  graph = multiple node/edge types.
- **Edge features:** counts, after-hours fraction, novelty (first contact). The
  model weights *how* a user connects, not just whether.
- **Message passing:** each node updates itself by aggregating neighbours' info;
  repeat → information spreads several hops.
- **GATv2 (Graph Attention v2):** learns *attention weights* over neighbours — how
  much each connection matters. 4 heads (multiple attention patterns), 2 layers,
  full-batch.
- **Catches relational signals** a per-user model can't: emailing a competitor,
  clusters sharing an exfil file, contact outside normal circles.
- Output: **128-dim embedding** per user. Target AUPRC 0.45–0.55.

### How CNN → GNN chain (key conceptual question)
1. CNN encodes each user's 28-day behaviour → 128-dim embedding.
2. **Those embeddings are injected as the graph's user-node features** (graph
   builder seeds user nodes with zeros, then CNN embeddings replace them).
3. GNN refines: "how they behaved over time" + "the company they keep" → new
   128-dim embedding. The GNN doesn't start from scratch — it builds on the
   temporal representation.

### Model 4 — Fusion (`InsiderThreatDetector`)
- Concatenate **temporal(128) + graph(128) + static(16) = 272-dim**.
  - **static = OCEAN personality (5) + role emb (8) + dept emb (3).**
  - Static **bypasses the convolutions** — a constant-over-time value gets zero
    gradient through a temporal conv, so it joins only at fusion.
- **Classification head:** MLP `272 → 128 → 64 → 1` → risk score. (This head is a
  *different* MLP from the baseline.)
- Target AUPRC 0.50–0.60 — the full model.

### The progressive ladder (ACTUAL trained results on r4.2, 5-fold CV × 3 seeds)
| Step | Model | AUPRC | AUROC | P@10 | F1 | Status |
|---|---|---|---|---|---|---|
| 1 | MLP baseline (flattened window + OCEAN) | **0.448** | 0.834 | 0.81 | 0.53 | trained |
| 1b | XGBoost (129 handcrafted temporal features) | **0.985** | 0.9998 | 1.0 | 0.97 | trained |
| 2 | Temporal CNN (raw 18×28 window) | **0.997** | ~1.0 | 1.0 | 0.99 | trained |
| 3 | + GATv2 GNN | — | — | — | — | not yet run |
| 4 | Full fusion | — | — | — | — | not yet run |

**Read §6.5 — these numbers are very high and you MUST be able to explain why.**
The original literature *targets* were 0.15–0.25 (MLP) → 0.50–0.60 (fusion); the
actual results blow past them, which is itself something to address head-on.

---

## 4. Training & Evaluation (examiners love methodology)

- **Class imbalance ~0.4% positives** drives almost every design choice.
- **Focal loss (α=0.75, γ=2.0):** down-weights easy examples `(1−p)^γ`,
  up-weights rare positives via α. Plain cross-entropy would be swamped by easy
  negatives. (Lin et al., 2017 — RetinaNet.)
- **AUPRC (area under precision-recall curve) = primary metric.** Under heavy
  imbalance, accuracy and even AUROC look great while catching nothing; PR curves
  focus on the positive class. AUROC kept only for literature comparison.
- **Threshold calibration:** pick the probability cut-off that maximises F1 on
  validation, not a blind 0.5.
- **Operational metrics:** Precision@k (P@10/20/50 — "of the top-k alerts an
  analyst reviews, how many are real?"), per-scenario breakdown, **detection
  latency** (days after attack start until caught).
- **Leakage-safe user-level cross-validation:** split by *user* so the same person
  never appears in train and test. Temporal stratified k-fold, multiple seeds.
- **Cross-version generalization:** train on one CERT version, test on another.
  Works because the model is **inductive** — the CNN scores a window independent
  of user identity, and the GNN keys on the fixed schema. Only per-version
  role/dept id tables don't transfer (handled via an "unknown" embedding index).
  This is the strongest evidence the model learned *insider behaviour*, not
  memorised users.

---

## 5. The Dataset (CERT)

- **CMU CERT Insider Threat Dataset** — **synthetic**, from Carnegie Mellon's CERT
  Insider Threat Center (+ ExactData/DARPA). Synthetic because real insider data
  is private/unshareable; synthetic gives **perfect ground-truth labels**.
- Files: logon, device, file, email, http, LDAP (org chart), psychometric
  (OCEAN). `insiders.csv` = labels (user, start, end, scenario).
- **10 versions (r1–r6.2)** with differing schemas → 6-adapter pipeline. Primary:
  **r4.2** (1000 users, 70 insiders, 3 attack scenarios, ~16GB).
- **3 attack scenarios** (r4.2): e.g. ramping USB exfil before quitting; sysadmin
  sabotage; data theft to a competitor.
- **Limitation to acknowledge:** synthetic insiders may behave more cleanly than
  real ones; results may be optimistic vs. a real deployment.

---

## 6. Honest Status (don't get caught out)

- **NOW TRAINED on real CERT r4.2** (43,651 windows, 125 positive windows, 5-fold
  CV × 3 seeds): MLP baseline, XGBoost, and Temporal CNN — see §6.5 for numbers.
  A CNN checkpoint exists (`results_r4.2/checkpoints/temporal_cnn_r4.2.pt`).
- **NOT yet run:** the GNN and the full fusion model (steps 3–4 of the ladder).
  So the "graph adds relational lift" claim is still a hypothesis, not a result.
- **Working live (demo):** the web app runs on synthetic demo data; live scoring
  in the demo is still the statistical baseline (Module 1), independent of these
  trained checkpoints.
- **Next:** run GNN + fusion → load embeddings into Qdrant (Phase 8) → thesis
  materials (Phase 9: UMAP, error analysis, cost model, figures).

---

## 6.5 Trained Results & How to Defend Them (CRITICAL for the exam)

**The numbers (mean over 5 folds × 3 seeds, r4.2):**

| Model | AUPRC | AUROC | F1 | P@10 | P@20 | Input |
|---|---|---|---|---|---|---|
| MLP baseline | 0.448 ±0.05 | 0.834 | 0.53 | 0.81 | 0.54 | flattened window (504) + OCEAN (5) |
| XGBoost | 0.985 ±0.003 | 0.9998 | 0.97 | 1.0 | 0.99 | 129 handcrafted temporal features |
| Temporal CNN | 0.997 ±0.0003 | ~1.0 | 0.99 | 1.0 | 0.99 | raw 18×28 window |

**Three things an examiner WILL ask — have these answers ready:**

**(1) "0.997 AUPRC is near-perfect — is this real, or is it leakage?"**
This is the #1 risk question. Honest, defensible answer:
- *Why it can legitimately be this high:* CERT r4.2 insiders are **scripted** and
  their attack behaviour (scenario 2: job-site browsing + a spike in USB/file
  copies) is statistically very distinct from their own baseline. Once you
  z-score against each user's personal normal and look at a 28-day window, the
  positives become highly separable. High AUC/AUPRC on r4.2 is consistent with
  the literature for windowed, per-user-normalised detection.
- *What protects against leakage (be ready to state):* **user-level CV splits**
  (no user in both train and test), and **windows with 1–49% attack overlap are
  excluded** so labels aren't fuzzy.
- *The honest caveat to volunteer:* numbers this high warrant a leakage audit.
  The thing to double-check / be ready to discuss is **whether the per-user EMA
  baseline and the feature Standardizer are fit on training folds only** (fitting
  them on all data, including test windows, would leak). If you can confirm they
  are fit train-only, say so confidently; if unsure, say "that's the first thing
  I'd verify." Showing awareness of this beats pretending the number is obviously
  fine.

**(2) "Your XGBoost baseline already gets 0.985 — why do you need a CNN/GNN at
all?"** A fair and likely question, since XGBoost ≈ CNN here.
- The honest take: on r4.2 with good handcrafted temporal features, a classical
  model is *already excellent* — deep learning buys little **on this dataset**.
- The thesis case for the deep models: (a) the CNN reaches the same result from
  **raw data with no manual feature engineering** (the 129 handcrafted stats *are*
  the human-designed version of what the CNN learns); (b) the CNN/GNN produce
  **embeddings** that power similarity search (Qdrant suspect discovery) and
  **transfer across dataset versions** — things XGBoost on fixed features does not
  do; (c) the graph stage targets relational attacks (collusion, email-to-
  competitor) that per-window features miss. Frame deep learning as buying
  **generality, transfer, and embeddings**, not raw AUPRC on r4.2.

**(3) "Why does the MLP get only 0.448 when XGBoost gets 0.985 on similar data?"**
- Because the MLP eats the **raw flattened window** (504 numbers) with no temporal
  structure and no feature engineering, while XGBoost eats the **129 engineered
  stats** (per-feature deviation magnitude, slope, burst energy, days-above-2σ/3σ).
  The gap is the value of **feature engineering / temporal structure** — exactly
  the gap the CNN closes by learning those features from raw data. This is a clean
  ML story: *raw-flat << engineered-features ≈ learned-conv-features.*

**Other findings worth knowing:**
- **XGBoost top features:** `http_request_count` deviations dominate (~0.48 +
  0.15 importance), then `unique_filenames` / `file_count` deviations, then
  after-hours/weekend logon stats. This matches **scenario 2** (job browsing +
  data theft) — good sanity check that the model keys on sensible signals.
- **Only scenario 2 appears in `per_scenario`.** The windowed ≥50%-overlap
  labelling mainly yields positive windows for scenario 2's *sustained* attacks;
  shorter/sparser scenarios produce few/no positive windows. **Limitation to
  state:** the system is effectively evaluated on (and strongest at) scenario-2
  style data-exfiltration; only ~26–29 of 70 insiders are caught at the
  window level.
- **Detection latency is negative (median ≈ −16 days for CNN/XGB):** the first
  flagged window *starts* before the labelled attack start — i.e. early warning
  (the 28-day window picks up the run-up). Good story, but the very tight
  consistency (std ≈ 0.4 days) is another thing to sanity-check for artefacts.
- **n_positive = 125 windows** out of 43,651 → ~0.29% positive: confirms the
  extreme imbalance that justifies AUPRC + focal loss.

---

## 7. Glossary (quick-reference for the exam)

- **UEBA:** User & Entity Behaviour Analytics — detect threats from behaviour, not
  signatures.
- **Insider threat:** malicious/negligent action by someone with legitimate
  access.
- **EMA:** exponential moving average — recency-weighted running average.
- **z-score:** deviations from mean in units of standard deviation.
- **Embedding:** a fixed-length vector summarising something (here, a user's
  behaviour) so models/search can compare them.
- **Class imbalance:** vastly more negatives than positives; breaks accuracy/AUROC.
- **AUPRC / PR curve:** precision vs. recall; the honest metric under imbalance.
- **Focal loss:** loss that focuses learning on hard/rare examples.
- **Convolution:** sliding filter that detects local patterns (here, over time).
- **Causal convolution:** only uses past/present inputs (no future leakage).
- **Dilation:** spacing in a conv filter to grow receptive field cheaply.
- **LayerNorm vs BatchNorm:** normalise per-sample vs per-batch; LayerNorm is
  imbalance-safe.
- **Attention:** learned weighting of which inputs (days / neighbours) matter most.
- **GNN / message passing:** nodes update by aggregating neighbour info.
- **GATv2:** graph attention — learns how much each edge/neighbour matters.
- **Heterogeneous graph:** multiple node and edge types.
- **Inductive model:** generalises to unseen nodes/graphs (vs transductive, which
  memorises specific ones).
- **Qdrant / vector DB / k-NN:** store embeddings, find nearest neighbours fast.
- **AUPRC vs AUROC:** PR is positive-class focused; ROC can look good despite
  missing positives under imbalance.

---

## 8. Likely Exam Questions (practice these out loud)

**Conceptual / "why":**
1. Why per-user baselines instead of one global model?
2. Why is AUPRC the primary metric and not accuracy or AUROC?
3. What is focal loss and why is it needed here?
4. Why causal convolutions? What would break without them?
5. Why LayerNorm instead of BatchNorm in this setting?
6. Why dilated convolutions — what do they buy you?
7. What does the GNN catch that the CNN cannot, and vice versa?
8. How exactly do the CNN and GNN connect? (injection of embeddings)
9. Why do static features bypass the convolutions?
10. Why exclude windows with 1–49% attack overlap?
11. What makes the model "inductive," and why does that enable cross-version eval?
12. Why is the dataset synthetic, and what's the trade-off?

**Architecture / numbers:**
13. Walk through the data flow from raw log to risk score.
14. What are the embedding dimensions and how do they sum to 272?
15. What are the 4 modules and which are trained?
16. Describe the graph: node types, edge types, edge features.
17. What's the progressive ladder and the point of it?

**Methodology:**
18. How do you prevent data leakage in cross-validation?
19. What operational metrics beyond AUPRC, and why do they matter to an analyst?
20. What's detection latency and why report it?

**Results (NEW — based on the actual trained numbers, §6.5):**
21. Your CNN gets 0.997 AUPRC — justify that it isn't leakage.
22. XGBoost gets 0.985 with no neural net — why build the CNN/GNN at all?
23. Why does the MLP (0.448) trail XGBoost (0.985) so badly on similar data?
24. Which features drive the XGBoost model, and do they make sense?
25. Why does only scenario 2 show up, and what does that say about coverage?
26. Detection latency is negative — what does that mean and is it good or
    suspicious?
27. Where are the GNN and fusion results, and what do you *expect* them to add?

**Gotchas to be ready for:**
- "Are all three neural nets fused?" → **No.** CNN + GNN fuse; MLP is a separate
  baseline; the fusion *head* is its own MLP.
- "Is it trained?" → **MLP, XGBoost, and the CNN are trained on r4.2** (see §6.5);
  the **GNN and fusion are not yet run**. The live demo still uses the statistical
  baseline. Be upfront about what is and isn't done.
- "Aren't these results too good (0.997)?" → See §6.5(1): legitimate reasons +
  the leakage checklist (user-level splits, train-only baseline/standardizer).
- "Why deep learning if XGBoost hits 0.985?" → See §6.5(2): generality from raw
  data, transferable embeddings, similarity search, relational attacks.
- "Is this real data?" → **Synthetic**, with the trade-offs above.

---

## 9. Repo Facts

- Package: `backend/innersight/`; `from innersight.X import Y`;
  `cd backend && pip install -e .`.
- Run: `make demo` (Docker; frontend `:3000`, backend `:5001`, Qdrant `:6333`).
  Also `make test`, `make lint`, `make train-quick`.
- 79 commits · 202 tests passing · 50 source files · Phases 0–7 done.
- Stack: Python 3.13, pandas/NumPy/pyarrow, PyTorch, torch-geometric (GATv2),
  XGBoost, Qdrant, Flask, React 19/TypeScript.
- GitHub: https://github.com/Enterino8465/innersight.git
- Companion docs: `PHASE_WALKTHROUGH_FOR_PRESENTATION.html`,
  `CONTEXT_PRESENTATION_PREP.md`, `DATASET_CONTEXT.md`, `literature_targets.md`,
  `CONTEXT_FOR_NEW_CONVERSATION.md`.
