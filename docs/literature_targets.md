# Literature Review: CERT Insider Threat Detection

> **Scope & honesty note.** Every metric below is quoted from a source we could
> actually read; anything we could not verify from a primary/secondary source is
> marked **n/r** ("not reported / not verified") rather than guessed. Several
> citations in the original task brief had the wrong year or venue — corrected
> here and flagged with ⚠️. **No verified paper on CERT reports AUPRC**; most
> report AUC‑ROC or budget‑based cumulative recall (CR‑k), both of which behave
> very differently from AUPRC under CERT's extreme class imbalance.

## Summary Table

| Paper | Year | Method | Dataset | Split Type | AUPRC | AUC‑ROC | Notes |
|-------|------|--------|---------|-----------|-------|---------|-------|
| Lindauer, Glasser, Rosen, Wallnau ⚠️ | 2014 | Synthetic data generation (no detector) | Defines the CERT r‑series | — | — | — | Dataset‑generation paper; JoWUA (brief said "2020" — actually **2014**). No metrics. |
| Fangfang Yuan et al. ⚠️ | 2018 | LSTM → CNN (hybrid) | CERT r4.2 † | n/r | **0.9449** | — | "Insider Threat Detection with Deep Neural Network," ICCS 2018. Best‑config AUC; split type/granularity unverified. |
| Shuhan Yuan & Xintao Wu | 2021 | Survey (no model) | r4.2 / r6.2 | — | — | — | *Computers & Security* 104:102221. Argues ROC‑AUC is misleading; recommends **PR‑AUC** & cumulative recall. |
| Le & Zincir‑Heywood ⚠️ | 2018 | Supervised + unsupervised (HMM/SOM/Decision Tree) | CERT (r4.2?) | n/r | n/r | n/r | "Evaluating Insider Threat Detection Workflow…," IEEE SPW (brief said "2019" — title paper is **2018**). Numbers unreachable. |
| Le, Zincir‑Heywood & Heywood | 2020 | RF / LR / NN / LOF, multi‑granularity | CERT (r4.2–r6.2 tooling) | user‑level | n/r | n/r | *IEEE TNSM* 17(1):30–44. Best = RF @ day‑granularity (median‑diff). Exact numbers unverified (PDFs unreachable). |
| Chattopadhyay, Wang & Tan | 2018 | Deep autoencoder / RF / MLP | CERT r4.2 (S1–S3) | **random + 5‑fold** | n/r | n/r | *IEEE TCSS* 5(3):660–675. Reports up to **100 % P/R/F** at best operating point — heavily inflated (see notes). |
| Tuor et al. | 2017 | Online DNN / LSTM‑RNN | CERT **v6.2** | temporal | n/r | n/r | AAAI‑17 workshop. Metric = CR‑k; best **CR‑1000 ≈ 35.6**. Uses v6.2, *not* r4.2/r5.2. |
| Log2vec — Liu et al. | 2019 | Heterogeneous graph embedding + clustering | CERT | n/r | **0.6563** (r4.2) ‡ | — | CCS 2019. Original paper publishes no AUC/F1; ‡ value is LAN's 2024 re‑implementation. |
| ADSAGE — Garchery & Granitzer | 2020 | Attributed graph edges + LSTM | CERT **v6.2** | temporal | n/r | n/r | arXiv:2007.06985. Metric = CR‑k (e.g., logon **CR‑1000 = 0.925**); event→user‑day aggregation. |
| **LAN — Cai et al.** | 2024 | Temporal sequence + adaptive graph neighbors | **CERT r4.2 & r5.2** | **temporal** (train 2010 / test 2011) | n/r | **0.9369–0.9607** (r4.2) | arXiv:2403.09209. Best‑verified r4.2/r5.2 table; reports ~10 baselines under one clean temporal split. |

† CERT r4.2 corroborated by multiple **secondary** sources; the primary PDF was unparseable.
‡ AUC for Log2vec is reported by **LAN (Cai et al. 2024)**'s re‑implementation, not by the original authors.

## Our Targets (honest temporal evaluation on r4.2)

Based on the literature and architectural constraints:

- MLP baseline on z‑scored deviations: **AUPRC 0.15–0.25**
- Temporal CNN on deviation sequences: **AUPRC 0.35–0.50** (biggest jump)
- Temporal CNN + GATv2 graph: **AUPRC 0.45–0.55**
- Full fusion model: **AUPRC 0.50–0.60**
- Realistic ceiling: **0.55–0.65** (limited by ~70 positive examples, not architecture)

> Why these are expressed in AUPRC even though the literature mostly reports
> AUC‑ROC: under CERT r4.2's imbalance (~70 malicious users / ~1,000; malicious
> *events* on the order of 10⁻⁴–10⁻⁶ of all events), AUC‑ROC saturates near 1.0
> while AUPRC stays low and discriminative. A model at AUC‑ROC ≈ 0.94 (Yuan 2018,
> LAN 2024) can still have an AUPRC well under 0.5. Our targets are therefore
> deliberately *not* comparable to the headline AUC numbers above — that is the
> point of the next two sections.

## Key Insight

Many papers report AUC-ROC > 0.95 but use random splits or window-level
metrics that don't reflect real-world detection capability. Our evaluation uses
honest temporal splits where the model never sees future data. This makes our
numbers LOWER but HONEST — which is what a thesis committee respects.

Papers using random train/test splits will show inflated metrics because:
1. Future behavioral patterns leak into training data
2. The same user can appear in both train and test (identity leakage)
3. Window overlap means near-identical samples in train and test

Concrete examples from this review:

- **Chattopadhyay et al. (2018)** reports up to **100 % precision/recall/F‑score**,
  but those are *window‑level* metrics on a *random* 70/50/30 % split with
  *fivefold CV*, after **randomly undersampling** the benign class to a chosen
  imbalance ratio, reported at the **best window‑length × ratio** per scenario,
  and only for scenarios **S1–S3**. This is close to a worst case for every bias
  listed above.
- **AUC‑ROC is the wrong headline metric here.** Yuan & Wu's 2021 survey makes
  this argument explicitly: on CERT, ROC‑AUC values "cluster high" and are weakly
  discriminative; they recommend PR‑AUC and cumulative recall instead.
- **Budget‑recall (CR‑k)** papers (Tuor 2017, ADSAGE 2020) are more honest about
  operational cost (analyst alert budget) but still aren't directly comparable to
  AUPRC, and Tuor/ADSAGE use **v6.2**, not r4.2.
- Beware **post‑2024 "near‑perfect" GNN claims** (e.g., session‑graph GNNs citing
  ~99 % TPR / ~0 % FPR on r4.2). Perfect detection on CERT is a red flag for
  identity leakage, scenario‑subset selection, or random splitting — always check
  the split protocol before trusting the number.

## Detailed Paper Notes

### 1. Lindauer, Glasser, Rosen, Wallnau (2014) — the dataset paper ⚠️

- **Citation:** B. Lindauer, J. Glasser, M. Rosen, K. C. Wallnau. "Generating Test
  Data for Insider Threat Detectors." *Journal of Wireless Mobile Networks,
  Ubiquitous Computing and Dependable Applications (JoWUA)*, 5(2):80–94, 2014.
- **Correction:** the brief listed "2020" and added Mike Theis; the verified
  record is **2014, JoWUA**, authors as above. (A related companion paper is
  Glasser & Lindauer, "Bridging the Gap: A Pragmatic Approach to Generating
  Insider Threat Data," IEEE S&P Workshops 2013.)
- **What it is:** describes the *generation* of the synthetic CERT insider‑threat
  corpus (agent/model‑based simulation of benign org activity + injected malicious
  scenarios) — the lineage that produced r4.2 / r5.2 / r6.2.
- **Method / eval / metrics:** no detector, no split, **no metrics**.
- **Caveat (raised by the authors themselves):** the data is *synthetic* — benign
  behavior and threat scenarios are designed, not observed — which limits external
  validity of any detector trained and tested on it.
- **Use to us:** the canonical citation when we describe where r4.2 comes from and
  why "synthetic" is a standing threat to validity.
- **Sources:** DBLP JoWUA v5; jowua.com PDF; Semantic Scholar.

### 2. Fangfang Yuan et al. (2018) — LSTM‑CNN ⚠️

- **Citation:** F. Yuan, Y. Cao, Y. Shang, Y. Liu, J. Tan, B. Fang. "Insider Threat
  Detection with Deep Neural Network." *ICCS 2018*, LNCS 10860, pp. 43–54.
  DOI 10.1007/978-3-319-93698-7_4.
- **Disambiguation:** this is **Fangfang Yuan** (the experimental LSTM‑CNN paper),
  distinct from **Shuhan Yuan & Xintao Wu**'s 2021 *survey* (entry 3). The brief
  conflated them.
- **Dataset:** CERT **r4.2** (secondary‑verified: ~1,000 users, ~70 insiders,
  ~32.8M events, 7,323 malicious events).
- **Method:** LSTM treats sequences of user actions as a "language" and extracts
  temporal feature vectors; these are reshaped into matrices and classified by a
  CNN. Features from logon/device/file/email/web logs.
- **Evaluation methodology:** **n/r** — split type (random vs temporal vs k‑fold),
  the unit (user‑day vs session vs window), and any rebalancing could not be
  verified from the primary PDF.
- **Key metric:** best‑config **AUC‑ROC = 0.9449** ("LSTM2+CNN3"). **AUPRC: n/r.**
  Secondary mentions of ">90 % precision/accuracy" could not be primary‑verified.
- **Caveats:** headline is AUC‑ROC (optimistic under imbalance); "best case" implies
  selection across configs; split/granularity unknown, so a random split or
  favorable balancing cannot be ruled out.
- **Sources:** Springer chapter; ICCS 2018 PDF; Semantic Scholar.

### 3. Shuhan Yuan & Xintao Wu (2021) — survey

- **Citation:** S. Yuan, X. Wu. "Deep learning for insider threat detection: Review,
  challenges and opportunities." *Computers & Security* 104:102221, 2021.
  DOI 10.1016/j.cose.2021.102221 (preprint arXiv:2005.12433).
- **What it is:** a *review* — no original model or metrics.
- **Dataset coverage:** introduces **r4.2** (dense: 1,000 employees / 70 insiders /
  32.77M activities / 7,323 malicious) and **r6.2** (sparse: 2,500 / 5 / 135.1M /
  470 malicious); both span Jan 2010–Jun 2011 with five log types + psychometrics;
  enumerates the five insider scenarios.
- **Most useful claim for us:** ROC‑AUC values on CERT "cluster high" and are weakly
  discriminative; the survey recommends **PR‑AUC** and **cumulative recall (CR‑k)**
  as the practically meaningful metrics — direct support for our AUPRC‑first stance.
- **Sources:** arXiv:2005.12433 (ar5iv full text); ScienceDirect; author project page.

### 4. Le & Zincir‑Heywood group (2018 SPW / 2019 IM / 2020 TNSM) ⚠️

- **A1 — the exact title in the brief:** D. C. Le, A. N. Zincir‑Heywood,
  "Evaluating Insider Threat Detection Workflow Using Supervised and Unsupervised
  Learning," *IEEE SPW* 2018, pp. 270–275, DOI 10.1109/SPW.2018.00043.
  **Correction:** this is **2018**, not 2019. Method = HMM / SOM / Decision Tree,
  combining unsupervised anomaly detection with supervised classification.
  Abstract claims "high detection, low false positive," but **numeric results were
  unreachable** (PDFs timed out; indexers returned 403/418) → **n/r**.
- **A2 — 2019 conference precursor:** D. C. Le, A. N. Zincir‑Heywood, "Machine
  learning based Insider Threat Modelling and Detection," *IFIP/IEEE IM* 2019. A
  "user‑centered" multi‑granularity system; details **n/r**.
- **A3 — the strongest‑methodology version:** D. C. Le, A. N. Zincir‑Heywood,
  M. I. Heywood, "Analyzing Data Granularity Levels for Insider Threat Detection
  Using Machine Learning," *IEEE TNSM* 17(1):30–44, 2020,
  DOI 10.1109/TNSM.2020.2967721.
  - **Method:** RF / Logistic Regression / Neural Network (+ XGBoost), and
    unsupervised LOF, across four granularities — **user‑week, user‑day,
    user‑session, sub‑session** — with temporal representations (median‑differential,
    concatenation). Their public tooling (`lcd-dal/feature-extraction-for-CERT…`)
    supports r4.1–r6.2.
  - **Eval:** user‑centered, multi‑granularity, "realistic conditions" incl.
    detection on *unseen* users/insiders. Qualitative result: **RF @ day‑granularity
    with median‑differential** performs best; LOF + week‑concatenation gives highest
    detection rate.
  - **Metrics:** exact AUC/DR/FPR table values **n/r** (primary PDF unreachable).
    ⚠️ A snippet quoting "AUC 0.94 / TPR >95 % / FPR <10 %" actually refers to the
    **WUIL** dataset cited in this paper's related work — do **not** attribute it to
    this CERT result.
- **Use to us:** the closest prior art to an honest, user‑level, multi‑granularity
  evaluation — worth citing for methodology even though we couldn't pin the numbers.
- **Sources:** bibbase; J‑GLOBAL (TNSM metadata); lcd‑dal GitHub; IEEE/DOI listings.

### 5. Chattopadhyay, Wang & Tan (2018) — scenario‑based, *inflated*

- **Citation:** P. Chattopadhyay, L. Wang, Y.‑P. Tan. "Scenario‑Based Insider Threat
  Detection From Cyber Activities." *IEEE Transactions on Computational Social
  Systems* 5(3):660–675, 2018. DOI 10.1109/TCSS.2018.2857473. (Verified from the
  primary PDF.)
- **Dataset:** CERT **r4.2** explicitly (1,000 users, Jan 2010–May 2011); scenario
  experiments cover **only S1, S2, S3** (plus one scenario‑agnostic experiment).
- **Method:** per‑scenario supervised time‑series classification. Single‑day activity
  features aggregated over a window into time‑series stats (mean, variance, Katz
  fractal dimension…); scenario‑specific dims (48/16/68 for S1/S2/S3). Classifier =
  two‑layer deep autoencoder, compared to RF and MLP.
- **Evaluation methodology:** **window‑level** instances; **random** split (r = 70/50/30 %
  train) plus **fivefold CV**; benign class **randomly undersampled** to target
  imbalance ratios (C1–C7); metrics reported at the **optimal window × ratio** per
  scenario.
- **Key metric:** precision/recall/F‑score (no AUPRC, no AUC). At best config the
  metrics hit **100 %** (e.g., S1 @ 40‑day window + C7; S3 @ 10‑day + C6); autoencoder
  ≈ RF, both high; MLP high recall but lower precision.
- **Caveats:** random split (leakage), window‑level (not user‑level), aggressive
  undersampling, best‑operating‑point cherry‑picking, S1–S3 only, and no full
  PR/ROC curve. **This is our canonical example of "inflated by methodology."**
- **Source:** NTU‑hosted primary PDF (read in full).

### 6. Tuor et al. (2017) — online DNN/LSTM (v6.2)

- **Citation:** A. Tuor, S. Kaplan, B. Hutchinson, N. Nichols, S. Robinson. "Deep
  Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity
  Data Streams." *AAAI‑17 Workshop on AI for Cyber Security*, 2017. arXiv:1710.00811.
- **Dataset:** CERT **v6.2** (⚠️ not r4.2/r5.2 — does not match our target release).
- **Method:** online/streaming **DNN** and **LSTM‑RNN** anomaly detectors; one shared
  model with per‑user hidden state; best variants use diagonal‑covariance Gaussian
  output ("DNN‑Diag"/"LSTM‑Diag"). Features = 414‑dim per‑user‑per‑day vectors.
- **Evaluation methodology:** **temporal** split — train days 1–418 (85 %), test
  419–516 (15 %); **user‑day** predictions; online updates; weekdays only.
- **Key metric:** budget recall **CR‑1000 ≈ 35.6** (DNN/LSTM‑Diag) vs Isolation Forest
  34.8, PCA 32.8, SVM 24.2; threat events averaged the 95.53rd percentile anomaly
  score. No AUC/AUPRC.
- **Caveats:** synthetic; budget‑recall not directly comparable to AUPRC; v6.2 is
  very sparse; user‑day aggregation hides event‑level behavior.
- **Sources:** arXiv:1710.00811 (ar5iv full text).

### 7. Log2vec — Liu et al. (2019) — heterogeneous graph embedding

- **Citation:** F. Liu, Y. Wen, D. Zhang, X. Jiang, X. Xing, D. Meng. "Log2vec: A
  Heterogeneous Graph Embedding Based Approach for Detecting Cyber Threats within
  Enterprise." *ACM CCS 2019*, pp. 1777–1794. DOI 10.1145/3319535.3363224.
- **Method:** heuristic rules turn log entries into a **heterogeneous graph**; an
  improved random‑walk/word2vec‑style embedding maps each entry to a vector; a
  clustering algorithm separates malicious vs benign entries (per‑user,
  log‑entry‑level).
- **Evaluation / metrics (original):** the abstract claims it "remarkably
  outperforms" DL and HMM baselines, but **no AUC/F1 numbers were verifiable** from
  any readable source → **n/r**.
- **Independently reported (LAN 2024 re‑implementation):** Log2vec AUC **0.6563**
  (r4.2) / **0.6178** (r5.2); DR 0.6793/0.6441; FPR 0.3022/0.3388 — i.e., *low*
  relative to modern sequence baselines.
- **Caveats:** hand‑crafted heuristic edges (manual feature engineering); original
  strong claims weakly sourced; cannot produce a ranked alert budget (DR@k = "–").
- **Sources:** ACM DL (403 on full text); Semantic Scholar; LAN arXiv:2403.09209.

### 8. ADSAGE — Garchery & Granitzer (2020) — attributed graph edges

- **Citation:** M. Garchery, M. Granitzer. "ADSAGE: Anomaly Detection in Sequences of
  Attributed Graph Edges applied to insider threat detection at fine‑grained level."
  arXiv:2007.06985, 2020 (preprint).
- **Dataset:** CERT **v6.2** (one instance of each of the 5 scenarios) + LANL auth.
- **Method:** events as **directed attributed graph edges** (user→computer,
  sender→recipients, user→domain) with learned source/destination embeddings +
  temporal/categorical attributes; per‑user LSTM + FFNN; negative‑sampling training.
  (Despite the name, *not* GraphSAGE.)
- **Evaluation methodology:** **temporal** split; fine‑grained/event‑level detection
  aggregated to **user‑day**; metrics = Recall@budget and Cumulative Recall@k.
- **Key metric:** CERT v6.2 logon **CR‑1000 = 0.925 ± 0.069**, CR‑4000 = 0.981; email
  CR‑1000 = 0.646; web CR‑4000 = 0.696. No AUC/AUPRC/F1.
- **Caveats:** synthetic; ~10⁻⁶ event‑level imbalance; trivial rule baselines
  ("Own PC"/"Known PC") are competitive; v6.2 not r4.2.
- **Sources:** arXiv:2007.06985 (ar5iv full text).

### 9. LAN — Cai et al. (2024) — best‑verified r4.2/r5.2 numbers

- **Citation:** X. Cai, Y. Wang, S. Xu, H. Li, Y. Zhang, Z. Liu, X. Yuan. "LAN:
  Learning Adaptive Neighbors for Real‑Time Insider Threat Detection."
  arXiv:2403.09209, 2024 (IEEE TIFS track).
- **Dataset:** **CERT r4.2 and r5.2** (Jan 2010–Jun 2011).
- **Method:** activity‑level model combining temporal sequence modeling with **graph
  structure learning ("adaptive neighbors")** over activity‑representation vectors
  (activity code = type × 24 + hour slot), plus an imbalance‑aware loss.
- **Evaluation methodology:** **temporal** split — train/validate on 2010, **test on
  Jan–Jun 2011**; activity/instance‑level; two settings (real‑time causal & post‑hoc
  bidirectional). Metrics = AUC, DR, FPR, DR@5/10/15 %.
- **Key metrics (verified from the paper's tables):**
  - *Real‑time (Table II):* LAN AUC **0.9369 (r4.2)**, **0.9439 (r5.2)**;
    FPR 0.1411 / 0.0867. Baselines r4.2: DeepLog 0.7469, Transformer 0.7981,
    FMLP 0.8526; r5.2: DeepLog 0.8549, OC4Seq 0.9202.
  - *Post‑hoc (Table III):* LAN AUC **0.9607 (r4.2)**, **0.9605 (r5.2)**;
    DR 0.9478 / 0.9024; DR@15 % 0.9739 / 0.9346. Log2vec here: 0.6563 / 0.6178.
    Other r4.2 baselines: RNN 0.8652, FMLP 0.8837, ITDBERT 0.7413, OC4Seq 0.8113.
- **Caveats:** synthetic CERT (authors acknowledge limited real‑world generality);
  **only AUC reported, no AUPRC** (AUC optimistic under imbalance); baseline numbers
  (incl. Log2vec) are the authors' own re‑implementations.
- **Why it matters to us:** the single best reference point — a *temporal* split on
  **r4.2 and r5.2** with ~10 baselines in one table. Our AUPRC targets sit "below"
  its AUC of ~0.94–0.96 precisely because AUPRC and AUC‑ROC are not comparable under
  this imbalance, not because our models are weaker.
- **Source:** arXiv:2403.09209.

---

### Cross‑cutting takeaways

1. **No verified CERT paper reports AUPRC.** Detection papers report AUC‑ROC
   (Yuan 2018, LAN 2024), budget recall CR‑k (Tuor 2017, ADSAGE 2020), or
   precision/recall/F at a chosen operating point (Chattopadhyay 2018). Our AUPRC
   targets are therefore *new ground for honest comparison*, not a lowball of any
   published AUPRC.
2. **AUC‑ROC on synthetic CERT is inflated** by extreme imbalance; ~0.94–0.96 AUC
   does not imply usable precision at a realistic alert budget.
3. **Split protocol dominates the headline number.** Random‑split, window‑level,
   undersampled, best‑operating‑point evaluations (Chattopadhyay) yield ~100 %;
   temporal, user/activity‑level evaluations (Tuor, ADSAGE, LAN) yield more sober
   numbers. We adopt the latter.
4. **Best anchors for a thesis comparison on r4.2/r5.2:** LAN (Cai et al. 2024) for
   modern temporal AUC + baseline table; Le/Zincir‑Heywood TNSM 2020 for
   user‑level multi‑granularity methodology; Yuan & Wu 2021 for the metric‑choice
   argument; Lindauer 2014 for the dataset's provenance and synthetic‑data caveat.
