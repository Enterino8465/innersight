"""Cross-version generalization evaluation (Phase 6).

Evaluates a model trained on one CERT version against a *different* version's
data. This works because the model is inductive: the temporal encoder scores a
window independently of user identity, and the graph encoder keys on the
windowed-graph schema (the same 4 node / 10 edge types in every version), so it
runs unchanged on the target version's completely different users and graph.

The one version-specific component is the static role/department embedding
tables — their integer ids are assigned per version, so they do not transfer.
Target users are therefore scored with the "unknown" embedding index (0); the
OCEAN scores, being numeric, are recomputed from the target's psychometric data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.data.pipeline import load_version
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.fusion_model import InsiderThreatDetector
from innersight.scripts import compute_baselines
from innersight.scripts.train_fusion import _build_ocean_map, _predict
from innersight.scripts.train_temporal_graph import _build_period_graphs, _build_window_registry
from innersight.training.evaluation import compute_metrics, detection_latency, per_scenario_metrics

logger = logging.getLogger(__name__)

_OCEAN_DIM = 5


def evaluate_cross_version(
    checkpoint_path: str,
    source_version: str,
    target_data_dir: str,
    target_version: str,
    store_dir: str = "feature_store",
) -> dict:
    """Evaluate a *source*-trained fusion model on *target*-version data.

    Args:
        checkpoint_path: Path to a fusion (InsiderThreatDetector) checkpoint.
        source_version: Version the model was trained on (recorded in results).
        target_data_dir: Dataset directory for the target version.
        target_version: Version to evaluate against.
        store_dir: Feature store directory (target deviations computed if absent).

    Returns:
        Results dict with AUPRC and the full metric set, per-scenario breakdown,
        and detection latency on the target version.
    """
    # ── Reconstruct the model from the checkpoint config ──────────────────────
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    config = ckpt.get("config", {})
    temporal_cfg = config.get("temporal") or None
    graph_cfg = config.get("graph") or None
    num_roles = int(config.get("num_roles", 50))
    num_depts = int(config.get("num_depts", 15))

    # ── Target deviations (compute from raw data if not cached) ───────────────
    store = FeatureStore(store_dir)
    deviations = store.load_deviations(target_version)
    if deviations is None:
        logger.info("evaluate_cross_version | computing target deviations for %s …", target_version)
        rc = compute_baselines.main(
            ["--version", target_version, "--data-dir", target_data_dir, "--store-dir", store_dir])
        if rc != 0:
            raise RuntimeError(f"compute_baselines failed for target version {target_version}")
        deviations = store.load_deviations(target_version)

    # ── Target labels, raw logs (for graphs), psychometric ────────────────────
    answers_dir = Path(target_data_dir) / "answers"
    attack_windows = (
        {r.user_id: r for r in load_insiders(answers_dir, target_version)}
        if answers_dir.exists() else {}
    )
    logger.info("evaluate_cross_version | loading target raw logs and psychometric …")
    target_dataset = load_version(target_data_dir, target_version)
    ocean_map = _build_ocean_map(target_dataset.psychometric)

    win_dataset = DeviationWindowDataset(deviations, attack_windows)
    windows_t, user_ids, period_keys, y, metas = _build_window_registry(win_dataset)
    if len(y) == 0:
        logger.warning("evaluate_cross_version | no target windows produced.")
        return {"source_version": source_version, "target_version": target_version,
                "n_windows": 0, "n_positive": 0, "metrics": compute_metrics(np.array([]), np.array([]))}

    period_graphs = _build_period_graphs(target_dataset.logs, period_keys, exclude_users=None)
    metadata = next(iter(period_graphs.values())).metadata()

    model = InsiderThreatDetector(metadata, num_roles=num_roles, num_depts=num_depts,
                                  temporal_config=temporal_cfg, graph_config=graph_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Static context: OCEAN from target; role/dept unknown (vocab is per-version) ──
    ocean = np.stack([ocean_map.get(u, np.zeros(_OCEAN_DIM)) for u in user_ids]).astype(np.float32)
    roles = np.zeros(len(user_ids), dtype=np.int64)  # unknown index — source vocab does not map
    depts = np.zeros(len(user_ids), dtype=np.int64)

    registry = {"windows": windows_t, "user_ids": user_ids, "periods": period_keys, "y": y,
                "ocean": ocean, "roles": roles, "depts": depts}

    # ── Inference over every target window ────────────────────────────────────
    probs = _predict(model, np.arange(len(y)), registry, period_graphs)

    metrics = compute_metrics(probs, y)
    scenarios = np.array([m.get("scenario", 0) for m in metas])
    scenario_metrics = per_scenario_metrics(probs, y, scenarios)
    latency = detection_latency(probs, metas, metrics["threshold_best"], attack_windows)
    logger.info("evaluate_cross_version | %s→%s: AUPRC=%.4f | %d/%d insiders detected.",
                source_version, target_version, metrics["auprc"],
                latency["detected_count"], latency["total_insiders"])

    return {
        "source_version": source_version,
        "target_version": target_version,
        "checkpoint": str(checkpoint_path),
        "n_windows": int(len(y)),
        "n_positive": int(y.sum()),
        "static_note": "role/department embeddings use the unknown index "
                       "(vocabularies are version-specific and do not transfer)",
        "metrics": metrics,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
    }
