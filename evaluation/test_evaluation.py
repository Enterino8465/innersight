"""Tests for Stage 7. Written against the ways this stage produces a *number*.

Stage 7 cannot fail loudly in the interesting cases. A refitted standardiser,
a re-swept threshold, an off-by-one in "day N of the attack", a v2 checkpoint
in the comparison table — every one of them produces a plausible figure and no
exception. So the tests here are mostly about the guards and the arithmetic,
not about happy paths.

The synthetic corpus below is the workhorse: a handful of users on a dense
calendar, one of them an insider with a known attack window, so that the
expected first-detection day can be written down by hand and compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import (
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    MODEL_A_NAME,
    MODEL_B_NAME,
    N_FEATURES,
    SCENARIO_COLUMN,
    USER_COLUMN,
    WINDOW_DAYS,
)
from evaluation.checkpoints import (
    Checkpoint,
    CheckpointError,
    headline_checkpoints,
    load_checkpoint,
    load_variant_checkpoint,
)
from evaluation.corpus import Corpus, EvaluationError, positions_within
from evaluation.falsepositives import (
    BENIGN_USER,
    INSIDE_WINDOW_GAP,
    NEAR_WINDOW,
    read,
    top_false_positives,
)
from evaluation.guards import (
    beats_random,
    finite_predictions,
    identical_rows,
    not_suspiciously_good,
    split_disjointness,
    standardiser_from_checkpoint,
    threshold_from_checkpoint,
)
from evaluation.latency import (
    BUCKETS,
    attack_windows,
    detections,
    read_verdict,
    recall_by_position,
    summarise,
)
from evaluation.scoring import describe_at_threshold, score
from evaluation.testpass import PhaseOrderError, record_opening, require_phase1
from modeling.data import Standardiser
from modeling.models import ModelA, ModelB

DAYS = 90
ATTACK_START = 40   # 0-based day index within the record
ATTACK_LENGTH = 20
#: Above every probability the stub model can produce. The planted signal
#: standardises to roughly z = 15, and sigmoid(15) is 1 - 3e-7, so a cutoff
#: this close to 1 flags nothing — which is how "never detected" is tested.
ABOVE_EVERYTHING = 0.999999999999


def _frame(n_benign: int = 6) -> pd.DataFrame:
    """A dense synthetic corpus: benign users plus one insider per scenario.

    Every user has ``DAYS`` consecutive calendar days, which is what
    ``modeling.data.day_index`` requires and what makes "day N of the attack"
    well defined. Feature 5 carries the signal so the tests can produce a
    model that ranks attack days highly without training one.
    """
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2010-01-04")
    users = [f"BEN{i:04d}" for i in range(n_benign)] + [
        "INS0001", "INS0002", "INS0003", "INS0004",
    ]
    # Two insiders land in validation and two in test, so the split stays
    # disjoint by user — the property the disjointness guard exists to check.
    scenario_of = {"INS0001": 1, "INS0002": 2, "INS0003": 1, "INS0004": 2}

    rng = np.random.default_rng(0)
    for user in users:
        scenario = scenario_of.get(user, 0)
        for day in range(DAYS):
            values = rng.normal(0.0, 0.4, size=N_FEATURES)
            inside = scenario and ATTACK_START <= day < ATTACK_START + ATTACK_LENGTH
            if inside:
                # Decaying signal, mirroring Stage 3's measured 0.40x falloff
                # across an attack window — so latency tests exercise the case
                # where late days are quieter, not louder.
                position = day - ATTACK_START
                values[5] += 6.0 * (1.0 - 0.6 * position / ATTACK_LENGTH)
            row: dict[str, object] = {
                USER_COLUMN: user,
                DAY_COLUMN: start + pd.Timedelta(days=day),
                LABEL_COLUMN: int(bool(inside)),
                SCENARIO_COLUMN: scenario,
            }
            row.update(dict(zip(FEATURE_NAMES, values.astype(np.float32))))
            rows.append(row)
    return pd.DataFrame(rows)


def _splits(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """One seed's split, disjoint by user, with insiders in val and in test."""
    benign = sorted(u for u in frame[USER_COLUMN].unique() if u.startswith("BEN"))
    return {
        "7": {
            "train": benign[:2],
            "val": benign[2:4] + ["INS0001", "INS0002"],
            "test": benign[4:] + ["INS0003", "INS0004"],
        }
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    frame = _frame()
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({"splits": _splits(frame)}))
    return Corpus.load(frame=frame, splits_file=splits_file, window_days=WINDOW_DAYS)


@pytest.fixture
def standardiser(corpus: Corpus) -> Standardiser:
    return corpus.refit_train_standardiser(7)


class _FeatureFiveModel(torch.nn.Module):
    """Scores a row by feature 5 alone. Deterministic, and no training needed.

    A stand-in for Model A that lets the latency and false-positive tests set
    up an exactly known ranking, which a trained model cannot give them.
    """

    n_features = N_FEATURES

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 5] * self.scale


def _checkpoint(standardiser: Standardiser, threshold: float, tmp_path: Path) -> Checkpoint:
    """A hand-built Checkpoint wrapping :class:`_FeatureFiveModel`."""
    return Checkpoint(
        path=tmp_path / f"{MODEL_A_NAME}_seed7.pt",
        model_name=MODEL_A_NAME,
        model_class="ModelA",
        input_kind="point",
        window_days=1,
        threshold=threshold,
        standardiser=standardiser,
        seed=7,
        device_trained_on="cpu",
        n_parameters=0,
        best_val_auprc=None,
        val_random_baseline=None,
        validation_threshold_selection=None,
        state_dict={},
    )


def _scored(corpus: Corpus, standardiser: Standardiser, tmp_path: Path, threshold: float):
    from evaluation import scoring

    checkpoint = _checkpoint(standardiser, threshold, tmp_path)
    split = corpus.build("val", 7, standardiser)
    model = _FeatureFiveModel()
    probs = torch.sigmoid(
        model(split.data.point_view().tensor)
    ).numpy().astype(np.float64)
    applied = describe_at_threshold(split.labels, probs, threshold)
    from modeling.metrics import evaluate

    return scoring.Scored(
        checkpoint=checkpoint,
        split=split,
        probs=probs,
        evaluation=evaluate(
            split.labels, probs, split.scenarios,
            choose_threshold=False, threshold=applied,
        ),
        threshold=threshold,
    )


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


class TestCorpus:
    def test_target_mask_costs_the_warm_up_and_no_positives(self, corpus: Corpus) -> None:
        # The whole comparison rests on both models predicting the same rows.
        # WINDOW_DAYS - 1 rows per user are warm-up and belong to neither.
        n_users = corpus.frame[USER_COLUMN].nunique()
        assert (~corpus.is_target).sum() == n_users * (WINDOW_DAYS - 1)
        labels = corpus.frame[LABEL_COLUMN].to_numpy()
        assert labels[~corpus.is_target].sum() == 0

    def test_days_align_with_the_feature_tensor(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        split = corpus.build("val", 7, standardiser)
        assert split.days.size == split.n_rows == split.data.n_targets
        assert np.array_equal(split.users, split.data.users)
        assert np.array_equal(split.labels, split.data.labels.numpy().astype(np.int64))

    def test_build_applies_the_standardiser_given_not_one_it_fits(
        self, corpus: Corpus
    ) -> None:
        # Hand it a deliberately wrong transform. If the module quietly fitted
        # its own, the features would come out standardised anyway.
        wrong = Standardiser(
            mean=np.full(N_FEATURES, 100.0), std=np.full(N_FEATURES, 0.5)
        )
        split = corpus.build("val", 7, wrong)
        assert split.data.features.mean().item() < -100.0

    def test_unknown_split_is_refused(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        with pytest.raises(EvaluationError, match="unknown split"):
            corpus.build("holdout", 7, standardiser)

    def test_raw_features_are_unstandardised_z_scores(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        split = corpus.build("val", 7, standardiser)
        # The synthetic insider's peak sits near z=6 in Stage 3's units. If
        # raw_features had been standardised, it would be near 1.
        assert split.raw_features[:, 5].max() > 3.0

    def test_fingerprint_detects_a_reordering(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        split = corpus.build("val", 7, standardiser)
        from dataclasses import replace

        shuffled = replace(split, users=split.users[::-1].copy())
        assert shuffled.fingerprint() != split.fingerprint()


class TestPositions:
    def test_day_one_is_the_windows_first_day(self) -> None:
        first = np.datetime64("2010-02-13")
        days = np.array(["2010-02-13", "2010-02-14", "2010-03-15"], dtype="datetime64[D]")
        assert positions_within(days, first).tolist() == [1, 2, 31]


# --------------------------------------------------------------------------
# checkpoints — the §3 refusal and the shape checks
# --------------------------------------------------------------------------


def _write_checkpoint(
    path: Path, *, model_class: str = "ModelA", input_kind: str = "point",
    window_days: int = 1, threshold: float = 0.5, days: int = WINDOW_DAYS,
) -> Path:
    model = ModelA() if model_class == "ModelA" else ModelB(days=days)
    torch.save(
        {
            "model_class": model_class,
            "state_dict": model.state_dict(),
            "input_kind": input_kind,
            "window_days": window_days,
            "threshold": threshold,
            "threshold_selection": {
                "threshold": threshold, "f1": 0.4, "precision": 0.3,
                "recall": 0.6, "predicted_positives": 100,
            },
            "standardiser": Standardiser(
                mean=np.zeros(N_FEATURES), std=np.ones(N_FEATURES)
            ).to_dict(),
            "seed": 42,
            "device": "cpu",
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "best_val_auprc": 0.15,
            "val_random_baseline": 0.004,
        },
        path,
    )
    return path


class TestCheckpointLoading:
    def test_v2_cannot_be_loaded_at_all(self, tmp_path: Path) -> None:
        # Brief §3. The tuning attempt lost; it must not reach a table.
        path = _write_checkpoint(
            tmp_path / "model_b_v2_seed42.pt", model_class="ModelB",
            input_kind="window", window_days=WINDOW_DAYS,
        )
        with pytest.raises(CheckpointError, match="not one of the reported"):
            load_checkpoint(path)
        with pytest.raises(CheckpointError, match="v2"):
            load_variant_checkpoint(path, model_name=MODEL_B_NAME)

    def test_variant_tags_are_refused_by_the_headline_loader(self, tmp_path: Path) -> None:
        for name in ("model_a_nopw_seed42.pt", "model_b_w7_seed42.pt"):
            path = _write_checkpoint(tmp_path / name)
            with pytest.raises(CheckpointError, match="not one of the reported"):
                load_checkpoint(path)

    def test_headline_names_load(self, tmp_path: Path) -> None:
        path = _write_checkpoint(tmp_path / "model_a_seed42.pt", threshold=0.9)
        checkpoint = load_checkpoint(path)
        assert checkpoint.model_name == MODEL_A_NAME
        assert checkpoint.threshold == pytest.approx(0.9)
        assert checkpoint.context_days == 1
        assert isinstance(checkpoint.build(), ModelA)

    def test_sample_names_load(self, tmp_path: Path) -> None:
        path = _write_checkpoint(tmp_path / "model_a_seed42.sample.pt")
        assert load_checkpoint(path).seed == 42

    def test_mismatched_input_kind_is_refused(self, tmp_path: Path) -> None:
        # A ModelB stamped "point" would be handed (batch, 18) tensors and
        # raise inside forward — but only after the run had started.
        path = _write_checkpoint(
            tmp_path / "model_b_seed42.pt", model_class="ModelB",
            input_kind="point", window_days=WINDOW_DAYS,
        )
        with pytest.raises(CheckpointError, match="inconsistent"):
            load_checkpoint(path)

    def test_degenerate_threshold_is_refused(self, tmp_path: Path) -> None:
        path = _write_checkpoint(tmp_path / "model_a_seed42.pt", threshold=1.0)
        with pytest.raises(CheckpointError, match="not a probability"):
            load_checkpoint(path)

    def test_missing_keys_are_named(self, tmp_path: Path) -> None:
        path = tmp_path / "model_a_seed42.pt"
        torch.save({"model_class": "ModelA"}, path)
        with pytest.raises(CheckpointError, match="missing"):
            load_checkpoint(path)

    def test_b7_window_length_is_taken_from_the_caller_not_the_file(
        self, tmp_path: Path
    ) -> None:
        # runner.save_checkpoint stamps WINDOW_DAYS on every window model, so
        # a 7-day control claims 28 on disk. The loader must not believe it.
        path = _write_checkpoint(
            tmp_path / "model_b_w7_seed42.pt", model_class="ModelB",
            input_kind="window", window_days=WINDOW_DAYS, days=7,
        )
        checkpoint = load_variant_checkpoint(
            path, model_name=MODEL_B_NAME, expected_window_days=7
        )
        assert checkpoint.window_days == 7
        assert checkpoint.context_days == 7

    def test_headline_pair_refuses_mismatched_seeds(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path / "model_a_seed42.pt")
        _write_checkpoint(
            tmp_path / "model_b_seed42.pt", model_class="ModelB",
            input_kind="window", window_days=WINDOW_DAYS,
        )
        pair = headline_checkpoints(42, directory=tmp_path)
        assert set(pair) == {MODEL_A_NAME, MODEL_B_NAME}
        assert pair[MODEL_B_NAME].context_days == WINDOW_DAYS


# --------------------------------------------------------------------------
# scoring — the threshold is applied, never chosen
# --------------------------------------------------------------------------


class TestThresholdIsApplied:
    def test_describe_at_threshold_uses_the_value_given(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.6, 0.4, 0.9])
        # A sweep would pick something that separates 0.9 alone (F1 = 0.67).
        # Handed 0.5, this must report what 0.5 actually does: 2 flagged, one
        # of them right.
        described = describe_at_threshold(labels, probs, 0.5)
        assert described.threshold == 0.5
        assert described.predicted_positives == 2
        assert described.precision == pytest.approx(0.5)
        assert described.recall == pytest.approx(0.5)

    def test_no_rows_above_the_threshold_is_zero_not_a_crash(self) -> None:
        described = describe_at_threshold(np.array([0, 1]), np.array([0.1, 0.2]), 0.99)
        assert described.predicted_positives == 0
        assert described.precision == 0.0
        assert described.f1 == 0.0

    def test_scoring_never_reselects(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.9)
        assert scored.threshold == 0.9
        assert scored.evaluation.threshold is not None
        assert scored.evaluation.threshold.threshold == 0.9

    def test_score_runs_a_real_model_through_the_training_path(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        path = _write_checkpoint(tmp_path / "model_a_seed7.pt", threshold=0.5)
        checkpoint = load_checkpoint(path)
        split = corpus.build("val", 7, checkpoint.standardiser)
        scored = score(checkpoint, split, device="cpu")
        assert scored.probs.shape == (split.n_rows,)
        assert np.isfinite(scored.probs).all()


# --------------------------------------------------------------------------
# latency — the arithmetic the headline claim rests on
# --------------------------------------------------------------------------


class TestAttackWindows:
    def test_window_is_derived_from_the_labels(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        split = corpus.build("val", 7, standardiser)
        windows = attack_windows(split)
        assert set(windows) == {"INS0001", "INS0002"}
        window = windows["INS0001"]
        assert window.n_labelled_days == ATTACK_LENGTH
        assert window.span_days == ATTACK_LENGTH
        assert window.contiguous

    def test_benign_split_has_no_windows(self, corpus: Corpus, standardiser: Standardiser) -> None:
        benign = [u for u in corpus.frame[USER_COLUMN].unique() if u.startswith("BEN")]
        split = corpus.build("val", 7, standardiser, users=benign[:2])
        assert attack_windows(split) == {}


class TestDetections:
    def test_first_detection_day_is_one_based_from_the_window_start(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        # Feature 5 peaks on the attack's first day and decays, so a low
        # threshold must catch day 1 exactly.
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        records = {d.user: d for d in detections(scored)}
        assert records["INS0001"].detected
        assert records["INS0001"].first_detection_day == 1
        assert records["INS0001"].n_attack_days == ATTACK_LENGTH

    def test_an_undetectable_insider_is_counted_not_dropped(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=ABOVE_EVERYTHING)
        records = detections(scored)
        assert records, "insiders must still appear when nothing is flagged"
        assert all(not r.detected for r in records)
        assert all(r.first_detection_day is None for r in records)

    def test_summary_separates_never_caught_from_the_median(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=ABOVE_EVERYTHING)
        summary = summarise(detections(scored), MODEL_A_NAME)
        assert summary.n_insiders == 2
        assert summary.n_detected == 0
        assert summary.n_never_detected == 2
        assert summary.median_first_day is None


class TestPositionBuckets:
    def test_buckets_partition_positions_without_overlap(self) -> None:
        positions = np.arange(1, 60)
        counted = sum(int(bucket.contains(positions).sum()) for bucket in BUCKETS)
        assert counted == positions.size

    def test_bucket_totals_equal_the_attack_days(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        buckets = recall_by_position([scored])
        total = sum(b.total for b in buckets if b.model == MODEL_A_NAME)
        assert total == 2 * ATTACK_LENGTH

    def test_verdict_reads_a_climbing_profile_as_persistence(self) -> None:
        from evaluation.latency import BucketRecall, LatencySummary

        buckets = [
            BucketRecall(MODEL_B_NAME, None, "days 1-7", 1, 10),
            BucketRecall(MODEL_B_NAME, None, "days 29+", 9, 10),
        ]
        summary = LatencySummary(MODEL_B_NAME, None, 10, 8, 31.0, 25.0, 38.0, 20, 44)
        verdict = read_verdict(buckets, summary)
        assert verdict.reading == "persistence"
        assert "ongoing" in verdict.claim

    def test_verdict_reads_a_flat_early_profile_as_onset(self) -> None:
        from evaluation.latency import BucketRecall, LatencySummary

        buckets = [
            BucketRecall(MODEL_B_NAME, None, "days 1-7", 8, 10),
            BucketRecall(MODEL_B_NAME, None, "days 29+", 8, 10),
        ]
        summary = LatencySummary(MODEL_B_NAME, None, 10, 10, 2.0, 1.0, 3.0, 1, 6)
        verdict = read_verdict(buckets, summary)
        assert verdict.reading == "onset"


# --------------------------------------------------------------------------
# false positives — §8
# --------------------------------------------------------------------------


class TestFalsePositives:
    def test_rank_is_over_all_rows_not_only_benign_ones(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        rows = top_false_positives(scored, k=5)
        assert rows
        assert rows[0].rank >= 1
        assert all(row.rank < rows[i + 1].rank for i, row in enumerate(rows[:-1]))

    def test_all_eighteen_z_scores_are_reported(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        rows = top_false_positives(_scored(corpus, standardiser, tmp_path, 0.5), k=3)
        assert all(len(row.z_scores) == N_FEATURES for row in rows)
        assert all(set(row.z_scores) == set(FEATURE_NAMES) for row in rows)

    def test_reading_classifies_benign_users_as_a_real_failure_mode(self) -> None:
        from evaluation.falsepositives import FalsePositive

        rows = [
            FalsePositive(
                MODEL_B_NAME, 123, rank, "BEN0001", "2010-03-01", 0.99, 0,
                BENIGN_USER, None, {name: 0.0 for name in FEATURE_NAMES},
            )
            for rank in range(1, 4)
        ]
        assert "genuine false positives" in read(rows).sentence

    def test_reading_calls_precision_a_lower_bound_when_labels_are_narrow(self) -> None:
        from evaluation.falsepositives import FalsePositive

        rows = [
            FalsePositive(
                MODEL_B_NAME, 123, rank, "INS0001", "2010-03-01", 0.99, 2,
                INSIDE_WINDOW_GAP if rank % 2 else NEAR_WINDOW, rank,
                {name: 0.0 for name in FEATURE_NAMES},
            )
            for rank in range(1, 6)
        ]
        assert "lower bound" in read(rows).sentence


# --------------------------------------------------------------------------
# guards — the §13 list
# --------------------------------------------------------------------------


class TestGuards:
    def test_disjointness_passes_on_a_clean_split(self, corpus: Corpus) -> None:
        assert split_disjointness(corpus, [7]).passed

    def test_disjointness_catches_a_test_user_in_train(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        leaky = _splits(corpus.frame)
        leaky["7"]["train"] = leaky["7"]["train"] + ["INS0003"]
        path = tmp_path / "leaky.json"
        path.write_text(json.dumps({"splits": leaky}))
        from dataclasses import replace

        result = split_disjointness(replace(corpus, splits_file=path), [7])
        assert not result.passed
        assert "INS0003" in result.detail

    def test_standardiser_guard_passes_when_the_checkpoint_matches(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        result = standardiser_from_checkpoint(
            corpus, 7, [_checkpoint(standardiser, 0.5, tmp_path)]
        )
        assert result.passed
        assert "mean[0]" in result.detail  # §13 asks for the values printed

    def test_standardiser_guard_catches_a_refit_on_evaluation_rows(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        # The leak this whole stage is guarded against: statistics fitted on
        # the rows about to be scored rather than on training rows.
        val_users = corpus.users_for(7)["val"]
        belongs = corpus.frame[USER_COLUMN].isin(set(val_users)).to_numpy()
        leaked = Standardiser.fit(
            corpus.frame.loc[belongs & corpus.is_target, list(FEATURE_NAMES)]
            .to_numpy(dtype=np.float32)
        )
        result = standardiser_from_checkpoint(
            corpus, 7, [_checkpoint(leaked, 0.5, tmp_path)]
        )
        assert not result.passed
        assert "differs" in result.detail

    def test_standardiser_guard_catches_two_models_scaled_differently(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        other = Standardiser(
            mean=standardiser.mean + 1.0, std=standardiser.std.copy()
        )
        result = standardiser_from_checkpoint(
            corpus, 7,
            [_checkpoint(standardiser, 0.5, tmp_path), _checkpoint(other, 0.5, tmp_path)],
        )
        assert not result.passed

    def test_identical_rows_catches_a_different_row_set(
        self, corpus: Corpus, standardiser: Standardiser
    ) -> None:
        full = corpus.build("val", 7, standardiser)
        subset = corpus.build(
            "val", 7, standardiser, users=corpus.users_for(7)["val"][:2]
        )
        assert identical_rows(7, [full, full]).passed
        assert not identical_rows(7, [full, subset]).passed

    def test_threshold_guard_reads_the_applied_value(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        result = threshold_from_checkpoint([scored])
        assert result.passed
        assert "0.500000" in result.detail

    def test_threshold_guard_catches_a_swapped_value(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        tampered = replace(scored, threshold=0.7)
        assert not threshold_from_checkpoint([tampered]).passed

    def test_finite_and_baseline_guards(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        assert finite_predictions([scored]).passed
        assert beats_random([scored]).passed

    def test_leakage_ceiling_fails_on_a_perfect_ranking(
        self, corpus: Corpus, standardiser: Standardiser, tmp_path: Path
    ) -> None:
        # A model that ranks the labels exactly. At a 0.4% positive rate with
        # labels this coarse, that is leakage rather than a triumph — and the
        # guard has to say so on the number the project would most like to be
        # true.
        from dataclasses import replace

        from modeling.metrics import evaluate

        scored = _scored(corpus, standardiser, tmp_path, threshold=0.5)
        perfect = scored.split.labels.astype(np.float64) * 0.9 + 0.05
        cheating = replace(
            scored,
            probs=perfect,
            evaluation=evaluate(
                scored.split.labels, perfect, scored.split.scenarios,
                choose_threshold=False,
                threshold=describe_at_threshold(
                    scored.split.labels, perfect, 0.5
                ),
            ),
        )
        result = not_suspiciously_good([cheating])
        assert not result.passed
        assert "leakage" in result.name.lower()
        assert "Investigate before reporting" in result.detail

    def test_leakage_ceiling_passes_on_an_ordinary_score(self) -> None:
        from evaluation import scoring
        from modeling.metrics import evaluate

        labels = np.zeros(1000, dtype=np.int64)
        labels[:4] = 1
        rng = np.random.default_rng(1)
        probs = rng.random(1000)
        probs[:4] += 0.3  # better than chance, nowhere near perfect
        probs = np.clip(probs, 0.0, 1.0)
        ordinary = scoring.Scored(
            checkpoint=_checkpoint(
                Standardiser(mean=np.zeros(N_FEATURES), std=np.ones(N_FEATURES)),
                0.5, Path("/tmp"),
            ),
            split=None,  # type: ignore[arg-type]
            probs=probs,
            evaluation=evaluate(labels, probs, choose_threshold=False),
            threshold=0.5,
        )
        assert not_suspiciously_good([ordinary]).passed


# --------------------------------------------------------------------------
# phase ordering and the audit trail
# --------------------------------------------------------------------------


class TestPhaseOrdering:
    def test_test_split_cannot_be_opened_before_phase_1(self, tmp_path: Path) -> None:
        with pytest.raises(PhaseOrderError, match="Phase 1 has not been run"):
            require_phase1(directory=tmp_path)

    def test_phase_1_artifact_satisfies_the_precondition(self, tmp_path: Path) -> None:
        (tmp_path / "phase1_validation.json").write_text("{}")
        assert require_phase1(directory=tmp_path).is_file()

    def test_second_opening_is_refused_without_a_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evaluation import testpass

        monkeypatch.setattr(testpass, "OPENED_MARKER", tmp_path / "opened.json")
        record_opening(reason="first pass", seeds=[42], sample=False, force=False)
        with pytest.raises(EvaluationError, match="already opened"):
            record_opening(reason="again", seeds=[42], sample=False, force=False)

    def test_forced_reopening_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evaluation import testpass

        marker = tmp_path / "opened.json"
        monkeypatch.setattr(testpass, "OPENED_MARKER", marker)
        record_opening(reason="first pass", seeds=[42], sample=False, force=False)
        record_opening(reason="evaluation bug", seeds=[42], sample=False, force=True)
        history = json.loads(marker.read_text())["openings"]
        assert len(history) == 2
        assert history[1]["reason"] == "evaluation bug"
        assert history[1]["forced"] is True

    def test_a_sample_run_does_not_burn_the_real_test_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evaluation import testpass

        monkeypatch.setattr(testpass, "OPENED_MARKER", tmp_path / "opened.json")
        record_opening(reason="smoke", seeds=[42], sample=True, force=False)
        record_opening(reason="first pass", seeds=[42], sample=False, force=False)


# --------------------------------------------------------------------------
# the report builder
# --------------------------------------------------------------------------


class TestReport:
    def test_optional_sections_say_not_run_rather_than_disappearing(self) -> None:
        from evaluation.report import Sources, build

        sources = Sources(
            phase1=_minimal_phase1(), phase2=_minimal_phase2(),
            val_a=None, val_b=None, control_b7=None, ablation_a=None,
            sample=False,
        )
        text = build(sources)
        assert "not run" in text
        assert "## 4. The capacity control" in text
        assert "## 5. The `pos_weight` ablation" in text

    def test_random_baseline_appears_beside_the_headline_auprc(self) -> None:
        from evaluation.report import Sources, build

        sources = Sources(
            phase1=_minimal_phase1(), phase2=_minimal_phase2(),
            val_a=None, val_b=None, control_b7=None, ablation_a=None,
            sample=False,
        )
        text = build(sources)
        assert "0.00400" in text or "0.004" in text

    def test_sample_reports_are_stamped_as_meaningless(self) -> None:
        from evaluation.report import Sources, build

        sources = Sources(
            phase1=_minimal_phase1(), phase2=_minimal_phase2(),
            val_a=None, val_b=None, control_b7=None, ablation_a=None,
            sample=True,
        )
        assert "meaningless" in build(sources)


def _agg(mean: float, std: float = 0.0) -> dict[str, float]:
    return {"mean": mean, "std": std, "n": 3}


def _minimal_phase2() -> dict[str, object]:
    def head(model: str, auprc: float, params: int, days: int) -> dict[str, object]:
        return {
            "model": model, "context_days": days, "n_parameters": params,
            "seeds": [42, 123, 456], "auprc": _agg(auprc, 0.02),
            "random_baseline": _agg(0.004), "lift_over_random": _agg(auprc / 0.004),
            "auroc": _agg(0.94), "precision_at_10": _agg(0.8),
            "precision_at_50": _agg(0.57), "rows_flagged": _agg(500),
            "per_seed_auprc": {"42": auprc, "123": auprc, "456": auprc},
        }

    scenarios = {
        "1": {"hits_summed": 49, "attack_days_summed": 121, "pooled_recall": 0.405},
        "2": {"hits_summed": 225, "attack_days_summed": 990, "pooled_recall": 0.227},
        "3": {"hits_summed": 4, "attack_days_summed": 12, "pooled_recall": 0.333},
    }
    return {
        "phase": 2, "split": "test", "sample": False, "device": "cpu",
        "seeds": [42, 123, 456],
        "headline": {
            MODEL_A_NAME: head(MODEL_A_NAME, 0.15, 641, 1),
            MODEL_B_NAME: head(MODEL_B_NAME, 0.67, 4897, 28),
        },
        "per_scenario_pooled": {MODEL_A_NAME: scenarios, MODEL_B_NAME: scenarios},
        "detection_latency": {
            "summaries": [], "position_buckets": [], "per_insider": [],
        },
        "per_seed": [{"n_rows": 93000}],
        "guards": [{"guard": "example", "passed": True, "detail": "ok",
                    "fatal_if_failed": True}],
        "all_guards_passed": True,
    }


def _minimal_phase1() -> dict[str, object]:
    return {
        "phase": 1, "split": "validation", "seeds": [42, 123, 456],
        "detection_latency": {
            "summaries": [], "position_buckets": [], "verdicts": [],
            "per_insider": [],
        },
        "false_positive_inspection": {"readings": [], "rows": []},
        "guards": [{"guard": "example", "passed": True, "detail": "ok",
                    "fatal_if_failed": True}],
        "all_guards_passed": True,
    }


# --------------------------------------------------------------------------
# the capacity control's premise
# --------------------------------------------------------------------------


class TestCapacityControl:
    def test_shortening_the_window_does_not_change_the_parameter_count(self) -> None:
        # The entire control rests on this. If it were false, B@7 would vary
        # two things at once and separate nothing.
        def count(days: int) -> int:
            return sum(p.numel() for p in ModelB(days=days).parameters())

        assert count(7) == count(14) == count(WINDOW_DAYS)

    def test_seven_is_the_floor(self) -> None:
        with pytest.raises(ValueError, match="receptive field"):
            ModelB(days=6)
