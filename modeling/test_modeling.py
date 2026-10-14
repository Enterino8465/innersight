

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
import torch

from config import (
    FEATURE_NAMES,
    LABEL_COLUMN,
    MIN_DISTINCT_PREDICTIONS,
    N_FEATURES,
    SCENARIO_COLUMN,
    USER_COLUMN,
    WINDOW_DAYS,
    Z_CLIP,
)
from modeling.data import (
    DataError,
    Standardiser,
    clip_z_scores,
    day_index,
    load_splits,
    prepare,
    target_mask,
)
from modeling.metrics import PredictionStats, precision_at_k, sweep_threshold
from modeling.models import ModelA, count_parameters
from modeling.train import (
    TrainingConfig,
    compute_pos_weight,
    seed_everything,
    train_model,
)

RNG_SEED = 7
DAYS_PER_USER = 120


def make_frame(
    n_users: int = 12,
    days: int = DAYS_PER_USER,
    positives_from_day: int = 60,
    scale: float = 1.0,
    offset: float = 0.0,
    prefix: str = "U",
) -> pd.DataFrame:
    """A calendar-dense synthetic table with the same shape as dataset.parquet.

    ``scale``/``offset`` exist so a test can build a validation split whose
    values are wildly different from training's and then assert the
    standardiser did not notice.
    """
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for u in range(n_users):
        user = f"{prefix}{u:03d}"
        scenario = (u % 4)  # 0 benign, 1/2/3 insider types
        dates = pd.date_range("2010-01-04", periods=days, freq="D")
        values = rng.normal(0.0, 1.0, size=(days, N_FEATURES)) * scale + offset
        labels = np.zeros(days, dtype=np.int8)
        if scenario != 0:
            labels[positives_from_day : positives_from_day + 6] = 1
            # An insider's attack days look different, so the models have
            # something real to find and AUPRC is not pure noise.
            values[labels == 1] += 4.0 * scale
        frame = pd.DataFrame(values.astype(np.float32), columns=list(FEATURE_NAMES))
        frame.insert(0, USER_COLUMN, user)
        frame.insert(1, "day", dates)
        frame[SCENARIO_COLUMN] = np.int8(scenario)
        frame[LABEL_COLUMN] = labels
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def make_splits(frame: pd.DataFrame, n_val: int = 4) -> dict[str, list[str]]:
    users = sorted(frame[USER_COLUMN].unique())
    return {
        "train": users[:-n_val],
        "val": users[-n_val:],
        # Present, and deliberately never read by Stage 5.
        "test": [],
    }


# --------------------------------------------------------------------------
# 1. The row mask
# --------------------------------------------------------------------------


def test_first_27_days_are_never_prediction_targets() -> None:
    """Model B has no 28-day window ending before a user's 28th day.

    If Model A scored those rows and Model B did not, the two models would be
    measured on different data — and the rows in question are warm-up zeros,
    the easiest negatives in the table, so Model B's positive rate and
    therefore its AUPRC would rise for free.
    """
    frame = make_frame(n_users=3)
    mask = target_mask(frame)
    index = day_index(frame)

    assert not mask[index < WINDOW_DAYS - 1].any(), (
        "a row inside the 27-day warm-up was selected as a prediction target"
    )
    assert mask[index == WINDOW_DAYS - 1].all(), (
        "the 28th day of a user must be a prediction target"
    )
    assert mask.sum() == len(frame) - 3 * (WINDOW_DAYS - 1)


def test_row_mask_loses_no_positives() -> None:
    """On r4.2 the earliest attack window opens ~160 days into every record.

    The mask therefore costs ~27,000 warm-up rows and zero positives. This
    asserts the property on synthetic data; ``prepare`` measures and logs it
    on the real table every run rather than trusting it.
    """
    frame = make_frame(positives_from_day=60)
    mask = target_mask(frame)
    lost = int(frame.loc[~mask, LABEL_COLUMN].sum())
    assert lost == 0, f"{lost} positive rows were dropped by the row mask"


def test_mask_applies_within_user_not_across_the_table() -> None:
    """Each user gets their own 27-day warm-up, not one shared warm-up."""
    frame = make_frame(n_users=5)
    mask = target_mask(frame)
    per_user = pd.Series(mask).groupby(frame[USER_COLUMN].to_numpy()).sum()
    assert (per_user == DAYS_PER_USER - (WINDOW_DAYS - 1)).all()


def test_non_dense_calendar_is_refused() -> None:
    """A gap would make a 28-row window span more than 28 days, silently."""
    frame = make_frame(n_users=2)
    frame = frame.drop(index=50).reset_index(drop=True)
    with pytest.raises(DataError, match="calendar-day density"):
        target_mask(frame)


def test_history_rows_stay_available_to_model_b() -> None:
    """The mask selects what is *predicted*, not what is *readable*.

    Model B still needs the 27 rows before each target. They are in the
    array; they are simply never targets themselves.
    """
    frame = make_frame(n_users=6)
    data = prepare(1, frame=frame, splits=make_splits(frame))
    train = data.train
    assert train.features.shape[0] > train.n_targets
    windows = train.window_view().gather(torch.arange(4))
    assert windows.shape == (4, N_FEATURES, WINDOW_DAYS)
    # The last day of each window is the target row itself.
    torch.testing.assert_close(
        windows[:, :, -1], train.point_features()[:4]
    )


# --------------------------------------------------------------------------
# 2. The standardiser is fitted on train only
# --------------------------------------------------------------------------


def test_standardiser_ignores_validation_rows() -> None:
    """Fitting on everything leaks validation statistics into training.

    Built so the leak would be unmissable if it happened: the validation
    users' features are 50x larger and shifted by 1,000. If those rows touched
    the fit, the mean and std would move enormously. They must not move at all.
    """
    train_frame = make_frame(n_users=8, prefix="T")
    val_frame = make_frame(n_users=4, prefix="V", scale=50.0, offset=1000.0)
    combined = pd.concat([train_frame, val_frame], ignore_index=True)
    splits = {
        "train": sorted(train_frame[USER_COLUMN].unique()),
        "val": sorted(val_frame[USER_COLUMN].unique()),
        "test": [],
    }

    fitted = prepare(1, frame=combined, splits=splits).standardiser

    # What the statistics must be: the training users' target rows, and only
    # those. Computed here from the raw frame, independently of the pipeline.
    warmup = day_index(train_frame) < WINDOW_DAYS - 1
    expected = train_frame.loc[~warmup, list(FEATURE_NAMES)].to_numpy(
        dtype=np.float64
    )
    np.testing.assert_allclose(
        fitted.mean, expected.mean(axis=0), atol=1e-5,
        err_msg="validation rows moved the standardiser's mean",
    )
    np.testing.assert_allclose(
        fitted.std, expected.std(axis=0), atol=1e-5,
        err_msg="validation rows moved the standardiser's std",
    )
    # And a sanity floor: the validation users sit around 1,000. If they had
    # been included, the mean could not possibly still be near 0.
    assert np.abs(fitted.mean).max() < 1.0


def test_standardiser_is_fitted_on_target_rows_only() -> None:
    """The warm-up rows are not scored, so they must not set the scale either.

    Fitting on them would drag every feature's mean toward the warm-up value
    and shrink its std — a transform describing rows no model ever sees.
    """
    frame = make_frame(n_users=6)
    features = list(FEATURE_NAMES)
    # Make the warm-up rows extreme so including them would be obvious.
    warmup = day_index(frame) < WINDOW_DAYS - 1
    frame.loc[warmup, features] = 9.0

    splits = make_splits(frame, n_val=2)
    data = prepare(1, frame=frame, splits=splits)

    expected = frame.loc[
        frame[USER_COLUMN].isin(splits["train"]) & ~warmup, features
    ].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(
        data.standardiser.mean, expected.mean(axis=0), rtol=1e-5,
    )


def test_standardised_train_targets_have_zero_mean_unit_std() -> None:
    frame = make_frame(n_users=8)
    data = prepare(1, frame=frame, splits=make_splits(frame))
    points = data.train.point_features().numpy()
    np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(points.std(axis=0), 1.0, atol=1e-3)


def test_constant_feature_is_left_at_zero_not_divided_by_a_floor() -> None:
    """r4.2's ``large_attachment_count`` is identically zero.

    Dividing it by a 1e-6 floor would turn float noise into z-scores of 10^6.
    A dead feature should contribute nothing, forever.
    """
    features = np.zeros((100, N_FEATURES), dtype=np.float32)
    features[:, 0] = np.linspace(-3, 3, 100)
    standardiser = Standardiser.fit(features)
    assert standardiser.std[5] == 1.0
    assert np.isfinite(standardiser.apply(features)).all()
    assert (standardiser.apply(features)[:, 5] == 0.0).all()


# --------------------------------------------------------------------------
# 3. Clipping
# --------------------------------------------------------------------------


def test_clipping_bounds_extremes_and_leaves_ordinary_values_alone() -> None:
    """716 -> 10, -34.94 -> -10, 0.4 -> 0.4. The real min and max of r4.2."""
    frame = make_frame(n_users=1)
    features = list(FEATURE_NAMES)
    frame.loc[30, features[0]] = 716.26
    frame.loc[31, features[0]] = -34.94
    frame.loc[32, features[0]] = 0.4

    clipped, touched = clip_z_scores(frame)
    assert clipped.loc[30, features[0]] == pytest.approx(Z_CLIP)
    assert clipped.loc[31, features[0]] == pytest.approx(-Z_CLIP)
    assert clipped.loc[32, features[0]] == pytest.approx(0.4)
    assert touched == 2
    assert np.abs(clipped[features].to_numpy()).max() <= Z_CLIP


def test_clipping_happens_before_standardising() -> None:
    """Order matters and swapping it raises nothing.

    A single z of 716 inflates that feature's std enormously; standardising
    first and clipping after would squash every ordinary day toward zero and
    then clip a range that no longer reaches the bound. The check: with the
    outlier present, the standardised training rows still have unit std —
    which is only true if the outlier was clipped before the fit.
    """
    frame = make_frame(n_users=8)
    frame.loc[100, FEATURE_NAMES[0]] = 716.26

    data = prepare(1, frame=frame, splits=make_splits(frame))
    column = data.train.point_features().numpy()[:, 0]
    assert column.std() == pytest.approx(1.0, abs=1e-2)
    assert column.max() < 12.0, "an unclipped outlier survived into the tensors"


# --------------------------------------------------------------------------
# 4. pos_weight
# --------------------------------------------------------------------------


def test_pos_weight_is_train_only_and_moves_with_the_split() -> None:
    """Computing it from the whole table leaks the test set's class balance.

    Nothing would complain: the loss would simply be weighted by a number that
    knows something training is not entitled to know.
    """
    frame = make_frame(n_users=12)
    splits_a = make_splits(frame, n_val=4)
    users = sorted(frame[USER_COLUMN].unique())
    # A different number of insiders in training, so the ratio must move.
    splits_b = {"train": users[:6], "val": users[6:10], "test": []}

    a = prepare(1, frame=frame, splits=splits_a)
    b = prepare(1, frame=frame, splits=splits_b)

    assert a.pos_weight() == pytest.approx(
        a.train.n_negatives / a.train.n_positives
    )
    assert a.pos_weight() != pytest.approx(b.pos_weight()), (
        "pos_weight did not change when the training split changed — it is "
        "being computed from something other than the training rows"
    )
    # And it is not the whole table's ratio.
    whole = (len(frame) - frame[LABEL_COLUMN].sum()) / frame[LABEL_COLUMN].sum()
    assert a.pos_weight() != pytest.approx(whole)


def test_compute_pos_weight_matches_the_definition() -> None:
    labels = torch.tensor([0.0] * 244 + [1.0])
    assert compute_pos_weight(labels) == pytest.approx(244.0)


def test_training_without_positives_is_refused() -> None:
    frame = make_frame(n_users=4)
    frame[LABEL_COLUMN] = np.int8(0)
    data = prepare(1, frame=frame, splits=make_splits(frame, n_val=1))
    with pytest.raises(DataError, match="no positive rows"):
        data.pos_weight()


# --------------------------------------------------------------------------
# 5. No sigmoid in the model
# --------------------------------------------------------------------------


def test_model_a_returns_raw_logits() -> None:
    """A sigmoid in ``forward`` plus one inside BCEWithLogitsLoss squashes the
    output twice; gradients shrink and the model barely trains, with no error.
    """
    torch.manual_seed(0)
    model = ModelA()
    # Large inputs so a linear layer must produce values well outside [0, 1].
    x = torch.randn(256, N_FEATURES) * 50.0
    out = model(x)
    assert out.shape == (256,)
    assert out.min() < 0.0 or out.max() > 1.0, (
        "every output landed inside [0, 1] — there is a sigmoid in forward()"
    )
    assert not isinstance(model.net[-1], torch.nn.Sigmoid)


def test_model_a_is_small_on_purpose() -> None:
    """~640 parameters against ~1,140 positive training rows.

    A bigger network memorises those rows and generalises to nothing; this is
    a tripwire against a future 'improvement'.
    """
    assert count_parameters(ModelA()) == (N_FEATURES * 32 + 32) + (32 + 1)


# --------------------------------------------------------------------------
# 6. Determinism
# --------------------------------------------------------------------------


def _train_once(frame: pd.DataFrame, splits: dict, seed: int) -> dict:
    """Mirrors what ``runner.run_experiment`` does, seeding included.

    The seeding happens *before* the model is constructed, which is the whole
    point: seed afterwards and the weight initialisation is drawn from
    whatever state the process was already in, so "seed 42" reproduces only
    by luck.
    """
    data = prepare(seed, frame=frame, splits=splits)
    config = TrainingConfig(seed=seed, device="cpu", max_epochs=3)
    seed_everything(seed)
    result = train_model(
        ModelA(), data.train.point_view(), data.train.labels,
        data.val.point_view(), data.val.labels, config,
        val_scenarios=data.val.scenarios.numpy(),
        pos_weight=data.pos_weight(),
    )
    return result.best_state


def test_same_seed_twice_gives_identical_weights() -> None:
    """Weight init and batch shuffling are both seeded, so a rerun reproduces.

    Without this, "mean +/- sd across three seeds" measures the seed plus
    whatever the process happened to do, and no result is reproducible.
    """
    frame = make_frame(n_users=8)
    splits = make_splits(frame)
    first = _train_once(frame, splits, seed=42)
    second = _train_once(frame, splits, seed=42)
    for key, value in first.items():
        torch.testing.assert_close(value, second[key], rtol=0, atol=0)


def test_different_seeds_give_different_weights() -> None:
    frame = make_frame(n_users=8)
    splits = make_splits(frame)
    a = _train_once(frame, splits, seed=42)
    b = _train_once(frame, splits, seed=123)
    assert not torch.equal(a["net.0.weight"], b["net.0.weight"])


# --------------------------------------------------------------------------
# 7. User-level integrity
# --------------------------------------------------------------------------


def test_no_user_appears_in_both_train_and_val_tensors() -> None:
    """The split is by user. A person's Monday training a model tested on
    their Tuesday measures memorisation, and it looks *excellent*."""
    frame = make_frame(n_users=10)
    data = prepare(1, frame=frame, splits=make_splits(frame))
    assert not set(data.train.users) & set(data.val.users)
    assert not set(data.train.user_ids) & set(data.val.user_ids)


def test_overlapping_splits_are_refused() -> None:
    frame = make_frame(n_users=6)
    users = sorted(frame[USER_COLUMN].unique())
    bad = {"train": users[:5], "val": users[4:], "test": []}
    with pytest.raises(DataError, match="both train and val"):
        prepare(1, frame=frame, splits=bad)


def test_stage_5_never_builds_the_test_split() -> None:
    """Test data is opened once, in Stage 7. ``prepare`` cannot expose it."""
    frame = make_frame(n_users=9)
    users = sorted(frame[USER_COLUMN].unique())
    splits = {"train": users[:5], "val": users[5:7], "test": users[7:]}
    data = prepare(1, frame=frame, splits=splits)
    held_out = set(users[7:])
    assert not held_out & set(data.train.user_ids)
    assert not held_out & set(data.val.user_ids)
    assert not hasattr(data, "test")


def test_real_splits_file_parses_if_present() -> None:
    """Stage 4's output nests the seeds under a "splits" key; Stage 5 reads
    either shape, so a future flattening would not break it silently."""
    from config import SPLITS_SAMPLE_PATH

    if not SPLITS_SAMPLE_PATH.is_file():
        pytest.skip("sample splits not generated in this checkout")
    splits = load_splits(SPLITS_SAMPLE_PATH, 42)
    assert set(splits) == {"train", "val", "test"}
    payload = json.loads(SPLITS_SAMPLE_PATH.read_text())
    assert splits["train"] == tuple(payload["splits"]["42"]["train"])


# --------------------------------------------------------------------------
# 8. The collapse detector fires
# --------------------------------------------------------------------------


class _CollapsedModel(torch.nn.Module):
    """A model that has learned "nobody is an insider" — 99.6% accurate,
    worthless, and it produces a perfectly healthy-looking loss curve."""

    def __init__(self) -> None:
        super().__init__()
        # One parameter so Adam has something to optimise; multiplied by zero
        # so the output stays constant while the graph stays differentiable.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), -9.0) + 0.0 * self.dummy


def test_collapse_guard_fires_on_a_degenerate_model() -> None:
    stats = PredictionStats.of(np.full(50_000, 1e-4))
    assert stats.distinct == 1
    assert stats.collapsed


def test_collapse_guard_stays_quiet_on_a_healthy_spread() -> None:
    rng = np.random.default_rng(0)
    stats = PredictionStats.of(rng.uniform(0, 1, size=50_000))
    assert stats.distinct > MIN_DISTINCT_PREDICTIONS
    assert not stats.collapsed


def test_collapse_is_reported_through_a_real_training_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End to end: a collapsed model must set the flag and warn, not pass."""
    frame = make_frame(n_users=8)
    data = prepare(1, frame=frame, splits=make_splits(frame))
    config = TrainingConfig(seed=1, device="cpu", max_epochs=1)
    model = _CollapsedModel()

    with caplog.at_level(logging.WARNING):
        result = train_model(
            model, data.train.point_view(), data.train.labels,
            data.val.point_view(), data.val.labels, config,
            pos_weight=data.pos_weight(),
        )
    assert result.collapse_fired
    assert any("COLLAPSE" in record.message for record in caplog.records)


# --------------------------------------------------------------------------
# Metrics behaviour worth pinning
# --------------------------------------------------------------------------


def test_threshold_sweep_does_not_default_to_half() -> None:
    """With a weighted loss the outputs are deliberately not calibrated
    around 0.5, so the threshold has to be found, not assumed."""
    labels = np.zeros(1000, dtype=np.int64)
    labels[:20] = 1
    probs = np.full(1000, 0.001)
    probs[:20] = 0.02  # the positives score higher, but nowhere near 0.5
    choice = sweep_threshold(labels, probs)
    assert choice.threshold < 0.5
    assert choice.f1 == pytest.approx(1.0)


def test_precision_at_k_reads_the_top_of_the_ranking() -> None:
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(labels, probs, 2) == pytest.approx(1.0)
    assert precision_at_k(labels, probs, 4) == pytest.approx(0.5)


def test_end_to_end_run_beats_the_random_baseline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole pipeline, small: prepare -> train -> select -> evaluate."""
    frame = make_frame(n_users=12)
    data = prepare(42, frame=frame, splits=make_splits(frame))
    config = TrainingConfig(seed=42, device="cpu", max_epochs=8)
    with caplog.at_level(logging.INFO):
        result = train_model(
            ModelA(), data.train.point_view(), data.train.labels,
            data.val.point_view(), data.val.labels, config,
            val_scenarios=data.val.scenarios.numpy(),
            pos_weight=data.pos_weight(),
        )
    assert result.evaluation.auprc > result.evaluation.random_baseline
    assert result.evaluation.threshold is not None
    assert not result.collapse_fired
    assert result.best_epoch >= 1
