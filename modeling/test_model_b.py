

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from config import (
    FEATURE_NAMES,
    LABEL_COLUMN,
    N_FEATURES,
    USER_COLUMN,
    WINDOW_DAYS,
)
from modeling.data import DataError, day_index, prepare
from modeling.models import (
    MODEL_B_CHANNELS,
    MODEL_B_DILATION,
    MODEL_B_KERNEL,
    ModelA,
    ModelB,
    count_parameters,
)
from modeling.test_modeling import make_frame, make_splits
from modeling.train import TrainingConfig, seed_everything, train_model

# Codes small enough to survive Z_CLIP = 10 untouched, so a window's contents
# can be read back exactly after standardisation is inverted.
USER_CODE_STEP = 1.0
DAY_CODE_STEP = 0.05


# --------------------------------------------------------------------------
# 1. The time axis is the axis being convolved  (the important one)
# --------------------------------------------------------------------------


SHIFT_TOLERANCE = 1e-5
N_INITIALISATIONS = 8


class _FeatureAxisModel(torch.nn.Module):
    """Model B with the two axes swapped — the bug this section exists for.

    Written out rather than described, so the test below can be shown to
    *fail* on it. A test for a silent failure is only worth its lines if the
    failure it names would actually trip it, and the obvious version of this
    test does not: see
    :func:`test_the_shift_test_would_catch_a_feature_axis_convolution`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(WINDOW_DAYS, MODEL_B_CHANNELS, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(
            MODEL_B_CHANNELS, MODEL_B_CHANNELS, 3, padding=2, dilation=2
        )
        self.head = torch.nn.Linear(MODEL_B_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # days become channels, features become length
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        return self.head(h.mean(dim=2)).squeeze(-1)


def _bump(start: int, width: int = 4) -> torch.Tensor:
    """All eighteen features raised for ``width`` days starting at ``start``.

    Both bumps used below sit clear of the window edges by more than the
    7-day receptive field, so no part of either one is truncated by padding.
    """
    x = torch.zeros(1, N_FEATURES, WINDOW_DAYS)
    x[0, :, start : start + width] = 1.0
    return x


def _shift_response(model: torch.nn.Module) -> float:
    """How much the model's logit moves when a bump slides 8 days later."""
    early, late = _bump(8), _bump(16)
    with torch.no_grad():
        return abs(float(model(early)) - float(model(late)))


def test_model_convolves_across_days_not_across_features() -> None:
    """The same pattern eight days later must score the same.

    This is the property that distinguishes the two axes. A convolution over
    *time*, followed by 'same' padding and a global average pool, is
    translation-invariant: sliding a pattern along the window slides the
    feature maps with it and the mean is unchanged. A convolution over
    *features* treats each day as its own independent channel with its own
    weights, so moving a pattern from days 8-11 to days 16-19 hands it to an
    entirely different set of filters and the answer changes.

    The naive version of this test — rising ramp versus falling ramp — does
    **not** discriminate, because reversing the ramp also permutes the
    channels a feature-axis convolution reads, so that model responds too. It
    would have passed while the model produced nonsense. Hence this one.
    """
    responses = []
    for seed in range(N_INITIALISATIONS):
        seed_everything(seed)
        responses.append(_shift_response(ModelB().eval()))
    assert max(responses) < SHIFT_TOLERANCE, (
        f"sliding a pattern along the window changed the logit by up to "
        f"{max(responses):.2e} — the convolution is not running over the "
        f"time axis, or the padding is not 'same'"
    )


def test_the_shift_test_would_catch_a_feature_axis_convolution() -> None:
    """The mutation check: the test above must fail on the bug it names.

    Measured margin is about six orders of magnitude — 1e-8 for the real
    model against 1e-2 for the transposed one — so this is not a threshold
    balanced on a knife edge.
    """
    responses = []
    for seed in range(N_INITIALISATIONS):
        seed_everything(seed)
        responses.append(_shift_response(_FeatureAxisModel().eval()))
    assert min(responses) > SHIFT_TOLERANCE * 100, (
        f"the feature-axis model passed the time-axis test (responses up to "
        f"{max(responses):.2e}) — the test does not discriminate and proves "
        f"nothing"
    )


def test_the_model_can_tell_a_rising_trend_from_a_falling_one() -> None:
    """Not an axis test — a capability one, and the mechanism under test.

    Scenario 2 is a 45-59 day drift whose early days are loud and whose late
    days have been partly absorbed into the person's own baseline (Stage 3
    measured 0.40x). A model that could not distinguish "climbing" from
    "subsiding" would have no way to profit from the 28-day window.
    """
    rising = torch.zeros(1, N_FEATURES, WINDOW_DAYS)
    rising[0, :, :] = torch.linspace(-3.0, 3.0, WINDOW_DAYS)
    falling = rising.flip(2)
    differences = []
    for seed in range(N_INITIALISATIONS):
        seed_everything(seed)
        model = ModelB().eval()
        with torch.no_grad():
            differences.append(abs(float(model(rising)) - float(model(falling))))
    assert sum(d > 1e-4 for d in differences) >= N_INITIALISATIONS - 2, (
        f"the model is nearly blind to the direction of a trend: {differences}"
    )


def test_a_transposed_batch_is_refused_rather_than_silently_convolved() -> None:
    """``Conv1d`` would accept ``(B, 28, 18)`` and produce meaningless output.

    18 != 28, so the shape contract catches it outright. This is why the
    forward pass validates instead of trusting the caller.
    """
    model = ModelB()
    good = torch.randn(4, N_FEATURES, WINDOW_DAYS)
    with pytest.raises(ValueError, match="wrong axis"):
        model(good.transpose(1, 2))


def test_channels_are_features_and_length_is_days() -> None:
    """The wiring, asserted directly: 18 in-channels, not 28."""
    model = ModelB()
    assert model.conv1.in_channels == N_FEATURES
    assert model.conv1.out_channels == MODEL_B_CHANNELS
    assert model.conv2.dilation == (MODEL_B_DILATION,)
    # 'same' padding on both layers, so the pool averages the whole window.
    with torch.no_grad():
        h = model.activation(model.conv1(torch.zeros(2, N_FEATURES, WINDOW_DAYS)))
        assert h.shape == (2, MODEL_B_CHANNELS, WINDOW_DAYS)
        assert model.conv2(h).shape == (2, MODEL_B_CHANNELS, WINDOW_DAYS)


def test_receptive_field_reaches_seven_days() -> None:
    """Two k=3 layers see 5 days; dilating the second to 2 makes it 7, at the
    same parameter count. That reach is the point of the dilation."""
    assert ModelB().receptive_field == 7
    assert ModelB(dilation=1).receptive_field == 5
    assert count_parameters(ModelB()) == count_parameters(ModelB(dilation=1))


def test_pooling_is_over_time_so_a_sustained_shift_moves_the_output() -> None:
    """Scenario 2 is a sustained elevation, not a single spike.

    Raising every day of the window must move the logit. This would also hold
    for a feature-axis convolution, so it is not the time-axis test — it is
    the check that the *pool* is over days and the sustained level actually
    reaches the head.
    """
    seed_everything(0)
    model = ModelB().eval()
    base = torch.zeros(1, N_FEATURES, WINDOW_DAYS)
    lifted = base + 1.0
    with torch.no_grad():
        assert abs(float(model(base)) - float(model(lifted))) > 1e-4


# --------------------------------------------------------------------------
# 2 & 3. Window contents, and no cross-user contamination
# --------------------------------------------------------------------------


def _coded_frame(n_users: int = 10, days: int = 120) -> pd.DataFrame:
    """A frame whose first two features encode (user, day index) exactly.

    Both codes stay well inside ``Z_CLIP`` so clipping leaves them alone, and
    standardisation is affine, so the raw codes can be recovered exactly and a
    gathered window can be checked row by row against what it should hold.
    """
    frame = make_frame(n_users=n_users, days=days)
    users = sorted(frame[USER_COLUMN].unique())
    codes = {user: i * USER_CODE_STEP for i, user in enumerate(users)}
    frame[FEATURE_NAMES[0]] = (
        frame[USER_COLUMN].map(codes).to_numpy(dtype=np.float32)
    )
    frame[FEATURE_NAMES[1]] = (
        day_index(frame).astype(np.float32) * DAY_CODE_STEP
    )
    return frame


def _decode(windows: torch.Tensor, feature: int, mean, std) -> np.ndarray:
    """Undo the standardiser on one feature channel of a gathered window."""
    return (
        windows[:, feature, :].numpy().astype(np.float64) * std[feature]
        + mean[feature]
    )


def test_a_window_holds_exactly_the_28_days_ending_at_its_target() -> None:
    """Rows t-27 ... t of the same user, in date order, with no gap.

    Off by one in either direction is invisible: the tensor still has the
    right shape, training still converges, and the model is being handed a
    window that ends the day *before* the day it is asked to judge — which is
    a different and much harder problem than the one being reported.
    """
    frame = _coded_frame()
    data = prepare(1, frame=frame, splits=make_splits(frame))
    split = data.train
    mean, std = data.standardiser.mean, data.standardiser.std

    index = torch.arange(split.n_targets)
    windows = split.window_view().gather(index)
    assert windows.shape == (split.n_targets, N_FEATURES, WINDOW_DAYS)

    days = np.rint(_decode(windows, 1, mean, std) / DAY_CODE_STEP).astype(int)
    # Every window is 28 consecutive day-indices ...
    np.testing.assert_array_equal(np.diff(days, axis=1), 1)
    # ... and the LAST of them is the target's own day, not the one before it.
    target_days = day_index(frame)[
        frame[USER_COLUMN].isin(split.user_ids).to_numpy()
    ][split.target_positions.numpy()]
    np.testing.assert_array_equal(days[:, -1], target_days)
    np.testing.assert_array_equal(days[:, 0], target_days - (WINDOW_DAYS - 1))


def test_a_window_never_reaches_into_the_previous_users_rows() -> None:
    """Users are contiguous in the array, so a window that ran off the front
    of a user would silently splice two people's histories together — and the
    26 borrowed rows would still standardise, still train, still score."""
    frame = _coded_frame()
    data = prepare(1, frame=frame, splits=make_splits(frame))
    split = data.train
    mean, std = data.standardiser.mean, data.standardiser.std

    windows = split.window_view().gather(torch.arange(split.n_targets))
    codes = np.rint(
        _decode(windows, 0, mean, std) / USER_CODE_STEP
    ).astype(int)
    # Every one of the 28 rows in a window carries the same user code.
    assert (codes == codes[:, :1]).all(), (
        "at least one window spans two users"
    )
    # And that user is the target row's own user.
    users = sorted(frame[USER_COLUMN].unique())
    expected = np.array([users.index(u) for u in split.users])
    np.testing.assert_array_equal(codes[:, -1], expected)


def test_the_mask_and_the_window_length_cannot_disagree_silently() -> None:
    """If a target ever sat inside its user's first 27 rows, the window would
    reach before the array's start and wrap or truncate. ``window_view``
    refuses instead."""
    frame = make_frame(n_users=8)
    data = prepare(1, frame=frame, splits=make_splits(frame))
    split = data.train
    assert int(split.target_positions.min()) >= WINDOW_DAYS - 1

    # A split whose targets sit inside the warm-up: the window would reach
    # before row 0 of the array and index-select would wrap, not raise.
    broken = replace(
        split, target_positions=torch.tensor([0, 5], dtype=torch.int64)
    )
    with pytest.raises(DataError, match="mask and the window length"):
        broken.window_view()


# --------------------------------------------------------------------------
# 4. Model B is scored on exactly Model A's rows
# --------------------------------------------------------------------------


def test_both_models_see_the_same_target_rows_and_the_same_labels() -> None:
    """The comparison is void the moment the two models are measured on
    different rows — and a difference of a few thousand warm-up negatives
    moves AUPRC by several percent, entirely mechanically."""
    frame = make_frame(n_users=10)
    data = prepare(42, frame=frame, splits=make_splits(frame))

    for split in (data.train, data.val):
        point, window = split.point_view(), split.window_view()
        assert len(point) == len(window) == split.n_targets
        assert len(window) == split.labels.numel()
        # Same rows, same order: the last day of each window is Model A's row.
        index = torch.arange(split.n_targets)
        torch.testing.assert_close(
            window.gather(index)[:, :, -1], point.gather(index)
        )


def test_pos_weight_is_unchanged_by_which_model_reads_the_data() -> None:
    """``pos_weight`` comes from the prepared training labels, not from the
    view — so it is identical for both models by construction. On r4.2 it must
    print ~229.7 / 225.3 / 232.1; ~244 is the whole-table figure and means the
    row mask was not applied first."""
    frame = make_frame(n_users=12)
    data = prepare(42, frame=frame, splits=make_splits(frame))
    expected = data.train.n_negatives / data.train.n_positives
    assert data.pos_weight() == pytest.approx(expected)
    # The mask is what makes it differ from the whole table's ratio.
    whole = (len(frame) - frame[LABEL_COLUMN].sum()) / frame[LABEL_COLUMN].sum()
    assert data.pos_weight() != pytest.approx(whole)


# --------------------------------------------------------------------------
# 5. No sigmoid
# --------------------------------------------------------------------------


def test_model_b_returns_raw_logits() -> None:
    """A sigmoid in ``forward`` plus the one inside ``BCEWithLogitsLoss``
    squashes the output twice: gradients shrink toward nothing, the model
    barely trains, and the loss curve merely looks disappointing."""
    seed_everything(0)
    model = ModelB()
    x = torch.randn(256, N_FEATURES, WINDOW_DAYS) * 50.0
    out = model(x)
    assert out.min() < 0.0 or out.max() > 1.0, (
        "every output landed inside [0, 1] — there is a sigmoid in forward()"
    )
    assert not any(
        isinstance(module, torch.nn.Sigmoid) for module in model.modules()
    )


# --------------------------------------------------------------------------
# 6. Determinism
# --------------------------------------------------------------------------


def _train_model_b_once(frame: pd.DataFrame, splits: dict, seed: int) -> dict:
    """Mirrors ``runner.run_experiment``, seeding order included.

    ``seed_everything`` before the model is constructed, not after: seed
    afterwards and the weight initialisation is drawn from whatever state the
    process was already in, so "seed 42" reproduces only by luck.
    """
    data = prepare(seed, frame=frame, splits=splits)
    config = TrainingConfig(seed=seed, device="cpu", max_epochs=3)
    seed_everything(seed)
    result = train_model(
        ModelB(), data.train.window_view(), data.train.labels,
        data.val.window_view(), data.val.labels, config,
        val_scenarios=data.val.scenarios.numpy(),
        pos_weight=data.pos_weight(),
    )
    return result.best_state


def test_same_seed_twice_gives_identical_model_b_weights() -> None:
    frame = make_frame(n_users=8)
    splits = make_splits(frame)
    first = _train_model_b_once(frame, splits, seed=42)
    second = _train_model_b_once(frame, splits, seed=42)
    for key, value in first.items():
        torch.testing.assert_close(value, second[key], rtol=0, atol=0)


def test_different_seeds_give_different_model_b_weights() -> None:
    frame = make_frame(n_users=8)
    splits = make_splits(frame)
    a = _train_model_b_once(frame, splits, seed=42)
    b = _train_model_b_once(frame, splits, seed=123)
    assert not torch.equal(a["conv1.weight"], b["conv1.weight"])


# --------------------------------------------------------------------------
# 7. Shape and size contract
# --------------------------------------------------------------------------


def test_shape_contract() -> None:
    """(B, 18, 28) in, (B,) out.

    ``(B,)`` and not ``(B, 1)`` deliberately: ``BCEWithLogitsLoss`` given a
    ``(B, 1)`` input and a ``(B,)`` target broadcasts to ``(B, B)`` and trains
    happily on a loss that means nothing. ``train.predict`` writes into a
    ``(n,)`` buffer for the same reason. Model A returns ``(B,)``; so does
    Model B.
    """
    model = ModelB()
    for batch in (1, 7, 512):
        out = model(torch.randn(batch, N_FEATURES, WINDOW_DAYS))
        assert out.shape == (batch,)
    assert ModelA()(torch.randn(7, N_FEATURES)).shape == (7,)


def test_model_b_is_small_on_purpose() -> None:
    """4,897 parameters against ~1,140 positive training rows.

    Already a thinner ratio than Model A's ~640. A deeper Model B memorises
    those rows and wins the comparison for the wrong reason, which answers no
    question anybody asked. This is a tripwire against a future 'improvement'.
    """
    conv1 = N_FEATURES * MODEL_B_CHANNELS * MODEL_B_KERNEL + MODEL_B_CHANNELS
    conv2 = (
        MODEL_B_CHANNELS * MODEL_B_CHANNELS * MODEL_B_KERNEL + MODEL_B_CHANNELS
    )
    head = MODEL_B_CHANNELS + 1
    assert count_parameters(ModelB()) == conv1 + conv2 + head == 4_897


def test_a_receptive_field_longer_than_the_window_is_refused() -> None:
    """A second layer that sees only padding is a silent no-op, not an error."""
    with pytest.raises(ValueError, match="receptive field"):
        ModelB(days=5, dilation=8)
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        ModelB(kernel_size=4)


# --------------------------------------------------------------------------
# End to end, small
# --------------------------------------------------------------------------


def test_model_b_trains_through_the_unmodified_loop_and_beats_random() -> None:
    """``train_model`` takes the CNN and the window view with no edits.

    That is Stage 6's prime directive expressed as a test: if this needed a
    change in ``train.py``, Model A was trained under different conditions and
    the comparison is weakened.
    """
    frame = make_frame(n_users=12)
    data = prepare(42, frame=frame, splits=make_splits(frame))
    config = TrainingConfig(seed=42, device="cpu", max_epochs=8)
    seed_everything(42)
    result = train_model(
        ModelB(), data.train.window_view(), data.train.labels,
        data.val.window_view(), data.val.labels, config,
        val_scenarios=data.val.scenarios.numpy(),
        pos_weight=data.pos_weight(),
    )
    assert result.evaluation.auprc > result.evaluation.random_baseline
    assert result.evaluation.threshold is not None
    assert not result.collapse_fired
    assert result.n_parameters == 4_897

