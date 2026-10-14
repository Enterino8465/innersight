"""One loader, both models, no branch that knows which is which.

Stage 5 wrote Model A's checkpoints and Stage 6 wrote Model B's through the
*same* ``runner.save_checkpoint`` call, so each file already carries the four
things Stage 7 must not re-derive: the weights, the decision threshold chosen
on validation, the standardiser fitted on training rows, and enough shape
information (``input_kind``, ``window_days``) to rebuild the right architecture
and hand it the right view.

That is why this module has no ``if model_class == "ModelA"`` in its public
path beyond a single factory table. A loader that has to be *told* which model
it is loading is a loader that can be told wrong, and the entire result rests
on both models being evaluated by identical code.

**Model B v2 cannot be loaded here.** §3 of the Stage 7 brief is unambiguous:
the tuning attempt lost, it is not part of the result, and no ``*_v2_*``
checkpoint may appear in any comparison. Rather than relying on nobody typing
the path, :func:`load_checkpoint` accepts only filenames matching the shape
the headline experiment produced. Variant runs — the ``nopw`` ablation, the
``w7`` capacity control — are reachable only by asking for them explicitly,
and v2 is not reachable at all.

**The ``w7`` caveat, stated where it will be read.** ``runner.save_checkpoint``
writes ``window_days = config.WINDOW_DAYS`` for any window model rather than
the length the view actually used, so a 7-day Model B checkpoint claims 28 on
disk. Stage 7 is forbidden from editing ``runner.py``, and the control is
validation-only and never reloaded, so the field is simply not trusted:
:func:`load_variant_checkpoint` requires the caller to state the window length
it expects and raises if the weights disagree.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from config import CHECKPOINT_DIR, MODEL_A_NAME, MODEL_B_NAME, WINDOW_DAYS
from modeling.data import SplitData, Standardiser
from modeling.metrics import ThresholdChoice
from modeling.models import ModelA, ModelB
from modeling.train import InputView

logger = logging.getLogger(__name__)


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be trusted to produce a reportable number."""


_FACTORIES: Final[dict[str, Callable[[int], nn.Module]]] = {
    "ModelA": lambda days: ModelA(),
    "ModelB": lambda days: ModelB(days=days),
}

_EXPECTED_KIND: Final[dict[str, str]] = {"ModelA": "point", "ModelB": "window"}

_HEADLINE_NAME: Final[re.Pattern[str]] = re.compile(
    rf"^(?:{MODEL_A_NAME}|{MODEL_B_NAME})_seed\d+(?:\.sample)?\.pt$"
)

_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model_class",
        "state_dict",
        "input_kind",
        "window_days",
        "threshold",
        "standardiser",
        "seed",
    }
)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One trained model, plus the two things Stage 7 must not recompute."""

    path: Path
    model_name: str
    model_class: str
    input_kind: str
    window_days: int
    threshold: float
    standardiser: Standardiser
    seed: int
    device_trained_on: str
    n_parameters: int
    best_val_auprc: float | None
    val_random_baseline: float | None
    validation_threshold_selection: dict[str, Any] | None
    state_dict: dict[str, torch.Tensor]

    @property
    def context_days(self) -> int:
        """How many days of history this model sees. Model A sees one."""
        return self.window_days if self.input_kind == "window" else 1

    def build(self) -> nn.Module:
        """Rebuild the architecture and load the weights strictly.

        ``strict=True`` is the point of the exercise: it is the one check that
        the class named in the file is the class the weights belong to.
        """
        model = _FACTORIES[self.model_class](self.window_days)
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        return model

    def view(self, split: SplitData) -> InputView:
        """The input view this model was trained on, over the given rows."""
        if self.input_kind == "point":
            return split.point_view()
        return split.window_view(self.window_days)

    def threshold_choice(self) -> ThresholdChoice:
        """The stored cutoff, as the object :func:`modeling.metrics.evaluate`
        accepts — so applying it goes through the same code that applied it
        during training, with the sweep switched off."""
        selection = self.validation_threshold_selection or {}
        return ThresholdChoice(
            threshold=self.threshold,
            f1=float(selection.get("f1", float("nan"))),
            precision=float(selection.get("precision", float("nan"))),
            recall=float(selection.get("recall", float("nan"))),
            predicted_positives=int(selection.get("predicted_positives", 0)),
        )

    def describe(self) -> str:
        return (
            f"{self.model_name} seed {self.seed}: {self.model_class}, "
            f"{self.input_kind} input over {self.context_days} day(s), "
            f"{self.n_parameters:,} params, threshold {self.threshold:.6f}, "
            f"trained on {self.device_trained_on}"
        )


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(
            f"checkpoint not found at {path}. Stage 5 and 6 write these; run "
            f"`python -m modeling.run_model_a` / `run_model_b` first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path.name} does not contain a checkpoint dict")
    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        raise CheckpointError(
            f"{path.name} is missing {len(missing)} required key(s): "
            f"{', '.join(sorted(missing))}. It was not written by "
            f"runner.save_checkpoint."
        )
    return payload


def _assemble(path: Path, payload: dict[str, Any], model_name: str) -> Checkpoint:
    model_class = str(payload["model_class"])
    if model_class not in _FACTORIES:
        raise CheckpointError(
            f"{path.name} names architecture {model_class!r}, which Stage 7 "
            f"does not know how to rebuild (expected one of "
            f"{', '.join(sorted(_FACTORIES))})"
        )

    input_kind = str(payload["input_kind"])
    if input_kind != _EXPECTED_KIND[model_class]:
        raise CheckpointError(
            f"{path.name} says {model_class} was trained on {input_kind!r} "
            f"inputs, but {model_class} takes "
            f"{_EXPECTED_KIND[model_class]!r}. The file is inconsistent; do "
            f"not score anything with it."
        )

    threshold = float(payload["threshold"])
    if not 0.0 < threshold < 1.0:
        raise CheckpointError(
            f"{path.name} carries decision threshold {threshold!r}, which is "
            f"not a probability strictly inside (0, 1). A threshold of 0 or 1 "
            f"means the sweep found nothing and the flagged-row counts would "
            f"be meaningless."
        )

    return Checkpoint(
        path=path,
        model_name=model_name,
        model_class=model_class,
        input_kind=input_kind,
        window_days=int(payload["window_days"]),
        threshold=threshold,
        standardiser=Standardiser.from_dict(payload["standardiser"]),
        seed=int(payload["seed"]),
        device_trained_on=str(payload.get("device", "unknown")),
        n_parameters=int(payload.get("n_parameters", 0)),
        best_val_auprc=_optional_float(payload.get("best_val_auprc")),
        val_random_baseline=_optional_float(payload.get("val_random_baseline")),
        validation_threshold_selection=payload.get("threshold_selection"),
        state_dict=payload["state_dict"],
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_checkpoint(path: Path | str) -> Checkpoint:
    """Load one of the six checkpoints the reported experiment produced.

    Refuses anything whose filename carries a variant tag. That is the
    mechanism by which "do not put v2 in any comparison table" survives a
    tired evening: the path simply does not load.
    """
    path = Path(path)
    if not _HEADLINE_NAME.match(path.name):
        raise CheckpointError(
            f"{path.name} is not one of the reported experiment's "
            f"checkpoints. Stage 7 reports Model A and Model B only — the "
            f"v2 tuning attempt lost and is excluded by brief §3, and the "
            f"nopw / w7 runs are controls, not competing models. Use "
            f"load_variant_checkpoint() if you deliberately want one."
        )
    payload = _read(path)
    model_name = MODEL_A_NAME if path.name.startswith(MODEL_A_NAME) else MODEL_B_NAME
    return _assemble(path, payload, model_name)


def load_variant_checkpoint(
    path: Path | str,
    *,
    model_name: str,
    expected_window_days: int | None = None,
) -> Checkpoint:
    """Load a control run's checkpoint, with the window length asserted.

    ``runner.save_checkpoint`` stamps every window checkpoint with
    ``config.WINDOW_DAYS`` rather than the view's actual length, so a B@7 file
    claims 28. Stage 7 may not edit ``runner.py``, so instead the caller states
    what it expects and the weights are checked against it — a Model B whose
    convolutions were trained on 7 days still has to *accept* a 7-day tensor,
    and :meth:`Checkpoint.build` plus a forward pass is what proves it.

    ``*_v2_*`` remains refused here too. It is excluded from the result by the
    brief, not by convenience.
    """
    path = Path(path)
    if "_v2_" in path.name:
        raise CheckpointError(
            f"{path.name} is a Model B v2 checkpoint. Brief §3: v2 lost "
            f"(AUPRC 0.6767 -> 0.6022), is not part of the result, and must "
            f"not be loaded, tabulated or headlined. Its one appearance is a "
            f"discussion paragraph about pooling, which needs no weights."
        )
    payload = _read(path)
    checkpoint = _assemble(path, payload, model_name)
    if expected_window_days is None:
        return checkpoint

    if checkpoint.input_kind != "window":
        raise CheckpointError(
            f"{path.name} is a point model; expected_window_days does not "
            f"apply to it"
        )
    corrected = replace(checkpoint, window_days=int(expected_window_days))
    _assert_accepts_window(corrected, int(expected_window_days))
    return corrected


def _assert_accepts_window(checkpoint: Checkpoint, days: int) -> None:
    """Push one zero batch through and let the shape check do the talking."""
    model = checkpoint.build()
    probe = torch.zeros(1, model.n_features, days)
    with torch.no_grad():
        model(probe)


def headline_checkpoints(
    seed: int, *, sample: bool = False, directory: Path = CHECKPOINT_DIR
) -> dict[str, Checkpoint]:
    """Both models' checkpoints for one seed, keyed by model name.

    Loading them together is not a convenience: every guard in §13 that
    compares the two — same standardiser, same rows, same seed — needs both in
    hand, and a Stage 7 that loads one model at a time is a Stage 7 that can
    silently compare seed 42 against seed 123.
    """
    suffix = ".sample" if sample else ""
    loaded: dict[str, Checkpoint] = {}
    for model_name in (MODEL_A_NAME, MODEL_B_NAME):
        loaded[model_name] = load_checkpoint(
            directory / f"{model_name}_seed{seed}{suffix}.pt"
        )
    for model_name, checkpoint in loaded.items():
        if checkpoint.seed != seed:
            raise CheckpointError(
                f"{checkpoint.path.name} says it was trained on seed "
                f"{checkpoint.seed}, not {seed}. Comparing two models across "
                f"different splits is not a comparison."
            )
    if loaded[MODEL_B_NAME].window_days != WINDOW_DAYS:
        raise CheckpointError(
            f"the reported Model B must be the {WINDOW_DAYS}-day model; "
            f"{loaded[MODEL_B_NAME].path.name} says "
            f"{loaded[MODEL_B_NAME].window_days}"
        )
    return loaded
