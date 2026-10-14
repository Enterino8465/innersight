from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
)
from modeling.metrics import (
    Evaluation,
    PredictionStats,
    evaluate,
    log_evaluation,
    precision_at_k,
    probabilities,
)
from modeling.metrics import auprc as average_precision

logger = logging.getLogger(__name__)


class InputView(Protocol):
    """What the loop needs from an input source, and nothing more."""

    def __len__(self) -> int: ...
    def gather(self, index: torch.Tensor) -> torch.Tensor: ...
    def to(self, device: str | torch.device) -> InputView: ...


class TrainingError(RuntimeError):
    """A training run that must stop rather than produce a number."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Every knob, in one object, shared by both models verbatim."""

    seed: int
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOP_PATIENCE
    use_pos_weight: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Five numbers per epoch, not one.

    ``mean_predicted_prob`` and ``distinct`` are the collapse detector. A run
    where they flatten to a constant has failed in the way described at the
    top of this module, and the loss columns will not show it.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_auprc: float
    val_p_at_50: float
    mean_predicted_prob: float
    distinct: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        return (
            f"epoch {self.epoch:>3}  train_loss {self.train_loss:.5f}  "
            f"val_loss {self.val_loss:.5f}  val_AUPRC {self.val_auprc:.4f}  "
            f"val_P@50 {self.val_p_at_50:.3f}  "
            f"mean_prob {self.mean_predicted_prob:.5f}  "
            f"distinct {self.distinct:>7,}  {self.seconds:5.1f}s"
        )


@dataclass(slots=True)
class TrainingResult:
    """The outcome of one (model, seed) run."""

    history: list[EpochRecord]
    best_epoch: int
    best_auprc: float
    best_state: dict[str, torch.Tensor]
    evaluation: Evaluation
    val_probabilities: np.ndarray
    pos_weight: float | None
    n_parameters: int
    device: str
    stopped_early: bool
    seconds_total: float
    collapse_fired: bool = False

    @property
    def seconds_per_epoch(self) -> float:
        return self.seconds_total / max(len(self.history), 1)

    def history_as_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.history]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch explicitly at the start of every run.

    ``torch.manual_seed`` covers the MPS generator too. Bit-exact
    reproducibility *across* backends is still not guaranteed, which is one
    more reason the project fixes one device for Stages 5, 6 and 7 and records
    it rather than letting each run pick.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_pos_weight(labels: torch.Tensor) -> float:
    """``n_negatives / n_positives`` — from the training labels handed in.

    Passing the *training* labels is the caller's job and the caller does it
    from :meth:`modeling.data.PreparedData.pos_weight`. Computing this from
    the whole table would put the test set's class balance into the loss
    function, which is leakage that nothing will ever flag.
    """
    positives = float(labels.sum().item())
    if positives == 0:
        raise TrainingError("no positive rows in the training labels")
    return (labels.numel() - positives) / positives


@torch.no_grad()
def predict(
    model: nn.Module,
    view: InputView,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Raw logits for every row of a view, in order, on the CPU."""
    model.eval()
    n = len(view)
    out = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, batch_size):
        index = torch.arange(start, min(start + batch_size, n), dtype=torch.int64)
        logits = model(view.gather(index).to(device))
        out[start : start + index.numel()] = logits.detach().float().cpu()
    return out


def train_model(
    model: nn.Module,
    train_view: InputView,
    train_labels: torch.Tensor,
    val_view: InputView,
    val_labels: torch.Tensor,
    config: TrainingConfig,
    *,
    val_scenarios: np.ndarray | None = None,
    pos_weight: float | None = None,
) -> TrainingResult:
    """Train one model and return its best-AUPRC checkpoint.

    Nothing in this function inspects the model. Stage 6 calls it with a CNN
    and a 28-day window view and changes nothing else.
    """
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_view = train_view.to(device)
    val_view = val_view.to(device)
    y_train = train_labels.to(device).float()
    y_val_cpu = val_labels.detach().float().cpu()
    y_val_np = y_val_cpu.numpy().astype(np.int64)

    if pos_weight is None and config.use_pos_weight:
        pos_weight = compute_pos_weight(y_train.cpu())
    effective_weight = pos_weight if config.use_pos_weight else None

    weight_tensor = (
        torch.tensor([effective_weight], dtype=torch.float32, device=device)
        if effective_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.info(
        "training %s: %s parameters, device %s, batch %d, lr %g, "
        "pos_weight %s",
        type(model).__name__, f"{n_parameters:,}", config.device,
        config.batch_size, config.learning_rate,
        f"{effective_weight:.1f}" if effective_weight is not None
        else "DISABLED (plain BCE)",
    )

    n_train = len(train_view)
    if n_train != y_train.numel():
        raise TrainingError(
            f"training view has {n_train} rows but {y_train.numel()} labels"
        )

    shuffler = torch.Generator().manual_seed(config.seed)

    history: list[EpochRecord] = []
    best_auprc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    best_probs = np.zeros(y_val_np.size, dtype=np.float64)
    epochs_without_improvement = 0
    collapse_fired = False
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(n_train, generator=shuffler)
        running_loss = 0.0

        for start in range(0, n_train, config.batch_size):
            index = order[start : start + config.batch_size]
            inputs = train_view.gather(index).to(device)
            targets = y_train.index_select(0, index.to(device))
            optimiser.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise TrainingError(
                    f"loss became {loss.item()} at epoch {epoch}; training "
                    f"cannot continue. Check for unclipped z-scores or a "
                    f"learning rate far too high."
                )
            loss.backward()
            optimiser.step()
            running_loss += float(loss.item()) * index.numel()

        train_loss = running_loss / n_train

        val_logits = predict(model, val_view, config.batch_size, config.device)
        val_loss = float(
            criterion(val_logits.to(device), y_val_cpu.to(device)).item()
        )
        val_probs = probabilities(val_logits)
        stats = PredictionStats.of(val_probs)
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auprc=average_precision(y_val_np, val_probs),
            val_p_at_50=precision_at_k(y_val_np, val_probs, 50),
            mean_predicted_prob=stats.mean,
            distinct=stats.distinct,
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        logger.info("%s", record.line())

        if epoch == 1:
            _log_first_epoch_guards(stats, record, float(y_val_np.mean()))
        if stats.collapsed:
            collapse_fired = True

        if record.val_auprc > best_auprc:
            best_auprc = record.val_auprc
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            best_probs = val_probs
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(
                    "early stop: %d epochs without a better validation AUPRC "
                    "(best %.4f at epoch %d)",
                    config.patience, best_auprc, best_epoch,
                )
                break

    seconds_total = time.perf_counter() - started
    stopped_early = len(history) < config.max_epochs

    evaluation = evaluate(y_val_np, best_probs, val_scenarios)
    log_evaluation(evaluation, label="validation (best epoch)")
    logger.info(
        "best epoch %d, best val AUPRC %.4f, threshold %.6f, %.1fs total "
        "(%.2fs/epoch on %s)",
        best_epoch, best_auprc,
        evaluation.threshold.threshold if evaluation.threshold else float("nan"),
        seconds_total, seconds_total / max(len(history), 1), config.device,
    )

    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_auprc=best_auprc,
        best_state=best_state,
        evaluation=evaluation,
        val_probabilities=best_probs,
        pos_weight=effective_weight,
        n_parameters=n_parameters,
        device=config.device,
        stopped_early=stopped_early,
        seconds_total=seconds_total,
        collapse_fired=collapse_fired,
    )


def _log_first_epoch_guards(
    stats: PredictionStats, record: EpochRecord, baseline: float
) -> None:
    """The after-epoch-1 checks from the brief, run when they are still cheap."""
    logger.info(
        "  guard | predicted prob min %.6f max %.6f mean %.6f std %.6f",
        stats.minimum, stats.maximum, stats.mean, stats.std,
    )
    if stats.collapsed:
        logger.warning(
            "  guard | COLLAPSE: %d distinct predicted values after epoch 1 "
            "(floor %d for %s rows). Almost certainly pos_weight is not being "
            "applied.", stats.distinct, stats.collapse_floor,
            f"{stats.n_rows:,}",
        )
    else:
        logger.info("  guard | %s distinct predicted values, ok", f"{stats.distinct:,}")
    if record.val_auprc <= baseline:
        logger.warning(
            "  guard | val AUPRC %.4f is not above the random baseline %.5f "
            "after epoch 1", record.val_auprc, baseline,
        )
    else:
        logger.info(
            "  guard | val AUPRC %.4f > random baseline %.5f, ok",
            record.val_auprc, baseline,
        )
