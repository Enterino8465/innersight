"""The §13 guard list, as code that runs rather than a checklist that is read.

Every line of the brief's guard block is a way this stage can produce a
plausible number instead of an error:

```
ok   test users never appear in any training or validation split
ok   standardiser loaded from checkpoint, not refit          <- print the values
ok   threshold loaded from checkpoint, not re-swept          <- print the values
ok   both models scored on identical test rows, per seed     <- print the counts
ok   no NaN or inf in any prediction
ok   test AUPRC > random baseline for both models
```

Two of them ask for values to be *printed*, which is the part that matters:
a boolean saying "the standardiser came from the checkpoint" is exactly as
convincing as no check at all. So :class:`GuardResult` carries a detail string
that goes in the log and in ``final_results.json``, and the standardiser guard
prints the first few means and stds beside the refit ones.

The leakage ceiling is in here too. §13's closing line — treat a test AUPRC
above ~0.95 as leakage rather than success — is the guard most likely to be
argued with at 3am, so it fails loudly and the entry points exit non-zero on
it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import LEAKAGE_AUPRC
from evaluation.checkpoints import Checkpoint
from evaluation.corpus import Corpus, EvalSplit
from evaluation.scoring import Scored

logger = logging.getLogger(__name__)

_STATS_SHOWN = 3


@dataclass(frozen=True, slots=True)
class GuardResult:
    """One check, its verdict, and the values that justify it."""

    name: str
    passed: bool
    detail: str
    fatal: bool = True

    def line(self) -> str:
        return f"{'ok  ' if self.passed else 'FAIL'} {self.name} — {self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "guard": self.name,
            "passed": self.passed,
            "fatal_if_failed": self.fatal,
            "detail": self.detail,
        }


def split_disjointness(corpus: Corpus, seeds: Sequence[int]) -> GuardResult:
    """Test users appear in no training or validation split, for any seed.

    Checked per seed rather than globally: a user who trains under seed 42 and
    is tested under seed 123 is fine — those are independent experiments — but
    a user who does both *within* one seed invalidates that seed entirely.
    """
    offences: list[str] = []
    for seed in seeds:
        users = corpus.users_for(seed)
        test = set(users["test"])
        for other in ("train", "val"):
            shared = test & set(users[other])
            if shared:
                offences.append(
                    f"seed {seed}: {len(shared)} user(s) in both test and "
                    f"{other}, e.g. {sorted(shared)[:3]}"
                )
    if offences:
        return GuardResult(
            "test users never appear in train or val", False, "; ".join(offences)
        )
    sizes = ", ".join(
        f"seed {seed}: {len(corpus.users_for(seed)['test'])} test users"
        for seed in seeds
    )
    return GuardResult(
        "test users never appear in train or val", True, f"disjoint on all "
        f"{len(seeds)} seed(s) ({sizes})",
    )


def standardiser_from_checkpoint(
    corpus: Corpus,
    seed: int,
    checkpoints: Iterable[Checkpoint],
    *,
    transfer: bool = False,
) -> GuardResult:
    """The checkpoint's standardiser is the one this data's own refit gives —
    or, on a transfer run, provably is not.

    This proves two things at once. That the evaluation applied the stored
    transform rather than fitting a new one on the rows it was about to score
    — the leak §9 warns about — and that both models stored the *same*
    transform, without which they are not on the same data.

    A mismatch here is not something to work around. It means either the
    dataset changed since training or the checkpoint is not what it claims,
    and either way the comparison is void.

    **``transfer=True`` asks the opposite question, and asks it just as
    strictly.** A cross-release run — an r5.2 model scoring r4.2's folder —
    *cannot* pass the identity check: the stored transform was fitted on other
    people. Relaxing the check for that case would leave the run unguarded, so
    the check is inverted instead. A transfer run is required to FAIL the
    identity test. If a foreign checkpoint's standardiser did match a
    train-only refit on this folder to 1e-9, the wrong checkpoint was loaded or
    the two releases are not what they claim, and this fails and says so.

    So there are two questions and every run answers exactly one of them:

    ===========  ==========================================  =============
    run          question                                    passes when
    ===========  ==========================================  =============
    matched      is this standardiser the train-only refit?  it matches
    transfer     is this standardiser genuinely foreign?     it does not
    ===========  ==========================================  =============

    **Neither kind of run can masquerade as the other, and no run is
    unguarded.** The guard's *name* differs between the two modes as well as
    its verdict, so a transfer result read later cannot be mistaken for a
    matched one, and ``transfer`` is never inferred from the data — it is
    declared by the caller, which had to be given an explicit opt-in to say it.

    The check that both checkpoints carry the *same* standardiser is unchanged
    in both modes. Two models trained on different scalings are not comparable
    no matter which release's rows were put through them.
    """
    checkpoints = list(checkpoints)
    refit = corpus.refit_train_standardiser(seed)

    shown = ", ".join(
        f"mean[{i}]={refit.mean[i]:+.6f} std[{i}]={refit.std[i]:.6f}"
        for i in range(min(_STATS_SHOWN, refit.mean.size))
    )

    problems: list[str] = []
    distances: list[str] = []
    for checkpoint in checkpoints:
        stored = checkpoint.standardiser
        mean_matches = np.allclose(stored.mean, refit.mean, rtol=0, atol=1e-9)
        std_matches = np.allclose(stored.std, refit.std, rtol=0, atol=1e-9)
        if transfer:
            if mean_matches and std_matches:
                problems.append(
                    f"{checkpoint.path.name}: standardiser MATCHES a "
                    f"train-only refit on this folder to 1e-9, so it is not "
                    f"foreign at all — either this is not the checkpoint it "
                    f"claims to be or these two releases are not two releases"
                )
            else:
                distances.append(
                    f"{checkpoint.path.name}: mean differs by up to "
                    f"{np.abs(stored.mean - refit.mean).max():.3e}, std by up "
                    f"to {np.abs(stored.std - refit.std).max():.3e}"
                )
            continue
        if not mean_matches:
            problems.append(
                f"{checkpoint.path.name}: mean differs by up to "
                f"{np.abs(stored.mean - refit.mean).max():.3e}"
            )
        if not std_matches:
            problems.append(
                f"{checkpoint.path.name}: std differs by up to "
                f"{np.abs(stored.std - refit.std).max():.3e}"
            )

    first = checkpoints[0].standardiser
    for other_checkpoint in checkpoints[1:]:
        other = other_checkpoint.standardiser
        if not (
            np.array_equal(first.mean, other.mean)
            and np.array_equal(first.std, other.std)
        ):
            problems.append(
                f"{checkpoints[0].path.name} and "
                f"{other_checkpoint.path.name} carry different standardisers "
                f"— the two models were not trained on the same scaling"
            )

    if transfer:
        name = (
            f"seed {seed}: standardiser is foreign to this folder, as "
            f"a transfer run requires"
        )
        if problems:
            return GuardResult(name, False, "; ".join(problems))
        return GuardResult(
            name, True,
            f"none of the {len(checkpoints)} checkpoint(s) match a train-only "
            f"refit on this folder — {'; '.join(distances)}; this folder's "
            f"refit is {shown}",
        )

    name = f"seed {seed}: standardiser loaded from checkpoint, not refit"
    if problems:
        return GuardResult(name, False, "; ".join(problems))
    return GuardResult(
        name, True,
        f"all {len(checkpoints)} checkpoint(s) match a train-only refit to "
        f"1e-9; {shown}",
    )


def threshold_from_checkpoint(scored: Sequence[Scored]) -> GuardResult:
    """The applied cutoff is byte-for-byte the one stored on disk.

    Cheap, and worth having anyway: it is the check that would have caught a
    re-sweep introduced later by someone tidying up.
    """
    problems: list[str] = []
    printed: list[str] = []
    for item in scored:
        stored = item.checkpoint.threshold
        applied = item.threshold
        printed.append(
            f"{item.model_name} seed {item.seed}: {applied:.6f} "
            f"({item.evaluation.threshold.predicted_positives:,} rows flagged)"
            if item.evaluation.threshold
            else f"{item.model_name} seed {item.seed}: {applied:.6f}"
        )
        if stored != applied:
            problems.append(
                f"{item.model_name} seed {item.seed}: applied {applied!r} but "
                f"the checkpoint stores {stored!r}"
            )
    name = "threshold loaded from checkpoint, not re-swept"
    if problems:
        return GuardResult(name, False, "; ".join(problems))
    return GuardResult(name, True, "; ".join(printed))


def identical_rows(seed: int, splits: Sequence[EvalSplit]) -> GuardResult:
    """Both models saw the same rows, in the same order, with the same labels.

    Compared by digest rather than by row count. A count catches a dropped
    row; a digest also catches a reordering, which is the failure that would
    leave both AUPRCs plausible and both wrong.
    """
    name = f"seed {seed}: both models scored on identical rows"
    digests = {split.fingerprint() for split in splits}
    counts = {split.n_rows for split in splits}
    positives = {split.n_positives for split in splits}
    if len(digests) != 1:
        return GuardResult(
            name, False,
            f"row digests differ: {sorted(digests)} — the two models are not "
            f"being compared on the same data",
        )
    first = splits[0]
    return GuardResult(
        name, True,
        f"{counts.pop():,} rows, {positives.pop():,} positive "
        f"({first.random_baseline:.4%}), digest {digests.pop()}",
    )


def finite_predictions(scored: Sequence[Scored]) -> GuardResult:
    """No NaN or inf anywhere in any model's predictions."""
    problems = [
        f"{item.model_name} seed {item.seed}: "
        f"{int((~np.isfinite(item.probs)).sum())} non-finite"
        for item in scored
        if not np.isfinite(item.probs).all()
    ]
    name = "no NaN or inf in any prediction"
    if problems:
        return GuardResult(name, False, "; ".join(problems))
    total = sum(item.probs.size for item in scored)
    return GuardResult(name, True, f"{total:,} predictions, all finite")


def beats_random(scored: Sequence[Scored]) -> GuardResult:
    """Every model ranks better than a coin flip on the split it was given."""
    problems = [
        f"{item.model_name} seed {item.seed}: AUPRC "
        f"{item.evaluation.auprc:.5f} <= baseline "
        f"{item.evaluation.random_baseline:.5f}"
        for item in scored
        if item.evaluation.auprc <= item.evaluation.random_baseline
    ]
    name = "AUPRC above the random baseline for every model and seed"
    if problems:
        return GuardResult(name, False, "; ".join(problems))
    worst = min(scored, key=lambda s: s.evaluation.lift)
    return GuardResult(
        name, True,
        f"worst is {worst.model_name} seed {worst.seed} at "
        f"{worst.evaluation.lift:.1f}x random",
    )


def not_suspiciously_good(scored: Sequence[Scored]) -> GuardResult:
    """An AUPRC above ~0.95 at a 0.4% positive rate is leakage, not a result.

    §13's closing line, and the one guard that fails on a number the project
    would otherwise be delighted by. That is exactly why it is enforced in
    code and why the entry points exit non-zero on it: the temptation to
    explain it away is proportional to how good it looks.
    """
    offenders = [
        f"{item.model_name} seed {item.seed}: AUPRC {item.evaluation.auprc:.4f}"
        for item in scored
        if item.evaluation.auprc >= LEAKAGE_AUPRC
    ]
    name = f"no model scores AUPRC >= {LEAKAGE_AUPRC:.2f} (leakage ceiling)"
    if offenders:
        return GuardResult(
            name, False,
            f"{'; '.join(offenders)}. Investigate before reporting: the most "
            f"likely causes are a split applied by row instead of by user, a "
            f"standardiser fitted on everything, or a label leaking into a "
            f"feature.",
        )
    best = max(scored, key=lambda s: s.evaluation.auprc)
    return GuardResult(
        name, True,
        f"highest is {best.model_name} seed {best.seed} at "
        f"{best.evaluation.auprc:.4f}",
    )


_AUPRC_REPRODUCTION_TOLERANCE = 1e-3


def reproduces_training(scored: Sequence[Scored]) -> GuardResult:
    """Re-scoring validation from the checkpoint reproduces the training number.

    This is the guard that makes every other Stage 7 number believable. If
    loading the weights, rebuilding the architecture, applying the stored
    standardiser and running the stored threshold reproduces the validation
    AUPRC that Stage 5 and Stage 6 logged, then the same machinery applied to
    test is doing what it claims. If it does not, nothing downstream means
    anything — and the failure would otherwise show up only as a test number
    that is quietly a little off.

    Not fatal on its own: a genuine device difference can move the last
    decimal, and the detail string prints both figures so the size of the gap
    is visible rather than merely classified.
    """
    problems: list[str] = []
    printed: list[str] = []
    for item in scored:
        recorded = item.checkpoint.best_val_auprc
        if recorded is None:
            continue
        recomputed = item.evaluation.auprc
        printed.append(
            f"{item.model_name} seed {item.seed}: {recomputed:.4f} vs "
            f"{recorded:.4f} recorded"
        )
        if abs(recomputed - recorded) > _AUPRC_REPRODUCTION_TOLERANCE:
            problems.append(
                f"{item.model_name} seed {item.seed}: recomputed "
                f"{recomputed:.6f} but the checkpoint recorded "
                f"{recorded:.6f} (delta {recomputed - recorded:+.2e})"
            )
    name = "re-scoring validation reproduces the training run's AUPRC"
    if problems:
        return GuardResult(name, False, "; ".join(problems), fatal=False)
    return GuardResult(name, True, "; ".join(printed) or "nothing recorded to compare")


def report(results: Sequence[GuardResult]) -> bool:
    """Log the whole block in the brief's format. Returns True if all passed."""
    logger.info("-" * 72)
    logger.info("guards")
    for result in results:
        (logger.info if result.passed else logger.error)("  %s", result.line())
    failed = [r for r in results if not r.passed]
    fatal = [r for r in failed if r.fatal]
    if fatal:
        logger.error(
            "%d guard(s) failed. Stop and report the failure rather than "
            "working around it — brief §9.", len(fatal),
        )
    elif failed:
        logger.warning("%d non-fatal guard(s) failed", len(failed))
    else:
        logger.info("  all %d guards passed", len(results))
    logger.info("-" * 72)
    return not fatal
