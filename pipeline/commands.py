"""Running one stage as a subprocess, and saying what happened while it runs.

Also, since brief R8, the one precondition both use cases share:
:func:`require_answers`. It lives here for the reason stated below -- the part
both sides need and neither should own twice -- rather than in
:mod:`pipeline.score`, where it was written. ``pipeline.train`` importing it
from ``score`` would have made one entry point depend on the other, and would
have pulled :mod:`pipeline.inference` and the whole scoring stack into a
process whose job is to refuse in under a second.

Moved out of ``ui/jobs.py`` by brief S1, unchanged apart from one deletion:
the allowlist check is no longer here. That check belongs at the boundary it
protects -- the web layer -- and it stayed there, in
:func:`ui.jobs.require_allowed`. What is left here is the part both sides need
and neither should own twice: build an argv, stream the child's output a line
at a time, and turn an exit code into a sentence.

**Subprocesses, still, one level down.** ``pipeline.train`` shells out to the
six stage modules for the same reason the web layer shells out to
``pipeline.train``: a stage that dies must not take its parent with it, and --
the reason that actually forces it -- Stages 5 and 6 write to
``config.checkpoint_path()``, whose root is ``Path(config.__file__).parent``.
Only a child process with its own working directory and its own copy of
``config.py`` can be pointed somewhere else. See :mod:`pipeline.sandbox`.

``python -u -m`` for the same reason it was always ``-u``: without it Python
buffers stdout when it is a pipe, and a training run would show nothing for
eight minutes and then everything at once.

**Why the narration has its own logger.** Everything this process writes to
stdout is read by a browser, because ``ui/jobs.py`` merges the child's stderr
into its stdout and puts every line on the page. Before brief S1 the page saw
exactly the lines the UI chose to say; the ``logger.info`` calls inside
``store``, ``detect`` and ``inference`` went to the server's terminal and no
further. Handing the root logger to stdout would have added all of them to the
page. So :func:`configure_logging` puts the root logger at WARNING -- a real
problem still surfaces -- and gives the narration a logger of its own.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from config import (
    PROJECT_ROOT,
    UI_JOB_TIMEOUT_SECONDS,
    DatasetNotFoundError,
    insiders_csv_path,
)

logger = logging.getLogger(__name__)

EXIT_MEANINGS: Final[dict[int, str]] = {
    0: "completed cleanly",
    1: "failed",
    2: "bad device — the requested accelerator is not available",
    3: "a guard failed; the run refused to report a number",
    4: "ordering or audit trail refused the run",
}

_NARRATION: Final[str] = "innersight.page"


def configure_logging() -> None:
    """Send the narration to stdout unprefixed, and nothing else with it.

    Unprefixed because these lines are shown to a person in a browser, and
    because that is exactly what they looked like before this package existed.
    ``modeling/run_model_a.py`` already configures itself this way; the stages
    that print a timestamp keep printing one, since they are separate
    processes and this function does not reach them.
    """
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s", stream=sys.stdout
    )
    narration = logging.getLogger(_NARRATION)
    narration.setLevel(logging.INFO)
    narration.propagate = False
    if not narration.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        narration.addHandler(handler)


@dataclass(slots=True)
class Completed:
    """One subprocess that has finished, and how."""

    module: str
    argv: tuple[str, ...]
    exit_code: int
    lines: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def meaning(self) -> str:
        return EXIT_MEANINGS.get(self.exit_code, f"exited {self.exit_code}")


def launch(
    module: str,
    args: Sequence[str] = (),
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    timeout: int = UI_JOB_TIMEOUT_SECONDS,
) -> Completed:
    """Run one module to completion, streaming its output.

    Blocking on purpose. The caller owns the ordering of a multi-stage run; a
    launcher that also managed concurrency would put two different kinds of
    sequencing in one place.

    This function does not decide what may run. ``ui.jobs.launch`` checks the
    allowlist before calling it, and the two callers inside this package pass
    module names written as literals in their own source.
    """
    argv = (sys.executable, "-u", "-m", module, *map(str, args))
    logger.info("launching %s", " ".join(argv[2:]))
    if on_line:
        on_line(f"$ python -u -m {module} {' '.join(map(str, args))}")

    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    count = 0
    assert process.stdout is not None
    for raw in process.stdout:
        count += 1
        if on_line:
            on_line(raw.rstrip("\n"))

    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait()
        if on_line:
            on_line(f"! killed after {timeout} seconds without finishing")

    completed = Completed(module, tuple(argv), exit_code, count)
    logger.info("%s %s", module, completed.meaning())
    return completed


@dataclass(slots=True)
class Reporter:
    """One run's voice and its verdict, in the words the page already shows.

    This is ``ui.jobs.Task`` with the threading, the subscriber queues and the
    HTTP identity taken out -- what is left is the four things the moved code
    actually called: say a line, run a stage, and end in one of three states.
    Keeping the method names and the exact strings is not tidiness; it is what
    makes "no behaviour changed" a claim anyone can check by reading two files
    side by side.

    :attr:`envelope` is how the verdict crosses back to a web server that is
    no longer in the same process. ``ui/jobs.py`` reads it from the file named
    by ``--result-json`` and copies it onto its own task.
    """

    state: str = "running"
    exit_code: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    lines: int = field(default=0)

    def say(self, line: str) -> None:
        """Put one line on the page, and on the terminal of whoever ran this."""
        self.lines += 1
        logging.getLogger(_NARRATION).info("%s", line)

    def run(
        self,
        module: str,
        args: Sequence[str] = (),
        *,
        cwd: Path = PROJECT_ROOT,
        env: dict[str, str] | None = None,
    ) -> Completed:
        """Run one stage, streaming it into this run's output."""
        return launch(module, args, cwd=cwd, env=env, on_line=self.say)

    def refuse(self, reason: str) -> None:
        """End because the project's rules say this may not proceed.

        Distinct from :meth:`fail`: a refusal is the system working. The page
        colours them differently and so should anyone reading this code.
        """
        self.state = "refused"
        self.error = reason
        self.exit_code = 4
        self.say("")
        self.say("REFUSED — " + reason)

    def fail(self, reason: str, exit_code: int = 1) -> None:
        self.state = "failed"
        self.error = reason
        self.exit_code = exit_code
        self.say("")
        self.say(f"FAILED ({EXIT_MEANINGS.get(exit_code, exit_code)}) — {reason}")

    def succeed(self, result: dict[str, Any]) -> None:
        self.state = "succeeded"
        self.exit_code = 0
        self.result = result

    @property
    def finished(self) -> bool:
        return self.state != "running"

    @property
    def envelope(self) -> dict[str, Any]:
        """The verdict, as the web layer's task will read it."""
        return {
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "result": self.result,
        }


def require_answers(report: Reporter) -> Path | None:
    """The ground-truth file, or a refusal that says what to plug in.

    The folder of raw logs is not enough. Both use cases run ``labeling.label``
    -- scoring needs labels for AUPRC, training needs them for the target
    column -- labels come from ``answers/insiders.csv``, and that file lives
    outside the release folder, usually on the external drive. Resolving it up
    front turns a stack trace ninety seconds into an extraction into a sentence
    before the run starts.

    A refusal, not a failure: an unplugged drive is the system working. Exit 4,
    and the page colours it accordingly.
    """
    try:
        answers = insiders_csv_path()
    except DatasetNotFoundError as error:
        report.refuse(
            f"the ground-truth file could not be found, so there is nothing "
            f"to score against — a folder of logs has no labels in it. "
            f"{error} Set INNERSIGHT_ANSWERS_DIR, or plug the drive in."
        )
        return None
    if not answers.is_file():
        report.refuse(f"{answers} does not exist; labelling cannot run without it.")
        return None
    return answers
