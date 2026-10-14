"""Launching the two commands the UI is allowed to launch, and streaming them.

The UI shells out. It does not import a use case and call it, for two reasons
that are worth stating because the alternative looks simpler:

**A crashed pipeline must not take the web server down.** Stage 1 reads 16 GB
and Stage 6 trains a CNN. Either can die in a way that leaves the interpreter
in no state to serve a page explaining that it died.

**Subprocess isolation is what makes the refusal checkable.** A command this
module will not launch cannot be launched at all — there is no import of
:mod:`pipeline.score` or :mod:`pipeline.train` anywhere in ``ui/`` to be
reached by another route. That turns "we will not open the test split" from a
promise into a property of the code.

**The allowlist is checked before a command line exists.**
:func:`require_allowed` runs at the top of :func:`launch`, before the argument
vector is built and long before :class:`subprocess.Popen`. Since brief S1 it
reads a ``config.UI_ALLOWED_COMMANDS`` holding exactly two names —
``pipeline.train`` and ``pipeline.score``. Neither of them can reach
``evaluation.run_phase2``, which opens the test split, or
``evaluation.run_stage7``, which rewrites the frozen report; and proving that
about two commands is a much shorter argument than proving it about seven.

**Where the launcher went.** The Popen loop itself now lives in
:func:`pipeline.commands.launch`, because ``pipeline.train`` needs the same
loop to run its six stages and two copies of it would eventually disagree
about what an exit code means. What did *not* move is the allowlist: this
module checks it, and this module is the only place that ever hands
:func:`pipeline.commands.launch` a name that came from outside the source.

Subprocesses are launched with ``python -u -m`` so their output arrives a line
at a time. Without ``-u``, Python buffers stdout when it is a pipe and the page
would show nothing for eight minutes and then everything at once, which is the
failure mode this whole streaming layer exists to prevent.

**How a result gets back.** The page renders a structured block — per-seed
table, guards, manifest — that used to be a dict returned by a function call in
this process. It is now produced in another one, so :meth:`Task.launch` passes
``--result-json`` naming a file under ``runs/``, and reads the verdict out of it
when the process ends. Nothing about the *streamed* output changed: this file
is a side channel, not a log line.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from config import (
    PROJECT_ROOT,
    UI_ALLOWED_COMMANDS,
    UI_LOG_BUFFER_LINES,
    UI_RUN_DIR,
)
from pipeline import commands
from pipeline.commands import EXIT_MEANINGS, Completed

logger = logging.getLogger(__name__)


class CommandNotAllowed(RuntimeError):
    """A module the UI is not permitted to run. Raised before any process."""


_END: Final[object] = object()


def require_allowed(module: str) -> str:
    """Return ``module`` if the UI may run it, otherwise raise.

    Deliberately not a boolean: a caller that has to remember to check a
    boolean is a caller that will one day forget.
    """
    if module not in UI_ALLOWED_COMMANDS:
        allowed = ", ".join(UI_ALLOWED_COMMANDS)
        raise CommandNotAllowed(
            f"{module!r} is not on the UI's allowlist and will not be "
            f"launched. Permitted: {allowed}. In particular "
            f"'evaluation.run_phase2' is absent permanently — it opens the "
            f"test split and writes the audit marker, and is a command a "
            f"human types on purpose."
        )
    return module


def launch(
    module: str,
    args: Sequence[str] = (),
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> Completed:
    """Check the allowlist, then run one command to completion.

    The check is the whole of what this function adds, and it happens first —
    before an argv exists, before a process exists, and whatever the caller
    passes. Everything after it is :func:`pipeline.commands.launch`.
    """
    require_allowed(module)
    return commands.launch(module, args, cwd=cwd, env=env, on_line=on_line)


class LineStream:
    """A growing list of output lines that several readers can follow live.

    A page opened halfway through a run has to see what it missed, so the
    stream keeps a bounded backlog and hands it to each new subscriber before
    switching them to live delivery. The lock is held only while snapshotting
    that backlog, so a slow browser cannot stall the pipeline reader thread.
    """

    def __init__(self, capacity: int = UI_LOG_BUFFER_LINES) -> None:
        self._buffer: deque[str] = deque(maxlen=capacity)
        self._subscribers: list[queue.Queue[Any]] = []
        self._lock = threading.Lock()
        self._closed = False

    def publish(self, line: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._buffer.append(line)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(line)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(_END)

    @property
    def closed(self) -> bool:
        return self._closed

    def backlog(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._buffer)

    def follow(self) -> Iterator[str]:
        """Yield the backlog, then every later line, then stop at close."""
        subscriber: queue.Queue[Any] = queue.Queue()
        with self._lock:
            backlog = tuple(self._buffer)
            if self._closed:
                yield from backlog
                return
            self._subscribers.append(subscriber)
        try:
            yield from backlog
            while True:
                item = subscriber.get()
                if item is _END:
                    return
                yield item
        finally:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)


@dataclass(slots=True)
class Task:
    """One button press: its live output, its verdict and its result block."""

    id: str
    label: str
    stream: LineStream = field(default_factory=LineStream)
    state: str = "running"
    exit_code: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    thread: threading.Thread | None = None

    def say(self, line: str) -> None:
        """Put one line on the page. The UI's own voice, not a subprocess's."""
        self.stream.publish(line)

    def launch(self, module: str, args: Sequence[str] = ()) -> None:
        """Run one allowlisted command and take its verdict as this task's.

        The command owns the outcome. Everything this method does afterwards
        is copying: the streamed lines already went to the page as they
        arrived, and the state, the error and the result block are read out of
        the file the command was told to write them to.
        """
        envelope = UI_RUN_DIR / f"task-{self.id}.json"
        finished = launch(
            module,
            [*(str(a) for a in args), "--result-json", str(envelope)],
            on_line=self.say,
        )
        self._absorb(envelope, finished)

    def _absorb(self, envelope: Path, finished: Completed) -> None:
        """Take the command's verdict, or invent one if it died before writing.

        Silent on purpose: whatever the command decided, it already said so on
        its own stdout and the page already has those lines. Saying it again
        here would make the page stutter. The one case that does speak is a
        process that ended without leaving a verdict at all — killed, or dead
        before it could write — because then nobody has said anything.
        """
        try:
            payload = json.loads(envelope.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("no verdict from %s: %s", finished.module, error)
            self.fail(
                f"{finished.module} {finished.meaning()}",
                exit_code=finished.exit_code or 1,
            )
            return
        self.state = payload["state"]
        self.exit_code = payload["exit_code"]
        self.error = payload["error"]
        self.result = payload["result"]

    def refuse(self, reason: str) -> None:
        """End the task because the project's rules say it may not proceed.

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "state": self.state,
            "exit_code": self.exit_code,
            "exit_meaning": (
                EXIT_MEANINGS.get(self.exit_code) if self.exit_code is not None
                else None
            ),
            "error": self.error,
            "result": self.result,
        }


TASKS: Final[dict[str, Task]] = {}


def start(label: str, work: Callable[[Task], None]) -> Task:
    """Begin a task in a worker thread and return it immediately.

    ``work`` owns the outcome: it calls :meth:`Task.launch`, which copies the
    command's verdict onto the task. Anything it lets escape becomes a failure
    with the exception's message, because a task that ends in neither state
    leaves the page spinning forever.
    """
    task = Task(id=uuid.uuid4().hex[:12], label=label)
    TASKS[task.id] = task

    def runner() -> None:
        try:
            work(task)
            if not task.finished:
                task.fail("finished without reporting an outcome")
        except Exception as error:
            logger.exception("task %s failed", task.id)
            task.fail(f"{type(error).__name__}: {error}")
        finally:
            task.stream.close()

    task.thread = threading.Thread(target=runner, name=f"task-{task.id}", daemon=True)
    task.thread.start()
    return task


def sse_frames(task: Task) -> Iterator[str]:
    """The Server-Sent Events body for one task: every line, then the verdict."""
    for line in task.stream.follow():
        yield _frame("line", line)
    yield _frame("done", task.state)


def _frame(event: str, data: str) -> str:
    body = "\n".join(f"data: {part}" for part in data.split("\n"))
    return f"event: {event}\n{body}\n\n"
