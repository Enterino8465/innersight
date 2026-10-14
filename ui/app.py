"""Flask routes and Server-Sent Events. No science happens in this file.

Since brief S1 that sentence is structural rather than aspirational. There are
two things this project does — train a model on a folder, and score a folder
with a stored model — and each of them is now one command:

.. code-block:: bash

    python -m pipeline.train --release 4.2
    python -m pipeline.score --release 4.2 --model frozen

**What is left here.** A route reads two or three values off a request, turns
them into an argument vector, and hands that to :func:`ui.jobs.Task.launch`.
The lines the command prints go to the browser as they arrive; its verdict
comes back through the file named by ``--result-json``. Nothing in this file
derives a table, scores a row, writes a checkpoint or decides anything about a
release — the only judgement it makes is which of two commands to start.

**The three read-only views are still calls, not commands**, and deliberately:
:func:`pipeline.detect.identify` reads five header lines,
:func:`pipeline.store.list_runs` reads the manifests, and
:func:`pipeline.results.trained_runs` reshapes those manifests into tables.
They compute nothing, write nothing and finish instantly, and an instant
answer is worth having on stage. The line this file may not cross is drawn where it matters:
``ui/`` cannot reach :mod:`pipeline.score`, :mod:`pipeline.train`,
:mod:`pipeline.inference` or :mod:`pipeline.sandbox`, and there is a test that
walks the import graph and says so.

Bound to ``127.0.0.1`` only. This is a laptop tool for one person standing in
front of one examiner, not a service.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from config import (
    DEVICE_CHOICES,
    DEVICE_PREFERENCE,
    PROJECT_ROOT,
    RELEASE_CAPABILITY,
    SEEDS,
    UI_DATASET_CHOICES,
    UI_HOST,
    UI_PORT,
    ensure_directories,
    resolve_release_folder,
)
from pipeline import detect, results, store
from ui import jobs

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        devices=DEVICE_CHOICES,
        default_device=DEVICE_PREFERENCE,
        seeds=", ".join(str(s) for s in SEEDS),
        project_root=str(PROJECT_ROOT),
        folders=folder_choices(),
    )


@app.get("/api/folders")
def api_folders() -> Response:
    """The dropdown's contents, re-resolved. Drives get plugged in mid-session."""
    return jsonify(folder_choices())


def folder_choices() -> list[dict[str, Any]]:
    """The two releases, where each one currently is, and whether it is there.

    The menu offers a *place to look*, not a claim about what is there. Its
    label never reaches the pipeline: whichever entry is chosen, the path goes
    through :func:`pipeline.detect.identify`, which reads the five header lines
    and decides for itself. Swap these two entries and the page would notice.
    """
    found: list[dict[str, Any]] = []
    for release, candidates in UI_DATASET_CHOICES.items():
        path = resolve_release_folder(release)
        found.append({
            "release": release,
            "label": f"CERT r{release}",
            "path": str(path) if path else "",
            "available": path is not None,
            "capability": RELEASE_CAPABILITY.get(release, "unknown"),
            "looked_in": [str(candidate) for candidate in candidates],
        })
    return found


def locate(release: str) -> tuple[Path | None, str]:
    """Where a chosen release's folder is, or a sentence about why not.

    One line, because the answer has to be the same one the commands give and
    there is no version of "the same sentence" that survives being written
    twice. The implementation moved to :mod:`pipeline.detect` under brief S1.
    """
    return detect.locate(release)


@app.post("/api/detect")
def api_detect() -> Response:
    """Identify a chosen release's folder. Cheap: five header lines."""
    release = str((request.json or {}).get("release", ""))
    folder, where = locate(release)
    if folder is None:
        return jsonify({"allowed": False, "release": None, "report": where})
    detection = detect.identify(folder)
    return jsonify(detection.as_dict() | {"report": detect.format_report(detection)})


@app.get("/api/runs")
def api_runs() -> Response:
    """Every model the UI has trained and kept."""
    return jsonify([manifest.as_dict() for manifest in store.list_runs()])


@app.get("/api/models")
def api_models() -> Response:
    """Every model this UI has trained. Reads manifests; computes nothing.

    The third view, and the only one with no run behind it. It is a record of
    the training sessions started from this page -- nothing else. It does not
    open ``final_results.json``, does not load a checkpoint and does not score
    a row; each run's numbers were written by the run itself and kept in its
    own ``manifest.json``. An empty list is the correct answer on a machine
    where nothing has been trained here yet, and the page says so rather than
    filling the space with the command line's results.
    """
    return jsonify(results.trained_runs())


@app.post("/api/infer")
def api_infer() -> Response:
    """Start button 1 and return the task id the page will follow."""
    payload = request.json or {}
    device = str(payload.get("device", DEVICE_PREFERENCE))
    run_id = str(payload.get("stored_run", "")).strip()

    args = ["--device", device]
    if payload.get("sample"):
        args.append("--sample")
    else:
        args += ["--release", str(payload.get("release", ""))]
    if run_id:
        args += ["--model", run_id]
    if payload.get("allow_transfer"):
        args.append("--allow-transfer")

    task = jobs.start("inference", lambda t: t.launch("pipeline.score", args))
    return jsonify({"task": task.id})


@app.post("/api/train")
def api_train() -> Response:
    """Start button 2 and return the task id the page will follow."""
    payload = request.json or {}
    seeds = payload.get("seeds") or SEEDS
    max_epochs = payload.get("max_epochs")

    args = [
        "--release", str(payload.get("release", "")),
        "--device", str(payload.get("device", DEVICE_PREFERENCE)),
        "--seeds", *(str(int(seed)) for seed in seeds),
    ]
    if max_epochs:
        args += ["--max-epochs", str(int(max_epochs))]

    task = jobs.start("training", lambda t: t.launch("pipeline.train", args))
    return jsonify({"task": task.id})


@app.get("/api/task/<task_id>")
def api_task(task_id: str) -> Response:
    task = jobs.TASKS.get(task_id)
    if task is None:
        return jsonify({"error": f"no task {task_id}"}), 404
    return jsonify(task.as_dict())


@app.get("/api/stream/<task_id>")
def api_stream(task_id: str) -> Response:
    """The live log. One `data:` frame per line, closed when the task ends."""
    task = jobs.TASKS.get(task_id)
    if task is None:
        return Response("event: done\ndata: missing\n\n", mimetype="text/event-stream")
    return Response(
        jobs.sse_frames(task),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main(argv: list[str] | None = None) -> int:
    """Serve the page on the loopback interface and nowhere else."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ui.app",
        description="Stage 8 — the local InnerSight web UI.",
    )
    parser.add_argument("--port", type=int, default=UI_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_directories()
    logger.info("InnerSight UI on http://%s:%d", UI_HOST, args.port)
    app.run(host=UI_HOST, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
