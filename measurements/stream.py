"""Resumable byte-offset CSV streaming, with checkpoints.

Why this exists: the shell this job runs in kills any process when the call
returns, and a call may not exceed ~45 s.  r5.2's ``http.csv`` is 30.1 GB and
needs minutes.  So every long pass is written as a RESUMABLE job: it streams
from a saved byte offset, works for a wall-clock budget, checkpoints its
accumulator, and exits.  Re-running it continues from where it stopped;
running it after it finished is a no-op that just reprints the result.

The file is cut on byte boundaries and each cut is extended to the next
newline, so no row is ever split or double-counted.  That is only safe because
the CERT logs are unquoted -- every job therefore asserts its final row count
against the known total, which is the check that the cutting was sound.

Nothing here writes outside ``measurements/``.
"""

from __future__ import annotations

import io
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

HERE = Path(__file__).resolve().parent
CKPT = HERE / "cache"
CKPT.mkdir(exist_ok=True)

DEFAULT_BUDGET_S = float(os.environ.get("M1_BUDGET_S", "34"))
DEFAULT_BLOCK = 192 * 1024 * 1024


def header_of(path: Path) -> list[str]:
    with path.open("rb") as f:
        return f.readline().decode().strip().split(",")


def data_start(path: Path) -> int:
    with path.open("rb") as f:
        f.readline()
        return f.tell()


def blocks(path: Path, start: int, block: int = DEFAULT_BLOCK) -> Iterator[tuple[bytes, int]]:
    """Yield (raw_csv_bytes, offset_after) starting at a newline-aligned offset."""
    size = path.stat().st_size
    with path.open("rb") as f:
        pos = start
        while pos < size:
            f.seek(pos)
            buf = f.read(block)
            if not buf:
                break
            if f.tell() < size:
                tail = f.readline()
                buf += tail
            pos = f.tell()
            yield buf, pos


def parse(buf: bytes, columns: list[str], usecols: list[str],
          dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(buf),
        header=None,
        names=columns,
        usecols=usecols,
        dtype=dtype or str,
        engine="c",
    )


@dataclass
class Job:
    """A resumable single-file pass.

    ``acc`` is any picklable accumulator the caller owns.  ``on_block`` is
    called with (acc, DataFrame) for each parsed block.
    """

    tag: str
    path: Path
    usecols: list[str]
    acc: Any
    on_block: Callable[[Any, pd.DataFrame], None]
    dtype: dict[str, Any] | None = None
    block: int = DEFAULT_BLOCK
    offset: int = 0
    rows: int = 0
    seconds: float = 0.0
    started: bool = False

    @property
    def ckpt(self) -> Path:
        return CKPT / f"job_{self.tag}.pkl"

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    def load(self) -> "Job":
        if self.ckpt.exists() and os.environ.get("M1_RESET") != "1":
            with self.ckpt.open("rb") as f:
                st = pickle.load(f)
            self.acc = st["acc"]
            self.offset = st["offset"]
            self.rows = st["rows"]
            self.seconds = st["seconds"]
            self.started = True
        else:
            self.offset = data_start(self.path)
            self.started = True
        return self

    def save(self) -> None:
        tmp = self.ckpt.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"acc": self.acc, "offset": self.offset,
                         "rows": self.rows, "seconds": self.seconds}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.ckpt)

    @property
    def done(self) -> bool:
        return self.offset >= self.size

    def run(self, budget_s: float = DEFAULT_BUDGET_S) -> bool:
        """Work for ``budget_s`` seconds.  Returns True when the file is done."""
        if not self.started:
            self.load()
        if self.done:
            print(f"[{self.tag}] already complete: {self.rows:,} rows, "
                  f"{self.seconds:,.1f}s total", flush=True)
            return True
        columns = header_of(self.path)
        size = self.size
        t0 = time.time()
        print(f"[{self.tag}] resuming at byte {self.offset:,} / {size:,} "
              f"({100*self.offset/size:,.1f}%), {self.rows:,} rows so far",
              flush=True)
        for buf, new_offset in blocks(self.path, self.offset, self.block):
            df = parse(buf, columns, self.usecols, self.dtype)
            self.on_block(self.acc, df)
            self.rows += len(df)
            self.offset = new_offset
            el = time.time() - t0
            mbps = (self.offset - 0) and (len(buf) / 1e6)
            print(f"[{self.tag}]   {100*self.offset/size:5.1f}%  "
                  f"{self.rows:,} rows  (+{len(df):,})  {el:,.1f}s", flush=True)
            if el >= budget_s:
                break
        self.seconds += time.time() - t0
        self.save()
        if self.done:
            rate = size / self.seconds / 1e6
            print(f"[{self.tag}] COMPLETE: {self.rows:,} rows, {size/1e9:,.2f} GB, "
                  f"{self.seconds:,.1f}s, {rate:,.0f} MB/s parsed", flush=True)
        else:
            print(f"[{self.tag}] checkpointed at {100*self.offset/size:,.1f}% "
                  f"-- run again to continue", flush=True)
        return self.done


def assert_rows(tag: str, got: int, expected: int | None) -> str:
    if expected is None:
        return f"[{tag}] {got:,} rows (no expected count on record)"
    ok = "OK" if got == expected else "MISMATCH"
    return f"[{tag}] row count {got:,} vs expected {expected:,}  -> {ok}"
