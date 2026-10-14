"""Stage 0 acceptance test: prove the environment and the data both work.

Run it after creating the virtual environment:

    python verify_setup.py            # fast; reads 5 rows per log
    python verify_setup.py --full     # also counts every row, including http.csv

Every check prints PASS, FAIL or INFO. The exit code is 0 only if no check
failed. This file is allowed to print; pipeline modules use logging.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import config


_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_failures: list[str] = []


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def _paint(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}" if _supports_colour() else text


def ok(message: str) -> None:
    print(f"  {_paint('PASS', _GREEN)}  {message}")


def fail(message: str) -> None:
    _failures.append(message)
    print(f"  {_paint('FAIL', _RED)}  {message}")


def info(message: str) -> None:
    print(f"  {_paint('INFO', _YELLOW)}  {message}")


def detail(message: str) -> None:
    print(f"        {_paint(message, _DIM)}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
)


def check_python() -> None:
    section("1. Python version")
    version = sys.version_info
    resolved = f"{version.major}.{version.minor}.{version.micro}"
    detail(f"interpreter: {sys.executable}")
    if version[:2] >= config.MIN_PYTHON:
        ok(f"Python {resolved} (need >= {config.MIN_PYTHON[0]}.{config.MIN_PYTHON[1]})")
    else:
        fail(
            f"Python {resolved} is too old; this project needs "
            f">= {config.MIN_PYTHON[0]}.{config.MIN_PYTHON[1]}"
        )


def check_imports() -> None:
    section("2. Dependencies")
    for module_name, distribution in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            fail(f"{distribution} does not import: {exc}")
            continue
        version = getattr(module, "__version__", "unknown version")
        ok(f"{distribution:<13} {version}")


def check_torch() -> None:
    section("3. torch works")
    try:
        import torch
    except ImportError as exc:
        fail(f"torch does not import: {exc}")
        return
    try:
        device = config.resolve_device()
    except Exception as exc:
        fail(f"config.resolve_device() raised {type(exc).__name__}: {exc}")
        return
    try:
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        product = a @ b
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
        if not torch.allclose(product, expected):
            fail(f"matmul on device {device!r} gave {product.tolist()}, "
                 f"expected {expected.tolist()}")
            return
        ok(f"tensor matmul on device {device!r} -> {product.tolist()}")
        if device == "mps":
            detail("MPS is in use. This is a portability decision, not a speed "
                   "claim -- at ~640 and ~5,000 parameters CPU is comparable or "
                   "faster, because kernel-launch overhead dominates. Running "
                   "on Metal is what proves the pipeline is device-agnostic. "
                   "Both models must use the SAME device (Stages 5, 6 and 7).")
        elif torch.backends.mps.is_available():
            detail("MPS is available but resolve_device() returned cpu.")
        else:
            detail("No Metal on this machine; everything runs on CPU. Fine -- "
                   "just record it, and use the same device for Stages 6 and 7.")
    except Exception as exc:
        fail(f"torch tensor operation on device {device!r} raised "
             f"{type(exc).__name__}: {exc}")


def check_project_directories() -> None:
    section("4. Project directories")
    config.ensure_directories()
    for label, path in (
        ("PROJECT_ROOT", config.PROJECT_ROOT),
        ("DATA_DIR", config.DATA_DIR),
        ("OUTPUT_DIR", config.OUTPUT_DIR),
        ("CHECKPOINT_DIR", config.CHECKPOINT_DIR),
        ("FIGURE_DIR", config.FIGURE_DIR),
    ):
        if path.is_dir():
            ok(f"{label:<14} {path}")
        else:
            fail(f"{label} is missing and could not be created: {path}")


def check_feature_layout() -> None:
    section("5. Feature layout")
    if len(set(config.FEATURE_NAMES)) != len(config.FEATURE_NAMES):
        fail("FEATURE_NAMES contains a duplicate name")
        return
    if config.N_FEATURES != 18:
        fail(f"expected 18 features, config declares {config.N_FEATURES}")
        return
    ok(f"{config.N_FEATURES} feature names, unique, order frozen")
    detail(", ".join(config.FEATURE_NAMES[:5]) + ", ...")
    try:
        config.SplitFractions(train=0.6, val=0.2, test=0.3)
    except ValueError:
        ok("SplitFractions rejects fractions that do not sum to 1.0")
    else:
        fail("SplitFractions accepted 0.6/0.2/0.3, which does not sum to 1.0")


def check_sample_dataset() -> Path | None:
    section("6. Sample dataset")
    try:
        directory = config.resolve_dataset(sample=True)
    except config.DatasetNotFoundError as exc:
        fail(str(exc).splitlines()[0])
        for line in str(exc).splitlines()[1:]:
            detail(line.strip())
        return None
    ok(f"resolved to {directory}")

    for name in config.LOG_NAMES:
        spec = config.LOG_SPECS[name]
        path = directory / spec.filename
        if not path.is_file():
            fail(f"{spec.filename} is missing from {directory}")
            continue
        size_mb = path.stat().st_size / 1_048_576
        ok(f"{spec.filename:<11} present ({size_mb:,.1f} MB)")
    return directory


def check_schemas(directory: Path, count_rows: bool) -> None:
    section("7. Log schemas")
    import pandas as pd

    for name in config.LOG_NAMES:
        spec = config.LOG_SPECS[name]
        path = directory / spec.filename
        if not path.is_file():
            continue
        try:
            head = pd.read_csv(path, nrows=5, dtype="str")
        except Exception as exc:
            fail(f"{spec.filename} could not be read: "
                 f"{type(exc).__name__}: {exc}")
            continue
        found = tuple(head.columns)
        if found == spec.columns:
            ok(f"{spec.filename:<11} {', '.join(found)}")
        else:
            fail(
                f"{spec.filename} header mismatch\n"
                f"        expected: {', '.join(spec.columns)}\n"
                f"        found:    {', '.join(found)}"
            )
        detail(f"{len(head)} rows read; first date {head['date'].iloc[0]}")

    if not count_rows:
        detail("row counts skipped; re-run with --full to count them "
               "(http.csv is ~150 MB in the sample)")
        return

    section("7b. Row counts")
    for name in config.LOG_NAMES:
        spec = config.LOG_SPECS[name]
        path = directory / spec.filename
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle) - 1
        expected = spec.expected_sample_rows
        if expected is None:
            info(
                f"{spec.filename:<11} {rows:>9,} rows "
                f"(no recorded count for r{config.ACTIVE_RELEASE} yet)"
            )
        elif rows == expected:
            ok(f"{spec.filename:<11} {rows:>9,} rows")
        else:
            info(
                f"{spec.filename:<11} {rows:>9,} rows "
                f"(expected {expected:,} -- the sample was "
                f"probably rebuilt; not an error)"
            )


def check_full_dataset() -> None:
    section("8. Full dataset (informational)")
    drive, *mirrors = config.FULL_DATASET_CANDIDATES
    if config.external_drive_mounted():
        info(f"external drive mounted: {drive}")
    else:
        info(f"external drive not mounted ({drive}) -- fine, "
             f"development runs on the sample")
    for mirror in mirrors:
        if mirror.is_dir():
            info(f"local mirror available: {mirror}")
    try:
        answers = config.resolve_answers()
    except config.DatasetNotFoundError:
        info("answers directory not reachable yet; Stage 2 will need it")
    else:
        info(f"answers directory: {answers}")


def check_parquet_roundtrip() -> None:
    section("9. Parquet round-trip")
    import pandas as pd

    target = config.DATA_DIR / "_smoketest.parquet"
    frame = pd.DataFrame(
        {
            config.USER_COLUMN: ["ACM2278", "CDE1846", "PLJ1771"],
            config.DAY_COLUMN: pd.to_datetime(
                ["2010-01-04", "2010-01-05", "2010-01-06"]
            ),
            config.FEATURE_NAMES[0]: [3, 1, 7],
        }
    )
    try:
        frame.to_parquet(target, engine="pyarrow", index=False)
        restored = pd.read_parquet(target, engine="pyarrow")
        if not restored.equals(frame):
            fail("parquet round-trip changed the dataframe")
            return
        ok(f"wrote, read back and compared 3 rows via pyarrow ({target.name})")
    except Exception as exc:
        fail(f"parquet round-trip raised {type(exc).__name__}: {exc}")
    finally:
        target.unlink(missing_ok=True)
    if not target.exists():
        ok("smoke-test file deleted")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the InnerSight Stage 0 environment and data paths."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also count every row in every log, including the 150 MB http.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("InnerSight - Stage 0 verification")
    print(f"project root: {config.PROJECT_ROOT}")

    check_python()
    check_imports()
    check_torch()
    check_project_directories()
    check_feature_layout()
    directory = check_sample_dataset()
    if directory is not None:
        check_schemas(directory, args.full)
    check_full_dataset()
    check_parquet_roundtrip()

    section("Result")
    if _failures:
        print(f"  {_paint('FAILED', _RED)} - {len(_failures)} check(s) did not pass:")
        for message in _failures:
            print(f"    - {message.splitlines()[0]}")
        return 1
    print(f"  {_paint('ALL CHECKS PASSED', _GREEN)} - Stage 0 is ready; "
          f"Stage 1 lives in extraction/extract.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
