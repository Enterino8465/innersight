"""Shared I/O utilities used across the InnerSight backend.

Public API:
  safe_json_write(filepath, data)  — atomic write with fsync
  safe_json_read(filepath, default) — read with graceful fallback
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def safe_json_write(filepath: str, data) -> None:
    tmp = filepath + '.tmp'
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, filepath)
    except Exception as exc:
        logger.error('safe_json_write failed for %s: %s', filepath, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def safe_json_read(filepath: str, default: Any = None) -> Any:
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as exc:
        logger.warning('safe_json_read failed for %s: %s — returning default', filepath, exc)
        return default
