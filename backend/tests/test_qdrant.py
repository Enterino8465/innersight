"""Tests for the Qdrant-backed SuspectFinder (Phase 5).

``test_sync_and_query`` needs a running Qdrant server, so it is marked
``@pytest.mark.slow`` (excluded from the default run) and additionally skips
itself if Qdrant is not reachable. The graceful-failure test needs no server and
runs in the default suite.
"""

import time

import numpy as np
import pytest

from innersight.scoring.suspect_discovery import SuspectFinder


def test_suspect_finder_health_check_without_qdrant():
    # A closed port → connection refused → health_check returns False, no crash.
    finder = SuspectFinder(qdrant_url="http://localhost:6399")
    assert finder.health_check() is False


@pytest.mark.slow
def test_sync_and_query():
    finder = SuspectFinder(collection_name="innersight_test_phase5")
    if not finder.health_check():
        pytest.skip("Qdrant not running")

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((10, 128)).astype(float)
    embeddings[1] = embeddings[0] + 0.01 * rng.standard_normal(128)  # u1 ~ u0
    user_ids = [f"u{i}" for i in range(10)]
    metadata = [{"scenario": i % 3, "department": "Eng"} for i in range(10)]

    n = finder.sync_embeddings(embeddings, user_ids, metadata, "test")
    assert n == 10

    start = time.perf_counter()
    results = finder.find_similar("u0", k=5, version="test")
    elapsed = time.perf_counter() - start

    assert results, "expected similar users"
    assert all("similarity" in r for r in results)
    assert all(r["user_id"] != "u0" for r in results)  # self excluded
    assert results[0]["user_id"] == "u1"               # nearest is the planted twin
    assert elapsed < 0.1, f"query took {elapsed * 1000:.1f}ms (expected < 100ms)"
