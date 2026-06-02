"""Unit tests for backend/schema.py canonical contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from innersight.backend.schema import (
    FEATURE_NAMES,
    LOGON_SCHEMA,
    validate_dataframe,
)


def test_logon_schema_required_columns() -> None:
    assert LOGON_SCHEMA.required_columns == ['id', 'date', 'user', 'pc', 'activity']


def test_validate_dataframe_passes_on_valid() -> None:
    df = pd.DataFrame(columns=['id', 'date', 'user', 'pc', 'activity'])
    validate_dataframe(df, LOGON_SCHEMA, source='test')  # must not raise


def test_validate_dataframe_raises_on_missing_required() -> None:
    df = pd.DataFrame(columns=['id', 'date', 'user', 'pc'])  # missing 'activity'
    with pytest.raises(ValueError, match='activity'):
        validate_dataframe(df, LOGON_SCHEMA, source='test')


def test_feature_names_count() -> None:
    assert len(FEATURE_NAMES) == 18
