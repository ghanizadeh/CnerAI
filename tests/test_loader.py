"""tests/test_loader.py"""
import io
import pytest
import pandas as pd
from core.data.loader import load_csv, extended_describe, validate_df


@pytest.fixture
def sample_csv():
    content = "a,b,c\n1,2,x\n3,4,y\n5,,z\n"
    return io.StringIO(content)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 3, 5], "b": [2.0, 4.0, None], "c": ["x", "y", "z"]})


def test_load_csv_returns_dataframe(sample_csv):
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)


def test_extended_describe_columns(sample_df):
    desc = extended_describe(sample_df)
    assert "null_count" in desc.columns
    assert "null_%" in desc.columns
    assert "n_unique" in desc.columns


def test_validate_df_empty():
    warnings = validate_df(pd.DataFrame())
    assert any("empty" in w.lower() for w in warnings)


def test_validate_df_high_missing(sample_df):
    df = sample_df.copy()
    df["b"] = None  # 100% missing
    warnings = validate_df(df)
    assert any("missing" in w.lower() for w in warnings)


def test_validate_df_constant_col(sample_df):
    df = sample_df.copy()
    df["const"] = 42
    warnings = validate_df(df)
    assert any("constant" in w.lower() for w in warnings)


def test_validate_df_clean(sample_df):
    # No null%, no constant, small shape → empty warnings
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    warnings = validate_df(df)
    assert warnings == []
