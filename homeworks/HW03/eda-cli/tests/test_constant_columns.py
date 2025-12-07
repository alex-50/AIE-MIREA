import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)


def test_quality_flags_constant_columns():
    df = pd.DataFrame(
        {
            "constant_col": [1, 1, 1, 1],
            "value": [10, 20, 30, 40],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert 0.0 <= flags["quality_score"] <= 1.0
