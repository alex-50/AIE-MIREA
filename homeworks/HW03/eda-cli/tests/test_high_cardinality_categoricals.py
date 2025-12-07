import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)


def test_quality_flags_high_cardinality_categoricals():
    df = pd.DataFrame(
        {
            "category": [f"cat_{i}" for i in range(100)],
            "value": list(range(100)),
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_high_cardinality_categoricals"] is True
    assert 0.0 <= flags["quality_score"] <= 1.0
