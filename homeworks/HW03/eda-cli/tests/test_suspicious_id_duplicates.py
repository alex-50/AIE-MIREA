import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)


def test_quality_flags_suspicious_id_duplicates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],  # есть дубликат "2"
            "value": [10, 20, 30, 40],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_suspicious_id_duplicates"] is True
    assert 0.0 <= flags["quality_score"] <= 1.0
