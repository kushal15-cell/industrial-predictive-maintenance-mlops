import pandas as pd


def validate_cmapss_data(df: pd.DataFrame) -> None:
    """
    Validate raw CMAPSS FD001 data before feature engineering.

    Raises
    ------
    ValueError
        If any validation check fails.
    """

    expected_columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty.")

    if df.shape[1] != 26:
        raise ValueError(
            f"Validation failed: Expected 26 columns, got {df.shape[1]}."
        )

    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Validation failed: Missing columns: {missing_columns}"
        )

    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        raise ValueError(
            f"Validation failed: Found {missing_values} missing values."
        )

    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        raise ValueError(
            f"Validation failed: Found {duplicate_rows} duplicate rows."
        )

    if (df["unit_number"] <= 0).any():
        raise ValueError("Validation failed: unit_number contains non-positive values.")

    if (df["time_in_cycles"] <= 0).any():
        raise ValueError("Validation failed: time_in_cycles contains non-positive values.")

    print("Data validation passed successfully.")