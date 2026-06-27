import pandas as pd


def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life target column.

    RUL = max cycle for each engine - current cycle
    """

    df = df.copy()

    max_cycles = df.groupby("unit_number")["time_in_cycles"].max()
    df["max_cycle"] = df["unit_number"].map(max_cycles)

    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]

    df = df.drop(columns=["max_cycle"])

    return df


def cap_rul(df: pd.DataFrame, cap_value: int = 125) -> pd.DataFrame:
    """
    Cap RUL values to reduce the effect of very large early-life RUL values.

    In industry, this is common because early engine life degradation is not always linear.
    """

    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap_value)

    return df