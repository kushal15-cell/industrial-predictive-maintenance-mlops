import pandas as pd
from pathlib import Path


def load_cmapss_fd001(raw_path: Path) -> pd.DataFrame:
    """
    Load NASA CMAPSS FD001 training data.

    Parameters
    ----------
    raw_path : Path
        Path to the raw CMAPSSData folder.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with proper column names.
    """

    columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train_path = raw_path / "train_FD001.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"File not found: {train_path}")

    df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df