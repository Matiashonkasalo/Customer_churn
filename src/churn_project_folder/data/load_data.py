from pathlib import Path
import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw churn data from disk.

    Parameters
    ----------
    path : str or Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data as loaded from source.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)

    return df
