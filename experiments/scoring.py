import pandas as pd
from typing import Iterable, Tuple

def add_true_score(
    df: pd.DataFrame,
    map_col: str = "map50",
    weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.1, 0.1),
    group_cols: Iterable[str] = ("model", "epochs", "tag"),
) -> pd.DataFrame:

    T, P, M, D = weights
    df = df.copy()

    def _per_group(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["delta_map"] = sub[map_col].max() - sub[map_col]
        sub["true_score"] = (
            T * sub["throughput"]
            - P * sub["avg_power"]
            - M * sub["avg_mem"]
            - D * sub["delta_map"]
        )
        return sub

    return df.groupby(list(group_cols), sort=False, group_keys=False).apply(_per_group)
