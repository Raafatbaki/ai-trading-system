from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/splits")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]


def main():
    frames = []

    for symbol in SYMBOLS:
        file_path = DATA_DIR / f"{symbol}_tsmom_v2_multiframe_research.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data["pf_clean"] = data["profit_factor"].replace([np.inf, -np.inf], np.nan)

    summary = data.groupby(
        [
            "timeframe",
            "rebalance",
            "dataset",
        ],
        as_index=False,
    ).agg(
        assets=("symbol", "nunique"),
        total_trades=("trades", "sum"),
        positive_assets=(
            "return_pct",
            lambda x: int((x > 0).sum()),
        ),
        avg_return_pct=(
            "return_pct",
            "mean",
        ),
        median_return_pct=(
            "return_pct",
            "median",
        ),
        worst_asset_return_pct=(
            "return_pct",
            "min",
        ),
        avg_profit_factor=(
            "pf_clean",
            "mean",
        ),
        avg_max_dd_pct=(
            "max_dd_pct",
            "mean",
        ),
        avg_exposure_pct=(
            "exposure_pct",
            "mean",
        ),
    )

    numeric_columns = [
        "avg_return_pct",
        "median_return_pct",
        "worst_asset_return_pct",
        "avg_profit_factor",
        "avg_max_dd_pct",
        "avg_exposure_pct",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(2)

    print()
    print("=" * 120)
    print("TSMOM V2 - CROSS-ASSET SUMMARY")
    print("=" * 120)
    print()
    print(summary.to_string(index=False))

    output_file = DATA_DIR / "tsmom_v2_cross_asset_summary.csv"

    summary.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
