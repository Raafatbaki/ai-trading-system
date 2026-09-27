from pathlib import Path

import pandas as pd

MARKET = "spot"
TIMEFRAME = "1h"
DATASET = "validation"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

DATA_DIRECTORY = Path("data/splits")


def load_symbol_data(
    symbol: str,
) -> pd.DataFrame:

    trades_file = DATA_DIRECTORY / (
        f"{symbol}_{MARKET}_{TIMEFRAME}_" f"{DATASET}_tsmom_trades.csv"
    )

    signals_file = DATA_DIRECTORY / (
        f"{symbol}_{MARKET}_{TIMEFRAME}_" f"{DATASET}_tsmom_signals.csv"
    )

    if not trades_file.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_file}")

    if not signals_file.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_file}")

    trades = pd.read_csv(
        trades_file,
        parse_dates=[
            "entry_signal_time",
            "entry_time",
            "exit_time",
        ],
    )

    signals = pd.read_csv(
        signals_file,
        parse_dates=["timestamp"],
    )

    # We need the momentum value that created
    # the BUY signal.
    entry_signals = signals[signals["signal"] == "BUY"][
        [
            "timestamp",
            "momentum_4w",
            "close",
        ]
    ].copy()

    entry_signals = entry_signals.rename(
        columns={
            "timestamp": "entry_signal_time",
            "close": "signal_close",
        }
    )

    merged = trades.merge(
        entry_signals,
        on="entry_signal_time",
        how="left",
    )

    merged["symbol"] = symbol

    merged["momentum_pct"] = merged["momentum_4w"] * 100

    merged["duration_hours"] = (
        merged["exit_time"] - merged["entry_time"]
    ).dt.total_seconds() / 3600

    merged["result"] = "LOSS"

    merged.loc[
        merged["net_return_pct"] > 0,
        "result",
    ] = "WIN"

    merged.loc[
        merged["net_return_pct"] == 0,
        "result",
    ] = "FLAT"

    return merged


def print_symbol_report(
    symbol: str,
    dataframe: pd.DataFrame,
) -> None:

    print()
    print("=" * 95)
    print(f"{symbol} - TSMOM VALIDATION")
    print("=" * 95)

    winners = dataframe[dataframe["net_return_pct"] > 0]

    losers = dataframe[dataframe["net_return_pct"] < 0]

    print()
    print(f"Trades:      {len(dataframe)}")
    print(f"Winners:     {len(winners)}")
    print(f"Losers:      {len(losers)}")

    print(f"Avg return:  " f"{dataframe['net_return_pct'].mean():.2f}%")

    print(f"Avg momentum at entry: " f"{dataframe['momentum_pct'].mean():.2f}%")

    print()

    columns = [
        "entry_signal_time",
        "entry_time",
        "entry_price",
        "momentum_pct",
        "exit_time",
        "exit_price",
        "duration_hours",
        "net_return_pct",
        "result",
        "forced_exit",
    ]

    report = dataframe[columns].copy()

    report["entry_price"] = report["entry_price"].round(4)

    report["exit_price"] = report["exit_price"].round(4)

    report["momentum_pct"] = report["momentum_pct"].round(2)

    report["duration_hours"] = report["duration_hours"].round(1)

    report["net_return_pct"] = report["net_return_pct"].round(2)

    print(report.to_string(index=False))


def print_combined_report(
    dataframe: pd.DataFrame,
) -> None:

    print()
    print("=" * 95)
    print("CROSS-ASSET VALIDATION SUMMARY")
    print("=" * 95)

    rows = []

    for symbol, group in dataframe.groupby("symbol"):
        winners = group[group["net_return_pct"] > 0]

        win_rate = len(winners) / len(group) * 100

        rows.append(
            {
                "symbol": symbol,
                "trades": len(group),
                "wins": len(winners),
                "win_rate_pct": win_rate,
                "avg_return_pct": group["net_return_pct"].mean(),
                "avg_entry_momentum_pct": group["momentum_pct"].mean(),
                "avg_duration_hours": group["duration_hours"].mean(),
            }
        )

    summary = pd.DataFrame(rows)

    numeric_columns = [
        "win_rate_pct",
        "avg_return_pct",
        "avg_entry_momentum_pct",
        "avg_duration_hours",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(2)

    print()
    print(summary.to_string(index=False))

    print()
    print("-" * 95)

    all_winners = dataframe[dataframe["net_return_pct"] > 0]

    print(f"Total trades: " f"{len(dataframe)}")

    print(f"Total winners: " f"{len(all_winners)}")

    print(f"Overall win rate: " f"{len(all_winners) / len(dataframe) * 100:.2f}%")

    print(f"Average trade: " f"{dataframe['net_return_pct'].mean():.2f}%")


def main():
    all_results = []

    for symbol in SYMBOLS:
        dataframe = load_symbol_data(symbol)

        all_results.append(dataframe)

        print_symbol_report(
            symbol,
            dataframe,
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    print_combined_report(combined)

    output_file = DATA_DIRECTORY / "tsmom_validation_analysis.csv"

    combined.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Full analysis saved to: " f"{output_file}")


if __name__ == "__main__":
    main()
