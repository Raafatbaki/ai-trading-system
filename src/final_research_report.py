from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/splits")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

TIMEFRAME = "1h"
REBALANCE = "daily"
MARKET = "spot"
STRATEGY = "tsmom_v2"


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan

    profits = trades.loc[
        trades["pnl_usdt"] > 0,
        "pnl_usdt",
    ].sum()

    losses = abs(
        trades.loc[
            trades["pnl_usdt"] < 0,
            "pnl_usdt",
        ].sum()
    )

    if losses == 0:
        if profits > 0:
            return float("inf")
        return 0.0

    return float(profits / losses)


def load_dev_validation() -> pd.DataFrame:
    rows = []

    for symbol in SYMBOLS:
        file_path = DATA_DIR / f"{symbol}_tsmom_v2_multiframe_research.csv"

        df = pd.read_csv(file_path)

        selected = df[
            (df["timeframe"] == TIMEFRAME) & (df["rebalance"] == REBALANCE)
        ].copy()

        rows.append(selected)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def load_walk_forward() -> pd.DataFrame:
    file_path = DATA_DIR / (
        f"tsmom_v2_{TIMEFRAME}_" f"{REBALANCE}_continuous_walk_forward.csv"
    )

    return pd.read_csv(file_path)


def load_holdout() -> pd.DataFrame:
    rows = []

    for symbol in SYMBOLS:
        base = (
            f"{symbol}_{MARKET}_{TIMEFRAME}_"
            f"historical_holdout_{STRATEGY}_{REBALANCE}"
        )

        trades_file = DATA_DIR / f"{base}_trades.csv"

        equity_file = DATA_DIR / f"{base}_equity.csv"

        signals_file = DATA_DIR / f"{base}_signals.csv"

        trades = pd.read_csv(trades_file)
        equity = pd.read_csv(equity_file)
        signals = pd.read_csv(signals_file)

        initial_capital = 100.0

        if trades.empty:
            final_capital = initial_capital
            wins = 0
            losses = 0
            return_pct = 0.0
            forced_exits = 0
        else:
            final_capital = float(trades.iloc[-1]["capital_after_trade"])

            wins = int((trades["pnl_usdt"] > 0).sum())

            losses = int((trades["pnl_usdt"] < 0).sum())

            return_pct = (final_capital / initial_capital - 1) * 100

            forced_exits = int(trades["forced_exit"].sum())

        first_price = float(signals.iloc[0]["open"])

        last_price = float(signals.iloc[-1]["close"])

        buy_hold_pct = (last_price / first_price - 1) * 100

        max_dd_pct = float(equity["drawdown_pct"].min())

        exposure_pct = equity["position_open"].astype(bool).mean() * 100

        rows.append(
            {
                "symbol": symbol,
                "trades": len(trades),
                "wins": wins,
                "losses": losses,
                "return_pct": return_pct,
                "buy_hold_pct": buy_hold_pct,
                "profit_factor": profit_factor(trades),
                "max_dd_pct": max_dd_pct,
                "exposure_pct": exposure_pct,
                "forced_exits": forced_exits,
            }
        )

    return pd.DataFrame(rows)


def main():
    dev_val = load_dev_validation()
    walk_forward = load_walk_forward()
    holdout = load_holdout()

    print()
    print("=" * 110)
    print("TSMOM V2 - FINAL RESEARCH REPORT")
    print("=" * 110)

    print()
    print("FROZEN CONFIGURATION")
    print("-" * 110)

    print("Market:        SPOT")
    print("Symbols:       BTCUSDT, ETHUSDT, SOLUSDT")
    print("Timeframe:     1h")
    print("Rebalance:     DAILY")
    print("Momentum:      28 days")
    print("Trend filter:  200 days")
    print("Mode:          LONG / CASH")
    print("Short:         DISABLED")
    print("Leverage:      DISABLED")

    print()
    print("=" * 110)
    print("DEVELOPMENT + VALIDATION")
    print("=" * 110)

    columns = [
        "symbol",
        "dataset",
        "trades",
        "win_rate_pct",
        "return_pct",
        "buy_hold_pct",
        "profit_factor",
        "max_dd_pct",
        "exposure_pct",
    ]

    print(dev_val[columns].round(2).to_string(index=False))

    print()
    print("=" * 110)
    print("CONTINUOUS WALK-FORWARD")
    print("=" * 110)

    total_trades = int(walk_forward["trades"].sum())

    total_wins = int(walk_forward["wins"].sum())

    total_losses = int(walk_forward["losses"].sum())

    profitable_folds = int((walk_forward["return_pct"] > 0).sum())

    losing_folds = int((walk_forward["return_pct"] < 0).sum())

    flat_folds = int((walk_forward["return_pct"] == 0).sum())

    print(f"Fold results:        {len(walk_forward)}")
    print(f"Profitable folds:    {profitable_folds}")
    print(f"Losing folds:        {losing_folds}")
    print(f"Flat folds:          {flat_folds}")
    print(f"Total trades:        {total_trades}")
    print(f"Total wins:          {total_wins}")
    print(f"Total losses:        {total_losses}")

    if total_trades:
        win_rate = total_wins / total_trades * 100
    else:
        win_rate = 0.0

    print(f"Overall win rate:    {win_rate:.2f}%")
    print("Average fold return: " f"{walk_forward['return_pct'].mean():.2f}%")
    print("Median fold return:  " f"{walk_forward['return_pct'].median():.2f}%")
    print("Worst fold return:   " f"{walk_forward['return_pct'].min():.2f}%")
    print("Best fold return:    " f"{walk_forward['return_pct'].max():.2f}%")
    print("Average max DD:      " f"{walk_forward['max_dd_pct'].mean():.2f}%")

    print()
    print("=" * 110)
    print("HISTORICAL HOLDOUT - FINAL TEST")
    print("=" * 110)

    print(holdout.round(2).to_string(index=False))

    print()
    print("=" * 110)
    print("HOLDOUT AGGREGATE")
    print("=" * 110)

    print(f"Total trades:     " f"{holdout['trades'].sum()}")

    print(f"Total wins:       " f"{holdout['wins'].sum()}")

    print(f"Forced exits:     " f"{holdout['forced_exits'].sum()}")

    print(f"Average return:   " f"{holdout['return_pct'].mean():.2f}%")

    print(f"Median return:    " f"{holdout['return_pct'].median():.2f}%")

    print(f"Average max DD:   " f"{holdout['max_dd_pct'].mean():.2f}%")

    output_dir = DATA_DIR / "final_report"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dev_val.to_csv(
        output_dir / "development_validation.csv",
        index=False,
    )

    walk_forward.to_csv(
        output_dir / "walk_forward.csv",
        index=False,
    )

    holdout.to_csv(
        output_dir / "historical_holdout.csv",
        index=False,
    )

    print()
    print(f"Report data saved to: {output_dir}")


if __name__ == "__main__":
    main()
