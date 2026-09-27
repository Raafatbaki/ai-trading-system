from pathlib import Path

import pandas as pd

TRADES_FILE = Path("data/BTCUSDT_spot_1h_trades.csv")

REGIME_FILE = Path("data/BTCUSDT_spot_4h_regime.csv")


def calculate_profit_factor(
    dataframe: pd.DataFrame,
) -> float:
    winners = dataframe[dataframe["pnl_usdt"] > 0]

    losers = dataframe[dataframe["pnl_usdt"] < 0]

    gross_profit = winners["pnl_usdt"].sum()

    gross_loss = abs(losers["pnl_usdt"].sum())

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def calculate_compounded_return(
    dataframe: pd.DataFrame,
) -> float:
    capital = 1.0

    for value in dataframe["net_return_pct"]:
        capital *= 1 + value / 100

    return (capital - 1) * 100


def main():
    trades = pd.read_csv(
        TRADES_FILE,
        parse_dates=[
            "entry_signal_time",
            "entry_time",
            "exit_time",
        ],
    )

    regime = pd.read_csv(
        REGIME_FILE,
        parse_dates=["timestamp"],
    )

    # A 4h candle becomes usable only AFTER it has closed.
    regime["available_at"] = regime["timestamp"] + pd.Timedelta(hours=4)

    regime = regime[
        [
            "available_at",
            "market_regime",
        ]
    ].sort_values("available_at")

    trades = trades.sort_values("entry_signal_time")

    merged = pd.merge_asof(
        trades,
        regime,
        left_on="entry_signal_time",
        right_on="available_at",
        direction="backward",
    )

    print()
    print("=" * 75)
    print("1H TREND STRATEGY BY 4H MARKET REGIME")
    print("=" * 75)

    regimes = [
        "TRENDING_BULLISH",
        "TRENDING_BEARISH",
        "SIDEWAYS",
        "TRANSITION",
    ]

    print()
    print(
        f"{'Regime':<20}"
        f"{'Trades':>8}"
        f"{'Win %':>10}"
        f"{'Return %':>12}"
        f"{'PF':>10}"
        f"{'Avg %':>10}"
    )

    print("-" * 75)

    for current_regime in regimes:
        subset = merged[merged["market_regime"] == current_regime]

        if subset.empty:
            continue

        winners = subset[subset["pnl_usdt"] > 0]

        win_rate = len(winners) / len(subset) * 100

        total_return = calculate_compounded_return(subset)

        profit_factor = calculate_profit_factor(subset)

        average_trade = subset["net_return_pct"].mean()

        print(
            f"{current_regime:<20}"
            f"{len(subset):>8}"
            f"{win_rate:>10.2f}"
            f"{total_return:>12.2f}"
            f"{profit_factor:>10.2f}"
            f"{average_trade:>10.3f}"
        )

    print()


if __name__ == "__main__":
    main()
