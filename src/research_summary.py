from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/splits")
WALK_FORWARD_DIR = Path("data/walk_forward")

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001


# ============================================================
# RESEARCH DECISIONS
# ============================================================

STRATEGY_DECISIONS = {
    "tsmom_v2": {
        "name": "TSMOM V2",
        "status": "BASELINE ONLY",
        "reason": (
            "Positive Development, Validation and Holdout, "
            "but trade count remains limited."
        ),
    },
    "donchian": {
        "name": "Donchian 20/10",
        "status": "REJECTED",
        "reason": ("Strong Development but failed cross-asset Validation."),
    },
    "bollinger": {
        "name": "Bollinger Mean Reversion",
        "status": "REJECTED",
        "reason": (
            "Failed to generalize across assets; "
            "high drawdown and weak Profit Factor."
        ),
    },
    "rsi_mr": {
        "name": "RSI Mean Reversion",
        "status": "REJECTED",
        "reason": ("No clear edge even on Development."),
    },
    "trend_pullback": {
        "name": "Trend Pullback",
        "status": "REJECTED",
        "reason": ("Walk-forward failed: most temporal folds were losing."),
    },
    "atr_breakout": {
        "name": "ATR Breakout",
        "status": "NOT CONFIRMED",
        "reason": (
            "Walk-forward was strong, but final Historical Holdout "
            "contained only five trades and was negative."
        ),
    },
}


# ============================================================
# FROZEN CONFIGURATIONS
# ============================================================

FROZEN_CONFIGURATIONS = {
    "tsmom_v2": {
        "timeframe": "1h",
        "rebalance": "daily",
    },
    "donchian": {
        "timeframe": "1h",
    },
    "bollinger": {
        "timeframe": "4h",
    },
    "rsi_mr": {
        "timeframe": "4h",
    },
    "trend_pullback": {
        "timeframe": "4h",
    },
    "atr_breakout": {
        "timeframe": "4h",
    },
}


# ============================================================
# WALK-FORWARD SOURCES
# ============================================================

WALK_FORWARD_SOURCES = {
    "tsmom_v2": Path("data/splits/" "tsmom_v2_1h_daily_continuous_walk_forward.csv"),
    "trend_pullback": Path(
        "data/walk_forward/" "trend_pullback_4h_walk_forward_folds.csv"
    ),
    "atr_breakout": Path("data/walk_forward/" "atr_breakout_4h_walk_forward_folds.csv"),
}


# ============================================================
# FILE IDENTIFICATION
# ============================================================


def identify_strategy(
    filename: str,
) -> str | None:

    if "_tsmom_v2_" in filename:
        return "tsmom_v2"

    if "_donchian_20_10_" in filename:
        return "donchian"

    if "_bollinger_mr_" in filename:
        return "bollinger"

    if "_rsi_mr_" in filename:
        return "rsi_mr"

    if "_trend_pullback_" in filename:
        return "trend_pullback"

    if "_atr_breakout_" in filename:
        return "atr_breakout"

    return None


def identify_dataset(
    filename: str,
) -> str | None:

    for dataset in [
        "historical_holdout",
        "development",
        "validation",
    ]:

        if f"_{dataset}_" in filename:
            return dataset

    return None


def identify_symbol(
    filename: str,
) -> str | None:

    for symbol in [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]:

        if filename.startswith(symbol):
            return symbol

    return None


def identify_timeframe(
    filename: str,
) -> str | None:

    for timeframe in [
        "15m",
        "1h",
        "4h",
    ]:

        if f"_spot_{timeframe}_" in filename:
            return timeframe

    return None


def identify_rebalance(
    filename: str,
) -> str | None:

    if "_daily_" in filename:
        return "daily"

    if "_weekly_" in filename:
        return "weekly"

    return None


# ============================================================
# CSV READING
# ============================================================


def safe_read_csv(
    file_path: Path,
) -> pd.DataFrame:

    try:

        return pd.read_csv(file_path)

    except pd.errors.EmptyDataError:

        return pd.DataFrame()


# ============================================================
# METRICS
# ============================================================


def calculate_trade_metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": float("nan"),
            "gross_profit_usdt": 0.0,
            "gross_loss_usdt": 0.0,
            "forced_exits": 0,
        }

    wins_dataframe = trades[trades["pnl_usdt"] > 0]

    losses_dataframe = trades[trades["pnl_usdt"] < 0]

    trades_count = len(trades)

    wins = len(wins_dataframe)

    losses = len(losses_dataframe)

    win_rate = wins / trades_count * 100

    gross_profit = float(wins_dataframe["pnl_usdt"].sum())

    gross_loss = abs(float(losses_dataframe["pnl_usdt"].sum()))

    if gross_loss > 0:

        profit_factor = gross_profit / gross_loss

    elif gross_profit > 0:

        profit_factor = float("inf")

    else:

        profit_factor = 0.0

    if "forced_exit" in trades.columns:

        forced_exits = int(trades["forced_exit"].sum())

    else:

        forced_exits = 0

    return {
        "trades": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "gross_profit_usdt": gross_profit,
        "gross_loss_usdt": gross_loss,
        "forced_exits": forced_exits,
    }


def calculate_buy_and_hold(
    signals: pd.DataFrame,
) -> float:

    if signals.empty:

        return float("nan")

    first_price = float(signals.iloc[0]["open"])

    last_price = float(signals.iloc[-1]["close"])

    entry_fee = INITIAL_CAPITAL * FEE_RATE

    investable_capital = INITIAL_CAPITAL - entry_fee

    quantity = investable_capital / first_price

    gross_final_value = quantity * last_price

    exit_fee = gross_final_value * FEE_RATE

    final_value = gross_final_value - exit_fee

    return ((final_value / INITIAL_CAPITAL) - 1) * 100


def calculate_aggregate_pf(
    dataframe: pd.DataFrame,
) -> float:

    if dataframe.empty:

        return float("nan")

    gross_profit = float(dataframe["gross_profit_usdt"].sum())

    gross_loss = float(dataframe["gross_loss_usdt"].sum())

    if gross_loss > 0:

        return gross_profit / gross_loss

    if gross_profit > 0:

        return float("inf")

    return 0.0


# ============================================================
# RELATED FILES
# ============================================================


def find_matching_file(
    trades_file: Path,
    suffix: str,
) -> Path:

    filename = trades_file.name.replace(
        "_trades.csv",
        suffix,
    )

    return trades_file.parent / filename


# ============================================================
# READ ONE EXPERIMENT
# ============================================================


def read_experiment(
    trades_file: Path,
) -> dict | None:

    filename = trades_file.name

    strategy = identify_strategy(filename)

    if strategy is None:
        return None

    dataset = identify_dataset(filename)

    symbol = identify_symbol(filename)

    timeframe = identify_timeframe(filename)

    rebalance = identify_rebalance(filename)

    if dataset is None or symbol is None or timeframe is None:
        return None

    # Ignore old TSMOM V2 files that did not
    # encode daily / weekly in their filenames.
    if strategy == "tsmom_v2" and rebalance is None:
        return None

    equity_file = find_matching_file(
        trades_file,
        "_equity.csv",
    )

    signals_file = find_matching_file(
        trades_file,
        "_signals.csv",
    )

    if not equity_file.exists():

        print("Skipping " f"{trades_file}: " "equity file missing.")

        return None

    trades = safe_read_csv(trades_file)

    equity = safe_read_csv(equity_file)

    if equity.empty:
        return None

    trade_metrics = calculate_trade_metrics(trades)

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    if "drawdown_pct" in equity.columns:

        max_drawdown = float(equity["drawdown_pct"].min())

    else:

        peak = equity["equity"].cummax()

        drawdown = ((equity["equity"] / peak) - 1) * 100

        max_drawdown = float(drawdown.min())

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    final_capital = float(equity.iloc[-1]["equity"])

    net_return = ((final_capital / INITIAL_CAPITAL) - 1) * 100

    # --------------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------------

    if signals_file.exists():

        signals = safe_read_csv(signals_file)

        buy_and_hold = calculate_buy_and_hold(signals)

    else:

        buy_and_hold = float("nan")

    return {
        "strategy": strategy,
        "strategy_name": (STRATEGY_DECISIONS[strategy]["name"]),
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset": dataset,
        "rebalance": rebalance,
        "trades": (trade_metrics["trades"]),
        "wins": (trade_metrics["wins"]),
        "losses": (trade_metrics["losses"]),
        "win_rate_pct": (trade_metrics["win_rate_pct"]),
        "gross_profit_usdt": (trade_metrics["gross_profit_usdt"]),
        "gross_loss_usdt": (trade_metrics["gross_loss_usdt"]),
        "profit_factor": (trade_metrics["profit_factor"]),
        "forced_exits": (trade_metrics["forced_exits"]),
        "net_return_pct": (net_return),
        "buy_hold_pct": (buy_and_hold),
        "max_drawdown_pct": (max_drawdown),
        "file": str(trades_file),
    }


# ============================================================
# COLLECT EXPERIMENTS
# ============================================================


def collect_experiments() -> pd.DataFrame:

    rows = []

    for trades_file in sorted(DATA_DIR.glob("*_trades.csv")):

        experiment = read_experiment(trades_file)

        if experiment is not None:

            rows.append(experiment)

    return pd.DataFrame(rows)


# ============================================================
# CONTINUOUS RETURN PER ASSET
# ============================================================


def calculate_continuous_returns(
    dataframe: pd.DataFrame,
) -> dict[str, float]:

    results = {}

    if dataframe.empty or "symbol" not in dataframe.columns:

        return results

    for symbol, group in dataframe.groupby("symbol"):

        group = group.sort_values("fold").reset_index(drop=True)

        # TSMOM continuous file.
        if "starting_equity" in group.columns and "ending_equity" in group.columns:

            start = float(group.iloc[0]["starting_equity"])

            end = float(group.iloc[-1]["ending_equity"])

            result = ((end / start) - 1) * 100

        # Newer walk-forward files.
        elif "start_capital" in group.columns and "end_capital" in group.columns:

            start = float(group.iloc[0]["start_capital"])

            end = float(group.iloc[-1]["end_capital"])

            result = ((end / start) - 1) * 100

        else:

            multiplier = (1 + (group["return_pct"] / 100)).prod()

            result = (multiplier - 1) * 100

        results[str(symbol)] = float(result)

    return results


def format_continuous_returns(
    returns: dict[str, float],
) -> str:

    if not returns:
        return "N/A"

    parts = []

    for symbol in sorted(returns.keys()):

        value = returns[symbol]

        parts.append(f"{symbol}={value:.2f}%")

    return " | ".join(parts)


# ============================================================
# WALK-FORWARD
# ============================================================


def collect_walk_forward() -> pd.DataFrame:

    rows = []

    for (
        strategy,
        file_path,
    ) in WALK_FORWARD_SOURCES.items():

        if not file_path.exists():

            print("Walk-forward file missing: " f"{file_path}")

            continue

        dataframe = pd.read_csv(file_path)

        if dataframe.empty:
            continue

        required_columns = [
            "trades",
            "wins",
            "losses",
            "return_pct",
            "profit_factor",
            "forced_exits",
            "symbol",
            "fold",
        ]

        missing = [
            column for column in required_columns if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(f"{file_path} missing columns: " f"{missing}")

        # Old TSMOM uses max_dd_pct.
        if "max_dd_pct" in dataframe.columns:

            drawdown_column = "max_dd_pct"

        # Newer files use max_drawdown.
        elif "max_drawdown" in dataframe.columns:

            drawdown_column = "max_drawdown"

        else:

            raise ValueError(f"No drawdown column in " f"{file_path}")

        total_folds = len(dataframe)

        profitable_folds = int((dataframe["return_pct"] > 0).sum())

        losing_folds = int((dataframe["return_pct"] < 0).sum())

        flat_folds = total_folds - profitable_folds - losing_folds

        pf_above_one = int((dataframe["profit_factor"] > 1).sum())

        total_trades = int(dataframe["trades"].sum())

        total_wins = int(dataframe["wins"].sum())

        total_losses = int(dataframe["losses"].sum())

        if total_trades > 0:

            win_rate = total_wins / total_trades * 100

        else:

            win_rate = 0.0

        continuous_returns = calculate_continuous_returns(dataframe)

        continuous_values = list(continuous_returns.values())

        if continuous_values:

            positive_assets = sum(value > 0 for value in continuous_values)

            negative_assets = sum(value < 0 for value in continuous_values)

            flat_assets = len(continuous_values) - positive_assets - negative_assets

            average_continuous_return = float(pd.Series(continuous_values).mean())

            worst_continuous_asset = float(min(continuous_values))

            best_continuous_asset = float(max(continuous_values))

        else:

            positive_assets = 0
            negative_assets = 0
            flat_assets = 0

            average_continuous_return = float("nan")

            worst_continuous_asset = float("nan")

            best_continuous_asset = float("nan")

        rows.append(
            {
                "strategy": strategy,
                "strategy_name": (STRATEGY_DECISIONS[strategy]["name"]),
                "folds": (total_folds),
                "profitable_folds": (profitable_folds),
                "losing_folds": (losing_folds),
                "flat_folds": (flat_folds),
                "pf_above_one_folds": (pf_above_one),
                "trades": (total_trades),
                "wins": (total_wins),
                "losses": (total_losses),
                "win_rate_pct": (win_rate),
                "average_fold_return_pct": float(dataframe["return_pct"].mean()),
                "median_fold_return_pct": float(dataframe["return_pct"].median()),
                "worst_fold_pct": float(dataframe["return_pct"].min()),
                "best_fold_pct": float(dataframe["return_pct"].max()),
                "average_max_drawdown_pct": float(dataframe[drawdown_column].mean()),
                "forced_exits": int(dataframe["forced_exits"].sum()),
                "assets": len(continuous_returns),
                "positive_assets": (positive_assets),
                "negative_assets": (negative_assets),
                "flat_assets": (flat_assets),
                "average_continuous_return_pct": (average_continuous_return),
                "worst_continuous_asset_pct": (worst_continuous_asset),
                "best_continuous_asset_pct": (best_continuous_asset),
                "continuous_returns": (format_continuous_returns(continuous_returns)),
                "file": str(file_path),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FROZEN CONFIG FILTER
# ============================================================


def filter_frozen_configuration(
    strategy: str,
    strategy_data: pd.DataFrame,
) -> pd.DataFrame:

    if strategy_data.empty:
        return strategy_data

    frozen = FROZEN_CONFIGURATIONS.get(strategy)

    if frozen is None:
        return strategy_data

    result = strategy_data.copy()

    timeframe = frozen.get("timeframe")

    if timeframe is not None:

        result = result[result["timeframe"] == timeframe].copy()

    rebalance = frozen.get("rebalance")

    if rebalance is not None:

        result = result[result["rebalance"] == rebalance].copy()

    return result


# ============================================================
# HELPERS
# ============================================================


def mean_value(
    dataframe: pd.DataFrame,
    column: str,
) -> float:

    if dataframe.empty:

        return float("nan")

    return float(dataframe[column].mean())


def total_value(
    dataframe: pd.DataFrame,
    column: str,
) -> int:

    if dataframe.empty:
        return 0

    return int(dataframe[column].sum())


# ============================================================
# STRATEGY SUMMARY
# ============================================================


def build_strategy_summary(
    experiments: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        strategy,
        decision,
    ) in STRATEGY_DECISIONS.items():

        strategy_data = experiments[experiments["strategy"] == strategy].copy()

        strategy_data = filter_frozen_configuration(
            strategy,
            strategy_data,
        )

        development = strategy_data[strategy_data["dataset"] == "development"].copy()

        validation = strategy_data[strategy_data["dataset"] == "validation"].copy()

        holdout = strategy_data[strategy_data["dataset"] == "historical_holdout"].copy()

        if walk_forward.empty or "strategy" not in walk_forward.columns:

            wf_data = pd.DataFrame()

        else:

            wf_data = walk_forward[walk_forward["strategy"] == strategy].copy()

        if wf_data.empty:

            wf = None

        else:

            wf = wf_data.iloc[0]

        frozen = FROZEN_CONFIGURATIONS.get(
            strategy,
            {},
        )

        rows.append(
            {
                "strategy": (decision["name"]),
                "status": (decision["status"]),
                "timeframe": (frozen.get("timeframe")),
                "rebalance": (frozen.get("rebalance")),
                # DEVELOPMENT
                "development_assets": (
                    development["symbol"].nunique() if not development.empty else 0
                ),
                "development_avg_return_pct": (
                    mean_value(
                        development,
                        "net_return_pct",
                    )
                ),
                "development_trades": (
                    total_value(
                        development,
                        "trades",
                    )
                ),
                "development_pf": (calculate_aggregate_pf(development)),
                "development_avg_dd_pct": (
                    mean_value(
                        development,
                        "max_drawdown_pct",
                    )
                ),
                # VALIDATION
                "validation_assets": (
                    validation["symbol"].nunique() if not validation.empty else 0
                ),
                "validation_avg_return_pct": (
                    mean_value(
                        validation,
                        "net_return_pct",
                    )
                ),
                "validation_trades": (
                    total_value(
                        validation,
                        "trades",
                    )
                ),
                "validation_pf": (calculate_aggregate_pf(validation)),
                "validation_avg_dd_pct": (
                    mean_value(
                        validation,
                        "max_drawdown_pct",
                    )
                ),
                # WALK-FORWARD
                "walk_forward_folds": (int(wf["folds"]) if wf is not None else 0),
                "walk_forward_profitable": (
                    int(wf["profitable_folds"]) if wf is not None else 0
                ),
                "walk_forward_losing": (
                    int(wf["losing_folds"]) if wf is not None else 0
                ),
                "walk_forward_flat": (int(wf["flat_folds"]) if wf is not None else 0),
                "walk_forward_pf_above_one": (
                    int(wf["pf_above_one_folds"]) if wf is not None else 0
                ),
                "walk_forward_trades": (int(wf["trades"]) if wf is not None else 0),
                "walk_forward_avg_return_pct": (
                    float(wf["average_fold_return_pct"])
                    if wf is not None
                    else float("nan")
                ),
                "walk_forward_median_pct": (
                    float(wf["median_fold_return_pct"])
                    if wf is not None
                    else float("nan")
                ),
                "walk_forward_worst_pct": (
                    float(wf["worst_fold_pct"]) if wf is not None else float("nan")
                ),
                "walk_forward_best_pct": (
                    float(wf["best_fold_pct"]) if wf is not None else float("nan")
                ),
                "walk_forward_avg_dd_pct": (
                    float(wf["average_max_drawdown_pct"])
                    if wf is not None
                    else float("nan")
                ),
                "walk_forward_assets": (int(wf["assets"]) if wf is not None else 0),
                "walk_forward_positive_assets": (
                    int(wf["positive_assets"]) if wf is not None else 0
                ),
                "walk_forward_negative_assets": (
                    int(wf["negative_assets"]) if wf is not None else 0
                ),
                "walk_forward_avg_continuous_pct": (
                    float(wf["average_continuous_return_pct"])
                    if wf is not None
                    else float("nan")
                ),
                "walk_forward_continuous_returns": (
                    str(wf["continuous_returns"]) if wf is not None else "N/A"
                ),
                # HOLDOUT
                "holdout_assets": (
                    holdout["symbol"].nunique() if not holdout.empty else 0
                ),
                "holdout_avg_return_pct": (
                    mean_value(
                        holdout,
                        "net_return_pct",
                    )
                ),
                "holdout_trades": (
                    total_value(
                        holdout,
                        "trades",
                    )
                ),
                "holdout_pf": (calculate_aggregate_pf(holdout)),
                "holdout_avg_dd_pct": (
                    mean_value(
                        holdout,
                        "max_drawdown_pct",
                    )
                ),
                "reason": (decision["reason"]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FORMATTING
# ============================================================


def format_number(
    value,
    suffix: str = "",
) -> str:

    if pd.isna(value):
        return "N/A"

    if value == float("inf"):
        return "INF"

    return f"{value:.2f}" f"{suffix}"


# ============================================================
# PRINT
# ============================================================


def print_strategy(
    row: pd.Series,
):

    print("-" * 118)

    print(row["strategy"])

    print("Status:        " f"{row['status']}")

    configuration = str(row["timeframe"])

    if pd.notna(row["rebalance"]):

        configuration += " / " + str(row["rebalance"])

    print("Configuration: " f"{configuration}")

    print()

    print(
        "Development:   "
        f"{format_number(row['development_avg_return_pct'], '%')} "
        f"| trades={int(row['development_trades'])} "
        f"| assets={int(row['development_assets'])} "
        f"| PF={format_number(row['development_pf'])} "
        f"| DD={format_number(row['development_avg_dd_pct'], '%')}"
    )

    print(
        "Validation:    "
        f"{format_number(row['validation_avg_return_pct'], '%')} "
        f"| trades={int(row['validation_trades'])} "
        f"| assets={int(row['validation_assets'])} "
        f"| PF={format_number(row['validation_pf'])} "
        f"| DD={format_number(row['validation_avg_dd_pct'], '%')}"
    )

    if row["walk_forward_folds"] > 0:

        print(
            "Walk-forward: "
            f"{int(row['walk_forward_profitable'])}"
            "/"
            f"{int(row['walk_forward_folds'])}"
            " profitable "
            f"| losing={int(row['walk_forward_losing'])} "
            f"| flat={int(row['walk_forward_flat'])} "
            f"| PF>1={int(row['walk_forward_pf_above_one'])} "
            f"| trades={int(row['walk_forward_trades'])}"
        )

        print(
            "WF returns:   "
            f"avg={format_number(row['walk_forward_avg_return_pct'], '%')} "
            f"| median={format_number(row['walk_forward_median_pct'], '%')} "
            f"| worst={format_number(row['walk_forward_worst_pct'], '%')} "
            f"| best={format_number(row['walk_forward_best_pct'], '%')} "
            f"| avg DD={format_number(row['walk_forward_avg_dd_pct'], '%')}"
        )

        print(
            "WF assets:    "
            f"{int(row['walk_forward_positive_assets'])}"
            "/"
            f"{int(row['walk_forward_assets'])}"
            " positive "
            f"| avg continuous="
            f"{format_number(row['walk_forward_avg_continuous_pct'], '%')}"
        )

        print("WF continuous:")

        print("  " f"{row['walk_forward_continuous_returns']}")

    else:

        print("Walk-forward: N/A")

    print(
        "Holdout:       "
        f"{format_number(row['holdout_avg_return_pct'], '%')} "
        f"| trades={int(row['holdout_trades'])} "
        f"| assets={int(row['holdout_assets'])} "
        f"| PF={format_number(row['holdout_pf'])} "
        f"| DD={format_number(row['holdout_avg_dd_pct'], '%')}"
    )

    print()

    print("Reason: " f"{row['reason']}")


# ============================================================
# MAIN
# ============================================================


def main():

    output_directory = Path("data/research")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments = collect_experiments()

    walk_forward = collect_walk_forward()

    summary = build_strategy_summary(
        experiments,
        walk_forward,
    )

    experiments_file = output_directory / "all_experiments.csv"

    walk_forward_file = output_directory / "walk_forward_summary.csv"

    summary_file = output_directory / "strategy_summary.csv"

    experiments.to_csv(
        experiments_file,
        index=False,
    )

    walk_forward.to_csv(
        walk_forward_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 118)

    print("AI TRADING SYSTEM — RESEARCH AUDIT")

    print("=" * 118)

    print()

    print(f"Experiments found: " f"{len(experiments)}")

    print("Strategy-level comparison uses " "FROZEN CONFIGURATIONS only.")

    print("TSMOM V2 walk-forward source: " "CONTINUOUS walk-forward.")

    print()

    for _, row in summary.iterrows():

        print_strategy(row)

    print()

    print("=" * 118)

    print("OUTPUT FILES")

    print("=" * 118)

    print()

    print("All experiments: " f"{experiments_file}")

    print("Walk-forward:    " f"{walk_forward_file}")

    print("Strategy summary:" f" {summary_file}")


if __name__ == "__main__":
    main()
