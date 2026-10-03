from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001

EPSILON = 1e-12


# ============================================================
# VALIDATION
# ============================================================


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "target_weight",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = dataframe.copy().reset_index(drop=True)

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    result = result.sort_values("timestamp").reset_index(drop=True)

    if result["timestamp"].duplicated().any():

        raise ValueError("Duplicate timestamps detected.")

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in price_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

        if (result[column] <= 0).any():

            raise ValueError(f"{column} contains " "non-positive values.")

    result["target_weight"] = pd.to_numeric(
        result["target_weight"],
        errors="coerce",
    )

    invalid_weight = result["target_weight"].notna() & (
        (result["target_weight"] < 0) | (result["target_weight"] > 1)
    )

    if invalid_weight.any():

        bad_rows = result.loc[
            invalid_weight,
            [
                "timestamp",
                "target_weight",
            ],
        ]

        raise ValueError("target_weight must be " "between 0 and 1.\n" f"{bad_rows}")

    return result


# ============================================================
# EXACT REBALANCING WITH FEES
# ============================================================


def calculate_rebalance(
    cash: float,
    quantity: float,
    price: float,
    target_weight: float,
    fee_rate: float,
) -> dict:

    asset_value = quantity * price

    equity_before = cash + asset_value

    if equity_before <= 0:

        raise RuntimeError("Portfolio equity is not positive.")

    current_weight = asset_value / equity_before

    difference = target_weight - current_weight

    # --------------------------------------------------------
    # Already at target.
    # --------------------------------------------------------

    if abs(difference) <= EPSILON:

        return {
            "side": None,
            "notional": 0.0,
            "fee": 0.0,
            "cash": cash,
            "quantity": quantity,
            "equity_before": equity_before,
            "equity_after": equity_before,
            "weight_before": current_weight,
            "weight_after": current_weight,
        }

    # --------------------------------------------------------
    # BUY
    #
    # We solve:
    #
    # new_asset_value
    # ---------------- = target_weight
    # equity_after_fee
    #
    # exactly, rather than approximately.
    # --------------------------------------------------------

    if target_weight > current_weight:

        numerator = target_weight * equity_before - asset_value

        denominator = 1 + target_weight * fee_rate

        notional = numerator / denominator

        notional = max(
            0.0,
            notional,
        )

        fee = notional * fee_rate

        new_cash = cash - notional - fee

        new_quantity = quantity + (notional / price)

        side = "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    else:

        numerator = asset_value - target_weight * equity_before

        denominator = 1 - target_weight * fee_rate

        notional = numerator / denominator

        notional = min(
            asset_value,
            max(
                0.0,
                notional,
            ),
        )

        fee = notional * fee_rate

        new_cash = cash + notional - fee

        new_quantity = quantity - (notional / price)

        if abs(new_quantity) <= EPSILON:

            new_quantity = 0.0

        side = "SELL"

    # --------------------------------------------------------
    # Post-trade state
    # --------------------------------------------------------

    new_asset_value = new_quantity * price

    equity_after = new_cash + new_asset_value

    if equity_after > 0:

        weight_after = new_asset_value / equity_after

    else:

        weight_after = 0.0

    return {
        "side": side,
        "notional": notional,
        "fee": fee,
        "cash": new_cash,
        "quantity": new_quantity,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "weight_before": current_weight,
        "weight_after": weight_after,
    }


# ============================================================
# BACKTEST
# ============================================================


def run_fractional_backtest(
    dataframe: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
    force_liquidation: bool = True,
):

    data = validate_dataframe(dataframe)

    if initial_capital <= 0:

        raise ValueError("initial_capital must be > 0.")

    if fee_rate < 0:

        raise ValueError("fee_rate must be >= 0.")

    cash = float(initial_capital)

    quantity = 0.0

    pending_target = None
    pending_signal_timestamp = None

    transactions = []
    equity_rows = []

    total_fees = 0.0

    # ========================================================
    # CANDLE LOOP
    # ========================================================

    for index, row in data.iterrows():

        timestamp = row["timestamp"]

        open_price = float(row["open"])

        close_price = float(row["close"])

        # ----------------------------------------------------
        # Execute yesterday's target at today's OPEN.
        # ----------------------------------------------------

        if pending_target is not None:

            rebalance = calculate_rebalance(
                cash=cash,
                quantity=quantity,
                price=open_price,
                target_weight=(pending_target),
                fee_rate=fee_rate,
            )

            cash = rebalance["cash"]

            quantity = rebalance["quantity"]

            total_fees += rebalance["fee"]

            if rebalance["side"] is not None and rebalance["notional"] > EPSILON:

                transactions.append(
                    {
                        "signal_timestamp": (pending_signal_timestamp),
                        "execution_timestamp": (timestamp),
                        "side": (rebalance["side"]),
                        "execution_price": (open_price),
                        "target_weight": (pending_target),
                        "weight_before": (rebalance["weight_before"]),
                        "weight_after": (rebalance["weight_after"]),
                        "notional": (rebalance["notional"]),
                        "fee": (rebalance["fee"]),
                        "cash_after": (cash),
                        "quantity_after": (quantity),
                        "forced": False,
                    }
                )

            pending_target = None
            pending_signal_timestamp = None

        # ----------------------------------------------------
        # Mark to market at current CLOSE.
        # ----------------------------------------------------

        asset_value_close = quantity * close_price

        equity_close = cash + asset_value_close

        if equity_close > 0:

            actual_weight_close = asset_value_close / equity_close

        else:

            actual_weight_close = 0.0

        equity_rows.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "quantity": quantity,
                "asset_value": (asset_value_close),
                "equity": (equity_close),
                "actual_weight": (actual_weight_close),
            }
        )

        # ----------------------------------------------------
        # Today's decision is only executable tomorrow.
        #
        # A target on the final candle is therefore ignored.
        # ----------------------------------------------------

        target = row["target_weight"]

        if index < len(data) - 1 and pd.notna(target):

            pending_target = float(target)

            pending_signal_timestamp = timestamp

    # ========================================================
    # FORCE FINAL LIQUIDATION
    #
    # Same research convention as the original backtester:
    # value any remaining open position as if liquidated at
    # the final candle CLOSE, including the exit fee.
    # ========================================================

    forced_exits = 0

    if force_liquidation and quantity > EPSILON:

        last_row = data.iloc[-1]

        timestamp = last_row["timestamp"]

        close_price = float(last_row["close"])

        asset_value = quantity * close_price

        fee = asset_value * fee_rate

        cash = cash + asset_value - fee

        total_fees += fee

        weight_before = 1.0

        equity_before = cash + fee

        if equity_before > 0:

            weight_before = asset_value / equity_before

        transactions.append(
            {
                "signal_timestamp": (timestamp),
                "execution_timestamp": (timestamp),
                "side": "SELL",
                "execution_price": (close_price),
                "target_weight": 0.0,
                "weight_before": (weight_before),
                "weight_after": 0.0,
                "notional": (asset_value),
                "fee": (fee),
                "cash_after": (cash),
                "quantity_after": 0.0,
                "forced": True,
            }
        )

        quantity = 0.0
        forced_exits = 1

        # Update final equity row after liquidation.
        equity_rows[-1]["cash"] = cash

        equity_rows[-1]["quantity"] = 0.0

        equity_rows[-1]["asset_value"] = 0.0

        equity_rows[-1]["equity"] = cash

        equity_rows[-1]["actual_weight"] = 0.0

    # ========================================================
    # RESULT TABLES
    # ========================================================

    transactions_df = pd.DataFrame(transactions)

    equity_df = pd.DataFrame(equity_rows)

    # ========================================================
    # DRAWDOWN
    # ========================================================

    running_peak = equity_df["equity"].cummax()

    drawdown = ((equity_df["equity"] / running_peak) - 1) * 100

    equity_df["drawdown_pct"] = drawdown

    max_drawdown = float(drawdown.min())

    # ========================================================
    # PERFORMANCE
    # ========================================================

    final_capital = float(equity_df.iloc[-1]["equity"])

    total_return = ((final_capital / initial_capital) - 1) * 100

    average_exposure = float(equity_df["actual_weight"].mean() * 100)

    invested_candles = float((equity_df["actual_weight"] > EPSILON).mean() * 100)

    if transactions_df.empty:

        rebalance_count = 0
        total_notional = 0.0

    else:

        rebalance_count = int((transactions_df["forced"] == False).sum())

        total_notional = float(transactions_df["notional"].sum())

    stats = {
        "initial_capital": (float(initial_capital)),
        "final_capital": (final_capital),
        "total_return": (total_return),
        "max_drawdown": (max_drawdown),
        "average_exposure_pct": (average_exposure),
        "invested_candles_pct": (invested_candles),
        "rebalance_count": (rebalance_count),
        "total_fees": (float(total_fees)),
        "total_notional": (total_notional),
        "forced_exits": (forced_exits),
    }

    return (
        transactions_df,
        equity_df,
        stats,
    )
