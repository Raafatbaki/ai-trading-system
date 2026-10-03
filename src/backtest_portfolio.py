from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001

EPSILON = 1e-12


# ============================================================
# PRICE DATA VALIDATION
# ============================================================


def prepare_price_data(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:

    if not price_data:
        raise ValueError("price_data is empty.")

    prepared = {}

    for symbol, dataframe in price_data.items():

        required = [
            "timestamp",
            "open",
            "close",
        ]

        missing = [column for column in required if column not in dataframe.columns]

        if missing:
            raise ValueError(f"{symbol}: missing columns " f"{missing}")

        result = dataframe.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
        )

        result = result.sort_values("timestamp").reset_index(drop=True)

        if result["timestamp"].duplicated().any():

            raise ValueError(f"{symbol}: duplicate timestamps.")

        for column in [
            "open",
            "close",
        ]:

            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            )

            if (result[column] <= 0).any():

                raise ValueError(f"{symbol}: invalid {column}.")

        result = result.set_index("timestamp")

        prepared[symbol] = result

    return prepared


# ============================================================
# WEIGHT VALIDATION
# ============================================================


def prepare_weights(
    weights: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:

    required = [
        "timestamp",
    ] + [f"{symbol}_weight" for symbol in symbols]

    missing = [column for column in required if column not in weights.columns]

    if missing:
        raise ValueError("Weight table missing columns: " f"{missing}")

    result = weights[required].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    result = result.sort_values("timestamp").reset_index(drop=True)

    if result["timestamp"].duplicated().any():

        raise ValueError("Duplicate portfolio decision " "timestamps.")

    weight_columns = [f"{symbol}_weight" for symbol in symbols]

    for column in weight_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

        if (result[column] < -EPSILON).any():

            raise ValueError(f"{column}: negative weight.")

        if (result[column] > 1.0 + EPSILON).any():

            raise ValueError(f"{column}: weight above 1.")

    total_weight = result[weight_columns].sum(axis=1)

    if (total_weight > 1.0 + EPSILON).any():

        raise ValueError("Portfolio leverage detected: " "weights sum above 100%.")

    return result.set_index("timestamp")


# ============================================================
# EXACT MULTI-ASSET REBALANCE
# ============================================================


def calculate_exact_rebalance(
    cash: float,
    quantities: dict[str, float],
    prices: dict[str, float | None],
    target_weights: dict[str, float],
    fee_rate: float,
) -> dict:

    symbols = list(quantities.keys())

    current_values = {}

    equity_before = float(cash)

    # --------------------------------------------------------
    # Current portfolio value
    # --------------------------------------------------------

    for symbol in symbols:

        quantity = float(quantities[symbol])

        price = prices.get(symbol)

        if quantity > EPSILON:

            if price is None or pd.isna(price) or price <= 0:

                raise RuntimeError(
                    f"{symbol}: missing execution " "price for held asset."
                )

            value = quantity * float(price)

        else:

            value = 0.0

        current_values[symbol] = value

        equity_before += value

    if equity_before <= 0:

        raise RuntimeError("Portfolio equity is not positive.")

    total_target_weight = sum(target_weights.values())

    if total_target_weight > (1.0 + EPSILON):

        raise ValueError("Target portfolio uses leverage.")

    for symbol in symbols:

        target = float(target_weights[symbol])

        if target < -EPSILON or target > 1.0 + EPSILON:

            raise ValueError(f"{symbol}: invalid target weight " f"{target}")

        if target > EPSILON:

            price = prices.get(symbol)

            if price is None or pd.isna(price) or price <= 0:

                raise RuntimeError(
                    f"{symbol}: missing execution " "price for target position."
                )

    # ========================================================
    # EQUITY AFTER FEES
    #
    # Find E_after satisfying:
    #
    # E_after
    # + fee_rate * sum(
    #     abs(target_i * E_after - current_i)
    #   )
    # = E_before
    #
    # This allows exact final portfolio weights even after
    # transaction fees.
    # ========================================================

    if fee_rate == 0:

        equity_after = equity_before

    else:

        def equation(
            candidate_equity: float,
        ) -> float:

            traded_notional = 0.0

            for symbol in symbols:

                desired_value = target_weights[symbol] * candidate_equity

                traded_notional += abs(desired_value - current_values[symbol])

            return candidate_equity + fee_rate * traded_notional - equity_before

        low = 0.0
        high = equity_before

        for _ in range(100):

            middle = (low + high) / 2.0

            if equation(middle) > 0:

                high = middle

            else:

                low = middle

        equity_after = (low + high) / 2.0

    # ========================================================
    # BUILD FINAL POSITIONS
    # ========================================================

    new_quantities = {}

    transactions = []

    total_fees = 0.0
    total_notional = 0.0

    for symbol in symbols:

        target_weight = float(target_weights[symbol])

        desired_value = target_weight * equity_after

        current_value = current_values[symbol]

        trade_value = desired_value - current_value

        notional = abs(trade_value)

        fee = notional * fee_rate

        total_fees += fee
        total_notional += notional

        if desired_value > EPSILON:

            price = float(prices[symbol])

            new_quantity = desired_value / price

        else:

            new_quantity = 0.0

        new_quantities[symbol] = new_quantity

        if notional > EPSILON:

            if trade_value > 0:
                side = "BUY"
            else:
                side = "SELL"

            transactions.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "notional": (notional),
                    "fee": fee,
                    "target_weight": (target_weight),
                    "value_before": (current_value),
                    "value_after": (desired_value),
                }
            )

    # Cash is whatever remains after applying all
    # target asset weights to post-fee equity.

    new_cash = equity_after * (1.0 - total_target_weight)

    # Numerical consistency check.
    reconstructed = new_cash + sum(
        new_quantities[symbol]
        * (float(prices[symbol]) if new_quantities[symbol] > EPSILON else 0.0)
        for symbol in symbols
    )

    if not np.isclose(
        reconstructed,
        equity_after,
        rtol=1e-9,
        atol=1e-9,
    ):

        raise RuntimeError("Portfolio rebalance " "consistency check failed.")

    return {
        "cash": (new_cash),
        "quantities": (new_quantities),
        "transactions": (transactions),
        "equity_before": (equity_before),
        "equity_after": (equity_after),
        "total_fees": (total_fees),
        "total_notional": (total_notional),
    }


# ============================================================
# GET PRICE
# ============================================================


def get_price(
    prepared_prices,
    symbol,
    timestamp,
    column,
):

    dataframe = prepared_prices[symbol]

    if timestamp not in (dataframe.index):

        return None

    value = dataframe.at[
        timestamp,
        column,
    ]

    if pd.isna(value):

        return None

    return float(value)


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================


def run_portfolio_backtest(
    price_data: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
    force_liquidation: bool = True,
):

    if initial_capital <= 0:

        raise ValueError("initial_capital must be > 0.")

    if fee_rate < 0:

        raise ValueError("fee_rate must be >= 0.")

    prepared_prices = prepare_price_data(price_data)

    symbols = list(prepared_prices.keys())

    prepared_weights = prepare_weights(
        weights,
        symbols,
    )

    # ========================================================
    # MASTER CALENDAR
    # ========================================================

    all_timestamps = set()

    for dataframe in prepared_prices.values():

        all_timestamps.update(dataframe.index.tolist())

    calendar = sorted(all_timestamps)

    if not calendar:

        raise RuntimeError("No price timestamps.")

    first_decision = prepared_weights.index.min()

    calendar = [timestamp for timestamp in calendar if timestamp >= first_decision]

    if not calendar:

        raise RuntimeError("No market data after " "first portfolio decision.")

    # ========================================================
    # INITIAL STATE
    # ========================================================

    cash = float(initial_capital)

    quantities = {symbol: 0.0 for symbol in symbols}

    pending_weights = None
    pending_signal_timestamp = None

    transactions = []
    equity_rows = []

    total_fees = 0.0
    total_notional = 0.0

    rebalance_events = 0

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for calendar_index, timestamp in enumerate(calendar):

        # ----------------------------------------------------
        # Execute previous decision at current OPEN.
        # ----------------------------------------------------

        if pending_weights is not None:

            open_prices = {
                symbol: (
                    get_price(
                        prepared_prices,
                        symbol,
                        timestamp,
                        "open",
                    )
                )
                for symbol in symbols
            }

            result = calculate_exact_rebalance(
                cash=cash,
                quantities=quantities,
                prices=open_prices,
                target_weights=(pending_weights),
                fee_rate=fee_rate,
            )

            cash = float(result["cash"])

            quantities = result["quantities"]

            total_fees += float(result["total_fees"])

            total_notional += float(result["total_notional"])

            if result["transactions"]:

                rebalance_events += 1

            for transaction in result["transactions"]:

                transactions.append(
                    {
                        "signal_timestamp": (pending_signal_timestamp),
                        "execution_timestamp": (timestamp),
                        **transaction,
                        "forced": False,
                    }
                )

            pending_weights = None
            pending_signal_timestamp = None

        # ----------------------------------------------------
        # MARK TO MARKET AT CLOSE
        # ----------------------------------------------------

        asset_values = {}

        total_asset_value = 0.0

        for symbol in symbols:

            quantity = float(quantities[symbol])

            if quantity <= EPSILON:

                asset_value = 0.0

            else:

                close_price = get_price(
                    prepared_prices,
                    symbol,
                    timestamp,
                    "close",
                )

                if close_price is None:

                    raise RuntimeError(
                        f"{symbol}: missing close "
                        f"for held position on "
                        f"{timestamp}"
                    )

                asset_value = quantity * close_price

            asset_values[symbol] = asset_value

            total_asset_value += asset_value

        equity = cash + total_asset_value

        if equity <= 0:

            raise RuntimeError("Portfolio equity became " "non-positive.")

        portfolio_exposure = total_asset_value / equity

        row = {
            "timestamp": (timestamp),
            "cash": (cash),
            "asset_value": (total_asset_value),
            "equity": (equity),
            "portfolio_exposure": (portfolio_exposure),
        }

        for symbol in symbols:

            value = asset_values[symbol]

            row[f"{symbol}_value"] = value

            row[f"{symbol}_weight"] = value / equity

        equity_rows.append(row)

        # ----------------------------------------------------
        # READ TODAY'S TARGET
        #
        # Execution happens on NEXT market day.
        # ----------------------------------------------------

        if timestamp in prepared_weights.index and calendar_index < len(calendar) - 1:

            decision = prepared_weights.loc[timestamp]

            pending_weights = {
                symbol: float(decision[f"{symbol}_weight"]) for symbol in symbols
            }

            pending_signal_timestamp = timestamp

    # ========================================================
    # FINAL LIQUIDATION
    # ========================================================

    forced_exits = 0

    if force_liquidation:

        has_position = any(quantity > EPSILON for quantity in quantities.values())

        if has_position:

            final_timestamp = calendar[-1]

            final_prices = {
                symbol: (
                    get_price(
                        prepared_prices,
                        symbol,
                        final_timestamp,
                        "close",
                    )
                )
                for symbol in symbols
            }

            zero_weights = {symbol: 0.0 for symbol in symbols}

            result = calculate_exact_rebalance(
                cash=cash,
                quantities=quantities,
                prices=final_prices,
                target_weights=(zero_weights),
                fee_rate=fee_rate,
            )

            cash = float(result["cash"])

            quantities = result["quantities"]

            total_fees += float(result["total_fees"])

            total_notional += float(result["total_notional"])

            for transaction in result["transactions"]:

                transactions.append(
                    {
                        "signal_timestamp": (final_timestamp),
                        "execution_timestamp": (final_timestamp),
                        **transaction,
                        "forced": True,
                    }
                )

            forced_exits = 1

            # Update final daily equity row.
            equity_rows[-1]["cash"] = cash

            equity_rows[-1]["asset_value"] = 0.0

            equity_rows[-1]["equity"] = cash

            equity_rows[-1]["portfolio_exposure"] = 0.0

            for symbol in symbols:

                equity_rows[-1][f"{symbol}_value"] = 0.0

                equity_rows[-1][f"{symbol}_weight"] = 0.0

    # ========================================================
    # DATAFRAMES
    # ========================================================

    transactions_df = pd.DataFrame(transactions)

    equity_df = pd.DataFrame(equity_rows)

    # ========================================================
    # DRAWDOWN
    # ========================================================

    running_peak = equity_df["equity"].cummax()

    equity_df["drawdown_pct"] = ((equity_df["equity"] / running_peak) - 1.0) * 100.0

    max_drawdown = float(equity_df["drawdown_pct"].min())

    # ========================================================
    # STATS
    # ========================================================

    final_capital = float(equity_df.iloc[-1]["equity"])

    total_return = ((final_capital / initial_capital) - 1.0) * 100.0

    average_exposure = float(equity_df["portfolio_exposure"].mean() * 100.0)

    invested_days = float((equity_df["portfolio_exposure"] > EPSILON).mean() * 100.0)

    if transactions_df.empty:

        transaction_count = 0

    else:

        transaction_count = int((transactions_df["forced"] == False).sum())

    stats = {
        "initial_capital": (float(initial_capital)),
        "final_capital": (final_capital),
        "total_return": (total_return),
        "max_drawdown": (max_drawdown),
        "average_exposure_pct": (average_exposure),
        "invested_days_pct": (invested_days),
        "rebalance_events": (rebalance_events),
        "transaction_count": (transaction_count),
        "total_fees": (float(total_fees)),
        "total_notional": (float(total_notional)),
        "forced_exits": (forced_exits),
    }

    return (
        transactions_df,
        equity_df,
        stats,
    )
