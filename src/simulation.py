import math
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from fill_model import check_fill
from config import ORDER_SIZE, PARTICIPATION_RATE, FILL_RULE


def _is_finite_number(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _get_queue_ahead(row: dict, quote_price: Optional[float], side: str, tick_size: float) -> float:
    if not _is_finite_number(quote_price):
        return 0.0

    quote_price = float(quote_price)
    tolerance = tick_size / 2

    for level in range(10):
        level_price = row.get(f"{side}_px_{level:02d}")
        level_size = row.get(f"{side}_sz_{level:02d}")

        if not _is_finite_number(level_price) or not _is_finite_number(level_size):
            continue

        if abs(float(level_price) - quote_price) <= tolerance:
            return float(level_size)

    return 0.0


def enforce_passive_quote(
    bid: Optional[float],
    ask: Optional[float],
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick_size: float,
) -> tuple[Optional[float], Optional[float], bool, bool]:
    bid_clamped, ask_clamped = False, False

    book_is_valid = (
        _is_finite_number(best_bid)
        and _is_finite_number(best_ask)
        and float(best_bid) < float(best_ask)
    )

    if not book_is_valid:
        return bid, ask, bid_clamped, ask_clamped

    max_bid = round(float(best_ask) - tick_size, 10)
    min_ask = round(float(best_bid) + tick_size, 10)

    if bid is not None and float(bid) > max_bid:
        bid, bid_clamped = max_bid, True

    if ask is not None and float(ask) < min_ask:
        ask, ask_clamped = min_ask, True

    return bid, ask, bid_clamped, ask_clamped


def liquidate_position(inventory: float, book_row: dict) -> tuple[float, bool]:
    if inventory == 0:
        return 0.0, False

    is_long = inventory > 0
    side_prefix = "bid" if is_long else "ask"
    remaining = abs(float(inventory))
    cash_delta = 0.0
    worst_price = None

    for level in range(10):
        price = book_row.get(f"{side_prefix}_px_{level:02d}")
        size = book_row.get(f"{side_prefix}_sz_{level:02d}")

        if not _is_finite_number(price) or not _is_finite_number(size) or float(size) <= 0:
            continue

        price = float(price)
        worst_price = price
        take = min(remaining, float(size))

        cash_delta += take * price if is_long else -take * price
        remaining -= take

        if remaining <= 1e-9:
            remaining = 0.0
            break

    exhausted = remaining > 0
    if exhausted:
        if worst_price is None:
            raise ValueError(
                "Liquidation nicht möglich: keine sichtbare Tiefe auf der "
                f"{side_prefix}-Seite im letzten Bin des Tages."
            )
        cash_delta += remaining * worst_price if is_long else -remaining * worst_price

    return cash_delta, exhausted


def run_simulation(
    df: pd.DataFrame,
    strategy,
    order_size: float = ORDER_SIZE,
    participation_rate: float = PARTICIPATION_RATE,
    quote_horizon_ms: int = 500,
    fill_rule: str = FILL_RULE,
) -> tuple[dict, pd.DataFrame]:
    if df.empty:
        return {}, pd.DataFrame()

    if not hasattr(strategy, "get_quote"):
        raise TypeError("strategy muss eine get_quote()-Methode besitzen.")
    if not hasattr(strategy, "name"):
        raise TypeError("strategy muss ein name-Attribut besitzen.")
    if not hasattr(strategy, "tick_size"):
        raise TypeError("strategy muss ein tick_size-Attribut besitzen.")

    tick_size = float(strategy.tick_size)

    if order_size <= 0:
        raise ValueError("order_size muss positiv sein.")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate muss größer als 0 und höchstens 1 sein.")
    if quote_horizon_ms <= 0:
        raise ValueError("quote_horizon_ms muss positiv sein.")
    if tick_size <= 0:
        raise ValueError("tick_size muss positiv sein.")

    required_columns = {
        "ts_recv", "trading_day", "mid", "spread", "imbalance_l1",
        "sigma_ewma_usd", "time_to_close",
        "bid_px_00", "ask_px_00", "sell_trade_price_min",
        "buy_trade_price_max", "sell_trade_volume", "buy_trade_volume",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Folgende benötigte Spalten fehlen im Feature-Datensatz: {missing}")

    simulation_df = df.copy()
    simulation_df["ts_recv"] = pd.to_datetime(simulation_df["ts_recv"])
    simulation_df = simulation_df.sort_values(["trading_day", "ts_recv"]).reset_index(drop=True)

    duplicate_count = int(simulation_df["ts_recv"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Der Feature-Datensatz enthält {duplicate_count:,} doppelte Timestamps.")

    expected_step = pd.Timedelta(milliseconds=quote_horizon_ms)

    results_by_day = {}
    summary_rows = []

    for trading_day, day_df in simulation_df.groupby("trading_day", sort=True):
        day_df = day_df.reset_index(drop=True)
        cash, inventory = 0.0, 0.0
        rows = []
        records = day_df.to_dict("records")

        for i in range(len(records) - 1):
            row, next_row = records[i], records[i + 1]

            current_time = pd.Timestamp(row["ts_recv"])
            next_time = pd.Timestamp(next_row["ts_recv"])

            time_gap = next_time - current_time
            if time_gap <= pd.Timedelta(0) or time_gap > expected_step:
                continue

            decision_mid = float(row["mid"])
            mark_price = float(next_row["mid"])

            if pd.isna(decision_mid) or pd.isna(mark_price):
                continue

            best_bid, best_ask = row.get("bid_px_00"), row.get("ask_px_00")
            if not _is_finite_number(best_bid) or not _is_finite_number(best_ask):
                continue

            inventory_before = inventory

            quote = strategy.get_quote(
                mid=decision_mid,
                spread=row["spread"],
                imbalance_l1=row["imbalance_l1"],
                sigma_ewma_usd=row.get("sigma_ewma_usd"),
                time_to_close=row.get("time_to_close"),
                inventory=inventory,
            )

            bid_quote, ask_quote, bid_clamped, ask_clamped = enforce_passive_quote(
                quote.bid, quote.ask, best_bid, best_ask, tick_size
            )

            queue_ahead_bid = _get_queue_ahead(row, bid_quote, "bid", tick_size)
            queue_ahead_ask = _get_queue_ahead(row, ask_quote, "ask", tick_size)

            fills = check_fill(
                bid_quote=bid_quote,
                ask_quote=ask_quote,
                best_bid=best_bid,
                best_ask=best_ask,
                next_row=next_row,
                queue_ahead_bid=queue_ahead_bid,
                queue_ahead_ask=queue_ahead_ask,
                order_size=order_size,
                participation_rate=participation_rate,
                fill_rule=fill_rule,
                tick_size=tick_size,
            )

            buy_fill_size, sell_fill_size = 0.0, 0.0

            for fill in fills:
                if fill.size <= 0:
                    continue

                if fill.side == "buy":
                    cash -= fill.price * fill.size
                    inventory += fill.size
                    buy_fill_size += fill.size
                elif fill.side == "sell":
                    cash += fill.price * fill.size
                    inventory -= fill.size
                    sell_fill_size += fill.size
                else:
                    raise ValueError(f"Unbekannte Fill-Seite: {fill.side}")

            total_fill_size = buy_fill_size + sell_fill_size
            pnl = cash + inventory * mark_price

            rows.append({
                "timestamp": next_time,
                "decision_timestamp": current_time,
                "strategy": strategy.name,
                "decision_mid": decision_mid,
                "mark_mid": mark_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "reference_price": getattr(quote, "reference_price", None),
                "inventory_penalty": getattr(quote, "inventory_penalty", None),
                "bid_quote": bid_quote,
                "ask_quote": ask_quote,
                "bid_quote_raw": quote.bid,
                "ask_quote_raw": quote.ask,
                "bid_clamped": bid_clamped,
                "ask_clamped": ask_clamped,
                "bid_active": bid_quote is not None,
                "ask_active": ask_quote is not None,
                "n_fills": len(fills),
                "buy_fill_size": buy_fill_size,
                "sell_fill_size": sell_fill_size,
                "fill_size": total_fill_size,
                "cash": cash,
                "inventory_before": inventory_before,
                "inventory": inventory,
                "pnl": pnl,
                "is_liquidation": False,
                "liquidation_cash": 0.0,
            })

        last_row = records[-1]
        inventory_before_liquidation = inventory
        pnl_before_liquidation = cash + inventory * float(last_row["mid"])

        liquidation_cash, exhausted = liquidate_position(inventory, last_row)

        if exhausted:
            print(
                f"  Liquidation Tag {trading_day}, Strategie {strategy.name}: "
                f"sichtbare Tiefe ueber 10 Level nicht ausreichend, Rest zum "
                f"schlechtesten sichtbaren Preis abgerechnet."
            )

        cash += liquidation_cash
        inventory = 0.0
        final_mid = float(last_row["mid"])
        pnl = cash

        rows.append({
            "timestamp": pd.Timestamp(last_row["ts_recv"]),
            "decision_timestamp": pd.Timestamp(last_row["ts_recv"]),
            "strategy": strategy.name,
            "decision_mid": final_mid,
            "mark_mid": final_mid,
            "best_bid": last_row.get("bid_px_00"),
            "best_ask": last_row.get("ask_px_00"),
            "reference_price": None,
            "inventory_penalty": None,
            "bid_quote": None,
            "ask_quote": None,
            "bid_quote_raw": None,
            "ask_quote_raw": None,
            "bid_clamped": False,
            "ask_clamped": False,
            "bid_active": False,
            "ask_active": False,
            "n_fills": 0,
            "buy_fill_size": 0.0,
            "sell_fill_size": 0.0,
            "fill_size": 0.0,
            "cash": cash,
            "inventory_before": inventory_before_liquidation,
            "inventory": inventory,
            "pnl": pnl,
            "is_liquidation": True,
            "liquidation_cash": liquidation_cash,
        })

        day_result = pd.DataFrame(rows)
        results_by_day[trading_day] = day_result

        summary_rows.append({
            "trading_day": trading_day,
            "strategy": strategy.name,
            "n_steps": len(day_result),
            "n_fills": int(day_result["n_fills"].sum()),
            "final_pnl": pnl,
            "liquidation_pnl": liquidation_cash,
            "pnl_before_liquidation": pnl_before_liquidation,
            "inventory_before_liquidation": inventory_before_liquidation,
            "liquidation_depth_exhausted": exhausted,
        })

    daily_summary = pd.DataFrame(summary_rows)

    return results_by_day, daily_summary


def report(results_by_day: dict, daily_summary: pd.DataFrame, show_plot: bool = True) -> None:
    if not results_by_day or daily_summary.empty:
        print("Keine Simulationsergebnisse vorhanden.")
        return

    all_results = pd.concat(results_by_day.values(), ignore_index=True)

    n_steps = len(all_results)
    active_quotes = int(all_results["bid_active"].sum() + all_results["ask_active"].sum())
    total_fills = int(all_results["n_fills"].sum())
    clamped_quotes = int(all_results["bid_clamped"].sum() + all_results["ask_clamped"].sum())
    total_volume = float(all_results["fill_size"].sum())

    print("=== Strategie-Report ===")
    print(f"Strategie:                  {daily_summary['strategy'].iloc[0]}")
    print(f"Handelstage:                {len(daily_summary):,}")
    print(f"Zeitschritte gesamt:        {n_steps:,}")
    print(f"Aktive Quotes:              {active_quotes:,}")
    print(f"Am Markt begrenzt:          {clamped_quotes:,}")

    if active_quotes > 0:
        print(f"Anteil begrenzt:            {clamped_quotes / active_quotes:.2%}")

    print(f"Fill-Ereignisse:            {total_fills:,}")
    print(f"Ausgeführtes Volumen:       {total_volume:,.0f}")

    if active_quotes > 0:
        print(f"Fill-Rate je aktiver Quote: {total_fills / active_quotes:.2%}")

    print(f"Summe Tages-P&L:            {daily_summary['final_pnl'].sum():,.2f}")
    print(f"Summe Liquidations-P&L:     {daily_summary['liquidation_pnl'].sum():,.2f}")
    print(
        f"Tage mit ausgeschöpfter Tiefe bei Liquidation: "
        f"{int(daily_summary['liquidation_depth_exhausted'].sum())}"
    )

    if not show_plot:
        return

    cumulative_pnl = daily_summary["final_pnl"].cumsum()

    _, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(daily_summary["trading_day"].astype(str), cumulative_pnl, marker="o")
    axes[0].set_title("Kumulierter Tages-P&L")
    axes[0].set_ylabel("P&L")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(daily_summary["trading_day"].astype(str), daily_summary["liquidation_pnl"])
    axes[1].set_title("Liquidationsertrag je Tag")
    axes[1].set_ylabel("P&L")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()