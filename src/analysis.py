from pathlib import Path

import pandas as pd

from metrics import summarize_daily
from strategies import NaiveStrategy
from simulation import run_simulation


DEFAULT_HALF_SPREADS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)


def select_days(feat: pd.DataFrame, days=None, n_days: int = None) -> pd.DataFrame:
    available = sorted(feat["trading_day"].dropna().astype(str).unique())

    if days is not None:
        selected = [str(day) for day in days]
    elif n_days is not None:
        selected = available[:n_days]
    else:
        raise ValueError("Entweder days oder n_days angeben.")

    unknown = set(selected).difference(available)
    if unknown:
        raise ValueError(f"Unbekannte Handelstage: {sorted(unknown)}")

    subset = feat.loc[feat["trading_day"].astype(str).isin(selected)]

    return subset.reset_index(drop=True)


def decompose_fills(results: pd.DataFrame, total_pnl: float) -> dict:
    buy_volume = results["buy_fill_size"].fillna(0.0)
    sell_volume = results["sell_fill_size"].fillna(0.0)

    quoted_buy = ((results["decision_mid"] - results["bid_quote"]) * buy_volume).fillna(0.0).sum()
    quoted_sell = ((results["ask_quote"] - results["decision_mid"]) * sell_volume).fillna(0.0).sum()
    quoted = quoted_buy + quoted_sell

    realized_buy = ((results["mark_mid"] - results["bid_quote"]) * buy_volume).fillna(0.0).sum()
    realized_sell = ((results["ask_quote"] - results["mark_mid"]) * sell_volume).fillna(0.0).sum()
    realized = realized_buy + realized_sell

    volume = float(buy_volume.sum() + sell_volume.sum())

    if volume <= 0:
        raise ValueError("Es wurden keine Fills ausgeführt.")

    return {
        "volume": volume,
        "quoted_total": float(quoted),
        "impact_total": float(realized - quoted),
        "realized_total": float(realized),
        "inventory_total": float(total_pnl) - float(realized),
        "final_pnl": float(total_pnl),
        "quoted_per_share": float(quoted) / volume,
        "impact_per_share": float(realized - quoted) / volume,
        "realized_per_share": float(realized) / volume,
    }


def compare_strategies(results_by_strategy: dict, out_path: Path = None) -> pd.DataFrame:
    rows = []

    for name, (results_by_day, daily_summary) in results_by_strategy.items():
        if not results_by_day or daily_summary.empty:
            continue

        results = pd.concat(results_by_day.values(), ignore_index=True)
        total_pnl = float(daily_summary["final_pnl"].sum())

        parts = decompose_fills(results, total_pnl)
        daily = summarize_daily(daily_summary)
        inventory = results["inventory"]

        rows.append({
            "strategy": name,
            "n_fills": int(results["n_fills"].sum()),
            "volume": parts["volume"],
            "quoted_ct": parts["quoted_per_share"] * 100,
            "impact_ct": parts["impact_per_share"] * 100,
            "realized_ct": parts["realized_per_share"] * 100,
            "realized_total": parts["realized_total"],
            "inventory_total": parts["inventory_total"],
            "final_pnl": parts["final_pnl"],
            "avg_abs_inv": float(inventory.abs().mean()),
            "inv_q99": float(inventory.abs().quantile(0.99)),
            "max_abs_inv": float(inventory.abs().max()),
            "final_inv": float(inventory.iloc[-1]),
            "sharpe_d": daily["sharpe_daily"],
            "sortino_d": daily["sortino_daily"],
            "var_95_d": daily["var_95_daily"],
            "es_95_d": daily["es_95_daily"],
            "max_dd_d": daily["max_drawdown_daily"],
        })

    table = pd.DataFrame(rows)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"Geschrieben nach {out_path}")

    return table


def sweep_half_spread(
    feat: pd.DataFrame,
    half_spreads=DEFAULT_HALF_SPREADS,
    order_size: float = 100.0,
    participation_rate: float = 0.05,
    quote_horizon_ms: int = 500,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    rows = []

    for half_spread in half_spreads:
        results_by_day, daily_summary = run_simulation(
            feat,
            NaiveStrategy(half_spread=half_spread, tick_size=tick_size),
            order_size=order_size,
            participation_rate=participation_rate,
            quote_horizon_ms=quote_horizon_ms,
        )

        if not results_by_day or daily_summary.empty:
            continue

        results = pd.concat(results_by_day.values(), ignore_index=True)

        if results["fill_size"].sum() <= 0:
            continue

        total_pnl = float(daily_summary["final_pnl"].sum())
        parts = decompose_fills(results, total_pnl)

        rows.append({
            "half_spread": half_spread,
            "n_fills": int(results["n_fills"].sum()),
            "volume": parts["volume"],
            "quoted_ct": parts["quoted_per_share"] * 100,
            "impact_ct": parts["impact_per_share"] * 100,
            "realized_ct": parts["realized_per_share"] * 100,
            "realized_total": parts["realized_total"],
            "inventory_total": parts["inventory_total"],
            "final_pnl": parts["final_pnl"],
            "avg_abs_inv": float(results["inventory"].abs().mean()),
        })

    return pd.DataFrame(rows)