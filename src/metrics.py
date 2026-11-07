import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std() == 0 or returns.std() != returns.std():
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or downside.std() != downside.std():
        return np.nan
    return returns.mean() / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(cum_pnl: pd.Series) -> float:
    pnl = pd.concat(
        [pd.Series([0.0]), cum_pnl.reset_index(drop=True)],
        ignore_index=True,
    )
    running_max = pnl.cummax()
    drawdown = pnl - running_max
    return drawdown.min()


def value_at_risk(pnl_changes: pd.Series, alpha: float = 0.99) -> float:
    return pnl_changes.quantile(1 - alpha)


def expected_shortfall(pnl_changes: pd.Series, alpha: float = 0.99) -> float:
    var = value_at_risk(pnl_changes, alpha)
    tail = pnl_changes[pnl_changes <= var]
    return tail.mean() if len(tail) else np.nan


def fill_rate(n_fills: int, n_active_quotes: int) -> float:
    return n_fills / n_active_quotes if n_active_quotes else np.nan


def count_active_quotes(results: pd.DataFrame) -> int:
    if "bid_active" in results.columns and "ask_active" in results.columns:
        return int(results["bid_active"].sum() + results["ask_active"].sum())

    return 2 * len(results)


def summarize(results: pd.DataFrame) -> dict:
    if "is_liquidation" in results.columns:
        results = results.loc[~results["is_liquidation"]]

    inventory = results["inventory"].abs()

    return {
        "avg_abs_inventory": float(inventory.mean()),
        "inventory_q99": float(inventory.quantile(0.99)),
    }


def summarize_daily(daily_summary: pd.DataFrame) -> dict:
    daily_returns = daily_summary["final_pnl"]

    loss_days = int((daily_returns < 0).sum())
    worst_idx = daily_returns.idxmin()

    return {
        "sharpe_daily": sharpe_ratio(daily_returns, periods_per_year=252),
        "sortino_daily": sortino_ratio(daily_returns, periods_per_year=252),
        "var_95_daily": value_at_risk(daily_returns, 0.95),
        "es_95_daily": expected_shortfall(daily_returns, 0.95),
        "max_drawdown_daily": max_drawdown(daily_returns.cumsum()),
        "mean_daily_pnl": float(daily_returns.mean()),
        "std_daily_pnl": float(daily_returns.std()),
        "loss_day_share": loss_days / len(daily_returns) if len(daily_returns) else np.nan,
        "worst_day": daily_summary.loc[worst_idx, "trading_day"],
        "worst_day_pnl": float(daily_returns.loc[worst_idx]),
    }


def summarize_minute(results_by_day: dict, alpha: float = 0.99) -> dict:
    if not results_by_day:
        raise ValueError("results_by_day ist leer.")

    per_day_frames = []

    for trading_day, day_result in results_by_day.items():
        minute_pnl = (
            day_result.set_index("timestamp")["pnl"]
            .sort_index()
            .resample("1min")
            .last()
            .dropna()
        )
        minute_returns = minute_pnl.diff().dropna()

        per_day_frames.append(pd.DataFrame({
            "trading_day": trading_day,
            "minute_return": minute_returns.values,
        }))

    pooled = pd.concat(per_day_frames, ignore_index=True)
    returns = pooled["minute_return"]

    var = value_at_risk(returns, alpha)
    tail = pooled.loc[returns <= var]
    tail_counts = tail["trading_day"].value_counts()

    tail_obs_by_day = {
        str(day): int(tail_counts.get(day, 0))
        for day in sorted(pooled["trading_day"].unique(), key=str)
    }

    return {
        "alpha": alpha,
        "n_minute_returns": int(len(returns)),
        "minute_var": var,
        "minute_es": expected_shortfall(returns, alpha),
        "minute_pnl_vol": float(returns.std()),
        "minute_sortino": sortino_ratio(returns, periods_per_year=1.0),
        "n_tail_obs": int(len(tail)),
        "tail_obs_by_day": tail_obs_by_day,
    }


def summarize_intraday_drawdown(results_by_day: dict) -> dict:
    if not results_by_day:
        raise ValueError("results_by_day ist leer.")

    daily_drawdowns = {
        str(trading_day): float(max_drawdown(day_result["pnl"]))
        for trading_day, day_result in results_by_day.items()
    }

    values = pd.Series(daily_drawdowns)

    return {
        "daily_drawdowns": daily_drawdowns,
        "mean_intraday_drawdown": float(values.mean()),
        "worst_intraday_drawdown": float(values.min()),
    }