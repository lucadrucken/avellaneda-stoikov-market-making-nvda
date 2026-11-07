import math
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    TICK_SIZE, K_MAX, ladder_column_name, K_SENSITIVITY_VALUES,
    K_ESTIMATION_CRITERIA, ORDER_SIZE, PARTICIPATION_RATE, FILL_RULE,
)
from strategies import (
    NaiveStrategy, ASLiteralStrategy, HardLimitStrategy, HybridStrategy,
)
from simulation import run_simulation
from analysis import decompose_fills


MAY_FEATURES_PATH = Path("../data/processed/nvda_202605_features.parquet")
CALIBRATION_DAYS = 5
OUT_DIR = Path("../results/kalibrierung_diagnose")
QUOTE_HORIZON_MS = 500

HALF_SPREADS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
GAMMA_GRID = tuple(np.logspace(-7, -4, 8))
Q_MAX_GRID = tuple(int(v) for v in np.round(np.logspace(2, np.log10(30000), 12)))
HYBRID_GAMMAS = (GAMMA_GRID[4], GAMMA_GRID[6], GAMMA_GRID[7])

HARD_LIMIT_HALF_SPREAD = 0.01
HYBRID_K = 100.0


def _curvature_pvalue(delta: np.ndarray, ln_lambda: np.ndarray) -> float:
    if len(delta) < 4:
        return float("nan")
    X = np.column_stack([delta, delta ** 2])
    X = sm.add_constant(X)
    model = sm.OLS(ln_lambda, X).fit()
    if model.df_resid <= 0:
        return float("nan")
    return float(model.pvalues[2])


def load_calibration_period(
    may_features_path: Path = MAY_FEATURES_PATH,
    n_days: int = CALIBRATION_DAYS,
) -> tuple[pd.DataFrame, list]:
    feat = pd.read_parquet(may_features_path)
    feat["ts_recv"] = pd.to_datetime(feat["ts_recv"])
    available_days = sorted(feat["trading_day"].dropna().unique())
    calib_days = available_days[-n_days:]
    calib = (
        feat.loc[feat["trading_day"].isin(calib_days)]
        .sort_values("ts_recv")
        .reset_index(drop=True)
    )
    return calib, calib_days


def diagnose_k_regression(
    calib: pd.DataFrame,
    tick_size: float = TICK_SIZE,
    k_max: int = K_MAX,
    criteria: dict = K_ESTIMATION_CRITERIA,
) -> dict:
    rows = []
    for k in range(0, k_max + 1):
        sell_col = ladder_column_name("sell", k)
        buy_col = ladder_column_name("buy", k)
        sell_mean = float(calib[sell_col].mean())
        buy_mean = float(calib[buy_col].mean())
        lam = (sell_mean + buy_mean) / 2.0
        n_support_bins = int(((calib[sell_col] > 0) | (calib[buy_col] > 0)).sum())
        rows.append({
            "k": k,
            "delta_usd": k * tick_size,
            "sell_mean_volume": sell_mean,
            "buy_mean_volume": buy_mean,
            "lambda": lam,
            "n_support_bins": n_support_bins,
        })
    lambda_table = pd.DataFrame(rows)
    valid = lambda_table.loc[lambda_table["lambda"] > 0].copy()
    valid["ln_lambda"] = np.log(valid["lambda"])
    x = valid["delta_usd"].to_numpy()
    y = valid["ln_lambda"].to_numpy()
    if len(x) > 2:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        b0, b1 = model.params
        k_hat, ln_A, r2 = -b1, b0, model.rsquared
        dof = model.df_resid
        if dof > 0:
            ci = model.conf_int(alpha=0.05)
            k_hat_ci = (-ci[1][1], -ci[1][0])
        else:
            k_hat_ci = (float("nan"), float("nan"))
    else:
        k_hat = ln_A = r2 = float("nan")
        dof = 0
        k_hat_ci = (float("nan"), float("nan"))
    curvature_p = _curvature_pvalue(x, y)
    n_support_points = len(valid)
    min_bins_per_point = int(valid["n_support_bins"].min()) if n_support_points else 0
    lo_k, hi_k = criteria["k_plausible_range"]
    checks = {
        "1_k_hat_positiv": (k_hat > 0, k_hat),
        "2_r_squared_min_0_80": (r2 >= criteria["min_r_squared"], r2),
        "3_ci_schliesst_null_aus": (k_hat_ci[0] > 0, k_hat_ci),
        "4_stuetzstellen_und_bins": (
            n_support_points >= criteria["min_support_points"]
            and min_bins_per_point >= criteria["min_fills_per_point"],
            (n_support_points, min_bins_per_point),
        ),
        "5_k_hat_plausibler_bereich": (lo_k <= k_hat <= hi_k, k_hat),
        "6_keine_signifikante_kruemmung": (curvature_p >= criteria["max_curvature_pvalue"], curvature_p),
    }
    return {
        "lambda_table": lambda_table,
        "k_hat": k_hat,
        "ln_A": ln_A,
        "r2": r2,
        "dof": dof,
        "k_hat_ci": k_hat_ci,
        "n_support_points": n_support_points,
        "min_bins_per_point": min_bins_per_point,
        "curvature_pvalue": curvature_p,
        "checks": checks,
        "all_criteria_pass": all(v[0] for v in checks.values()),
    }


def _inventory_stats(quoting: pd.DataFrame) -> dict:
    abs_inv = quoting["inventory"].abs()
    return {
        "mean_abs_inventory": float(abs_inv.mean()),
        "q50_abs_inventory": float(abs_inv.quantile(0.5)),
        "q90_abs_inventory": float(abs_inv.quantile(0.9)),
        "q99_abs_inventory": float(abs_inv.quantile(0.99)),
        "max_abs_inventory": float(abs_inv.max()),
    }


def diagnose_naive_sweep(
    calib: pd.DataFrame,
    half_spreads=HALF_SPREADS,
    order_size: float = ORDER_SIZE,
    participation_rate: float = PARTICIPATION_RATE,
    quote_horizon_ms: int = QUOTE_HORIZON_MS,
    tick_size: float = TICK_SIZE,
    fill_rule: str = FILL_RULE,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    results_cache = {}
    for half_spread in half_spreads:
        strategy = NaiveStrategy(half_spread=half_spread, tick_size=tick_size)
        results_by_day, daily_summary = run_simulation(
            calib, strategy,
            order_size=order_size, participation_rate=participation_rate,
            quote_horizon_ms=quote_horizon_ms, fill_rule=fill_rule,
        )
        if not results_by_day or daily_summary.empty:
            continue
        results = pd.concat(results_by_day.values(), ignore_index=True)
        results_cache[half_spread] = results
        quoting = results.loc[~results["is_liquidation"]]
        if results["fill_size"].sum() <= 0:
            continue
        total_pnl = float(daily_summary["final_pnl"].sum())
        parts = decompose_fills(results, total_pnl)
        n_active_quotes = int(results["bid_active"].sum() + results["ask_active"].sum())
        n_fills = int(results["n_fills"].sum())
        capped_buy = int(np.isclose(results["buy_fill_size"], order_size, atol=1e-6).sum())
        capped_sell = int(np.isclose(results["sell_fill_size"], order_size, atol=1e-6).sum())
        rows.append({
            "half_spread": half_spread,
            "n_fills": n_fills,
            "volume": parts["volume"],
            "quoted_ct": parts["quoted_per_share"] * 100,
            "impact_ct": parts["impact_per_share"] * 100,
            "realized_ct": parts["realized_per_share"] * 100,
            "realized_total": parts["realized_total"],
            "inventory_total": parts["inventory_total"],
            "final_pnl": parts["final_pnl"],
            "avg_abs_inv": float(results["inventory"].abs().mean()),
            "fill_rate_per_active_quote": n_fills / n_active_quotes if n_active_quotes else float("nan"),
            "mean_volume_per_fill": parts["volume"] / n_fills if n_fills else float("nan"),
            "order_size_capped_share": (capped_buy + capped_sell) / n_fills if n_fills else float("nan"),
            **_inventory_stats(quoting),
        })
    return pd.DataFrame(rows), results_cache


def diagnose_gamma_sweep(
    calib: pd.DataFrame,
    k_values,
    gamma_grid=GAMMA_GRID,
    order_size: float = ORDER_SIZE,
    participation_rate: float = PARTICIPATION_RATE,
    quote_horizon_ms: int = QUOTE_HORIZON_MS,
    tick_size: float = TICK_SIZE,
    fill_rule: str = FILL_RULE,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        for gamma in gamma_grid:
            strategy = ASLiteralStrategy(gamma=gamma, k=k, tick_size=tick_size)
            results_by_day, daily_summary = run_simulation(
                calib, strategy,
                order_size=order_size, participation_rate=participation_rate,
                quote_horizon_ms=quote_horizon_ms, fill_rule=fill_rule,
            )
            if not results_by_day or daily_summary.empty:
                continue
            results = pd.concat(results_by_day.values(), ignore_index=True)
            quoting = results.loc[~results["is_liquidation"]]
            n_active_quotes = int(results["bid_active"].sum() + results["ask_active"].sum())
            n_fills = int(results["n_fills"].sum())
            abs_skew = (quoting["decision_mid"] - quoting["reference_price"]).abs()
            liquidity_term = (2.0 / gamma) * math.log(1.0 + gamma / k)
            half_spread_series = (quoting["inventory_penalty"] + liquidity_term) / 2.0
            rows.append({
                "k": k,
                "gamma": gamma,
                "fill_rate_per_active_quote": n_fills / n_active_quotes if n_active_quotes else float("nan"),
                "total_volume": float(results["fill_size"].sum()),
                "n_fills": n_fills,
                "mean_abs_skew_cents": float(abs_skew.mean()) * 100,
                "max_abs_skew_cents": float(abs_skew.max()) * 100,
                "share_below_tick": float((abs_skew < tick_size).mean()),
                "mean_half_spread_cents": float(half_spread_series.mean()) * 100,
                **_inventory_stats(quoting),
            })
    return pd.DataFrame(rows)


def _limit_run_stats(
    calib, strategy, q_max, order_size, participation_rate,
    quote_horizon_ms, fill_rule,
) -> dict:
    results_by_day, daily_summary = run_simulation(
        calib, strategy,
        order_size=order_size, participation_rate=participation_rate,
        quote_horizon_ms=quote_horizon_ms, fill_rule=fill_rule,
    )
    if not results_by_day or daily_summary.empty:
        return {k: float("nan") for k in [
            "limit_binding_share", "one_sided_share", "n_fills", "final_pnl",
            "mean_abs_inventory", "q50_abs_inventory", "q90_abs_inventory",
            "q99_abs_inventory", "max_abs_inventory"
        ]}
    results = pd.concat(results_by_day.values(), ignore_index=True)
    quoting = results.loc[~results["is_liquidation"]]
    return {
        "limit_binding_share": float((quoting["inventory"].abs() >= q_max).mean()),
        "one_sided_share": float((quoting["bid_active"] ^ quoting["ask_active"]).mean()),
        "n_fills": int(results["n_fills"].sum()),
        "final_pnl": float(daily_summary["final_pnl"].sum()),
        **_inventory_stats(quoting),
    }


def diagnose_q_max_grid(
    calib: pd.DataFrame,
    q_max_grid=Q_MAX_GRID,
    hybrid_gammas=HYBRID_GAMMAS,
    hard_limit_half_spread: float = HARD_LIMIT_HALF_SPREAD,
    hybrid_k: float = HYBRID_K,
    order_size: float = ORDER_SIZE,
    participation_rate: float = PARTICIPATION_RATE,
    quote_horizon_ms: int = QUOTE_HORIZON_MS,
    tick_size: float = TICK_SIZE,
    fill_rule: str = FILL_RULE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hard_limit_rows = []
    hybrid_rows = []
    for q_max in q_max_grid:
        strategy = HardLimitStrategy(
            half_spread=hard_limit_half_spread, q_max=q_max, tick_size=tick_size,
        )
        hard_limit_rows.append({"q_max": q_max, "half_spread": hard_limit_half_spread, **_limit_run_stats(
            calib, strategy, q_max, order_size, participation_rate, quote_horizon_ms, fill_rule,
        )})
    for gamma in hybrid_gammas:
        for q_max in q_max_grid:
            strategy = HybridStrategy(gamma=gamma, k=hybrid_k, q_max=q_max, tick_size=tick_size)
            hybrid_rows.append({"gamma": gamma, "k": hybrid_k, "q_max": q_max, **_limit_run_stats(
                calib, strategy, q_max, order_size, participation_rate, quote_horizon_ms, fill_rule,
            )})
    return pd.DataFrame(hard_limit_rows), pd.DataFrame(hybrid_rows)


def diagnose_smallness(calib: pd.DataFrame, order_size: float = ORDER_SIZE) -> pd.DataFrame:
    mean_bid_00 = float(calib["bid_sz_00"].mean())
    mean_ask_00 = float(calib["ask_sz_00"].mean())
    mean_depth_l1 = float(calib["depth_l1"].mean())
    mean_depth_l10 = float(calib["depth_l10"].mean())
    return pd.DataFrame([{
        "mean_bid_sz_00": mean_bid_00,
        "mean_ask_sz_00": mean_ask_00,
        "mean_depth_l1_top_of_book": mean_depth_l1,
        "mean_depth_l10": mean_depth_l10,
        "order_size": order_size,
        "order_size_pct_of_bid_sz_00": order_size / mean_bid_00 * 100,
        "order_size_pct_of_ask_sz_00": order_size / mean_ask_00 * 100,
        "order_size_pct_of_depth_l1": order_size / mean_depth_l1 * 100,
        "order_size_pct_of_depth_l10": order_size / mean_depth_l10 * 100,
    }])


def _save_csv(df: pd.DataFrame, name: str, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)


def run_diagnostics() -> None:
    calib, calib_days = load_calibration_period()
    diag1 = diagnose_k_regression(calib)
    _save_csv(diag1["lambda_table"], "diagnose1_lambda_by_k.csv")
    
    k_values_diag3 = list(K_SENSITIVITY_VALUES)
    if diag1["all_criteria_pass"]:
        k_values_diag3 = sorted(set(k_values_diag3) | {round(diag1["k_hat"], 4)})
    
    diag2, _ = diagnose_naive_sweep(calib)
    _save_csv(diag2, "diagnose2_naiv_sweep.csv")
    
    diag3 = diagnose_gamma_sweep(calib, k_values=k_values_diag3)
    _save_csv(diag3, "diagnose3_gamma_sweep.csv")
    
    diag4_hl, diag4_hy = diagnose_q_max_grid(calib)
    _save_csv(diag4_hl, "diagnose4_q_max_hard_limit.csv")
    _save_csv(diag4_hy, "diagnose4_q_max_hybrid.csv")
    
    diag5 = diagnose_smallness(calib)
    _save_csv(diag5, "diagnose5_kleinheitsannahme.csv")


if __name__ == "__main__":
    run_diagnostics()