from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import FEATURES_PATH


PROCESSED_PATH = Path("../data/processed/nvda_202606_events.parquet")


def iter_days(path: Path = PROCESSED_PATH):
    parquet_file = pq.ParquetFile(path)
    current_day = None
    day_chunks = []

    for row_group_index in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(row_group_index).to_pandas()
        if chunk.empty:
            continue

        if "trading_day" not in chunk.columns:
            raise ValueError("Die Spalte 'trading_day' fehlt im Event-Parquet.")

        trading_days = chunk["trading_day"].dropna().astype(str).unique()
        if len(trading_days) != 1:
            raise ValueError(
                f"Eine Row Group enthält mehrere oder keine Handelstage: {trading_days}"
            )

        chunk_day = trading_days[0]
        if current_day is None:
            current_day = chunk_day

        if chunk_day != current_day:
            yield pd.concat(day_chunks, ignore_index=True)

            current_day = chunk_day
            day_chunks = []

        day_chunks.append(chunk)

    if day_chunks:
        yield pd.concat(day_chunks, ignore_index=True)


def check_required_columns(df: pd.DataFrame) -> None:
    required_columns = {"trading_day", "bid_px_00", "ask_px_00", "action"}

    if "ts_recv" not in df.columns and "ts_event" not in df.columns:
        raise ValueError("Weder 'ts_recv' noch 'ts_event' ist vorhanden.")

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Folgende Pflichtspalten fehlen: {missing}")


def check_day(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Leerer Handelstag an check_day() übergeben.")

    check_required_columns(df)
    df = df.copy()

    ts_col = "ts_recv" if "ts_recv" in df.columns else "ts_event"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    trading_day_values = df["trading_day"].dropna().astype(str).unique()
    if len(trading_day_values) != 1:
        raise ValueError(
            f"Der DataFrame enthält mehrere oder keine Handelstage: {trading_day_values}"
        )

    trading_day = trading_day_values[0]

    timestamp_nans = int(df[ts_col].isna().sum())
    duplicate_timestamps = int(df[ts_col].duplicated().sum())

    timestamp_differences = df[ts_col].diff()
    monotonic_violations = int((timestamp_differences < pd.Timedelta(0)).sum())
    equal_consecutive_timestamps = int((timestamp_differences == pd.Timedelta(0)).sum())

    top_of_book_nans = int(df[["bid_px_00", "ask_px_00"]].isna().sum().sum())
    invalid_bid_prices = int((df["bid_px_00"].notna() & (df["bid_px_00"] <= 0)).sum())
    invalid_ask_prices = int((df["ask_px_00"].notna() & (df["ask_px_00"] <= 0)).sum())
    crossed_book = int(
        (
            df["bid_px_00"].notna()
            & df["ask_px_00"].notna()
            & (df["bid_px_00"] >= df["ask_px_00"])
        ).sum()
    )

    valid_actions = {"A", "C", "M", "R", "T"}
    bad_action_values = int((df["action"].notna() & ~df["action"].isin(valid_actions)).sum())
    missing_action_values = int(df["action"].isna().sum())

    bad_symbol_values = 0
    if "symbol" in df.columns:
        bad_symbol_values = int(
            (df["symbol"].notna() & (df["symbol"].astype(str).str.upper() != "NVDA")).sum()
        )

    missing_trading_day = int(df["trading_day"].isna().sum())

    negative_sizes = 0
    if "size" in df.columns:
        sizes = pd.to_numeric(df["size"], errors="coerce")
        negative_sizes = int((sizes < 0).sum())

    trade_mask = df["action"] == "T"
    trade_count = int(trade_mask.sum())

    invalid_trade_prices = 0
    if "price" in df.columns:
        trade_prices = pd.to_numeric(df.loc[trade_mask, "price"], errors="coerce")
        invalid_trade_prices = int((trade_prices.isna() | (trade_prices <= 0)).sum())

    unknown_trade_sides = 0
    invalid_trade_sides = 0
    if "side" in df.columns:
        trade_sides = df.loc[trade_mask, "side"]
        unknown_trade_sides = int((trade_sides.isna() | (trade_sides == "N")).sum())
        invalid_trade_sides = int(
            (trade_sides.notna() & ~trade_sides.isin({"A", "B", "N"})).sum()
        )

    return {
        "trading_day": trading_day,
        "rows": len(df),
        "trade_count": trade_count,
        "timestamp_nans": timestamp_nans,
        "duplicate_timestamps": duplicate_timestamps,
        "equal_consecutive_timestamps": equal_consecutive_timestamps,
        "monotonic_violations": monotonic_violations,
        "crossed_book": crossed_book,
        "top_of_book_nans": top_of_book_nans,
        "invalid_bid_prices": invalid_bid_prices,
        "invalid_ask_prices": invalid_ask_prices,
        "bad_action_values": bad_action_values,
        "missing_action_values": missing_action_values,
        "bad_symbol_values": bad_symbol_values,
        "missing_trading_day": missing_trading_day,
        "negative_sizes": negative_sizes,
        "invalid_trade_prices": invalid_trade_prices,
        "unknown_trade_sides": unknown_trade_sides,
        "invalid_trade_sides": invalid_trade_sides,
    }


def run_all_checks(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event-Parquet nicht gefunden: {path}")

    reports = []
    for day_df in iter_days(path):
        trading_day = day_df["trading_day"].dropna().astype(str).iloc[0]
        print(f"Validierung: Tag {trading_day}")
        reports.append(check_day(day_df))
        del day_df

    if not reports:
        raise ValueError("Im Event-Parquet wurden keine Handelstage gefunden.")

    report = pd.DataFrame(reports).sort_values("trading_day").reset_index(drop=True)

    print("\n=== Validierungsreport pro Handelstag ===")
    print(report.to_string(index=False))

    numeric_columns = report.select_dtypes(include="number").columns
    totals = report[numeric_columns].sum()

    print("\n=== Gesamtwerte ===")
    print(f"Handelstage:                    {len(report):,}")
    print(f"Zeilen gesamt:                  {int(totals.get('rows', 0)):,}")
    print(f"Trades gesamt:                  {int(totals.get('trade_count', 0)):,}")
    print(f"Fehlende Timestamps:            {int(totals.get('timestamp_nans', 0)):,}")
    print(f"Doppelte Timestamps:            {int(totals.get('duplicate_timestamps', 0)):,}")
    print(f"Gleiche aufeinanderfolgende TS: {int(totals.get('equal_consecutive_timestamps', 0)):,}")
    print(f"Monotonieverstöße:              {int(totals.get('monotonic_violations', 0)):,}")
    print(f"Crossed oder Locked Books:      {int(totals.get('crossed_book', 0)):,}")
    print(f"Top-of-Book NaNs:               {int(totals.get('top_of_book_nans', 0)):,}")
    print(f"Ungültige Bid-Preise:           {int(totals.get('invalid_bid_prices', 0)):,}")
    print(f"Ungültige Ask-Preise:           {int(totals.get('invalid_ask_prices', 0)):,}")
    print(f"Ungültige Action-Werte:         {int(totals.get('bad_action_values', 0)):,}")
    print(f"Fehlende Action-Werte:          {int(totals.get('missing_action_values', 0)):,}")
    print(f"Falsche Symbol-Werte:           {int(totals.get('bad_symbol_values', 0)):,}")
    print(f"Fehlende Trading Days:          {int(totals.get('missing_trading_day', 0)):,}")
    print(f"Negative Größen:                {int(totals.get('negative_sizes', 0)):,}")
    print(f"Ungültige Trade-Preise:         {int(totals.get('invalid_trade_prices', 0)):,}")

    unknown_trade_sides = int(totals.get("unknown_trade_sides", 0))
    total_trades = int(totals.get("trade_count", 0))

    print(f"Trades ohne Aggressor-Seite:    {unknown_trade_sides:,}")
    if total_trades > 0:
        print(f"Anteil Trades ohne Side:        {unknown_trade_sides / total_trades:.2%}")

    print(f"Ungültige Trade-Seiten:         {int(totals.get('invalid_trade_sides', 0)):,}")

    critical_columns = [
        "timestamp_nans", "monotonic_violations", "invalid_bid_prices",
        "invalid_ask_prices", "bad_action_values", "missing_action_values",
        "bad_symbol_values", "missing_trading_day", "negative_sizes",
        "invalid_trade_prices", "invalid_trade_sides",
    ]

    critical_issues = sum(
        int(report[col].fillna(0).sum())
        for col in critical_columns
        if col in report.columns
    )

    print()
    if critical_issues == 0:
        print("Ergebnis: Keine kritischen Datenfehler gefunden.")
    else:
        print(f"Ergebnis: {critical_issues:,} kritische Datenprobleme gefunden.")

    if report["crossed_book"].sum() > 0:
        print("Hinweis: Locked oder crossed Books sollten geprüft werden.")

    if report["duplicate_timestamps"].sum() > 0:
        print(
            "Hinweis: Mehrere Events dürfen denselben Timestamp besitzen. "
            "Das ist bei Eventdaten nicht automatisch ein Fehler."
        )

    if unknown_trade_sides > 0:
        print(
            "Hinweis: Trades ohne bekannte Aggressor-Seite werden im "
            "Fill-Modell konservativ keinem Bid- oder Ask-Fill zugeordnet."
        )

    return report


def check_sigma_daily_scale(
    features_path: Path = FEATURES_PATH,
    low: float = 0.005,
    high: float = 0.10,
) -> dict:
    features = pd.read_parquet(features_path, columns=["sigma_ewma_usd", "mid"])

    mean_sigma = float(features["sigma_ewma_usd"].mean())
    mean_mid = float(features["mid"].mean())
    implied_daily_vol_pct = mean_sigma / mean_mid

    passed = low <= implied_daily_vol_pct <= high

    print(f"Pipeline-Dimensionstest sigma_ewma_usd ({features_path}):")
    print(f"  mean(sigma_ewma_usd) = {mean_sigma:.6f}")
    print(f"  mean(mid)            = {mean_mid:.6f}")
    print(f"  implizite Tagesvol.  = {implied_daily_vol_pct:.4%}")
    print(f"  plausibler Bereich   = [{low:.2%}, {high:.2%}]")
    print(f"  Ergebnis: {'BESTANDEN' if passed else 'FEHLGESCHLAGEN -- Einheitenfehler wahrscheinlich'}")

    if not passed:
        raise ValueError(
            f"Implizite Tagesvolatilitaet {implied_daily_vol_pct:.4%} liegt ausserhalb "
            f"des plausiblen Bereichs [{low:.2%}, {high:.2%}] -- vermutlich ein "
            f"Einheitenfehler in sigma_ewma_usd."
        )

    return {
        "mean_sigma": mean_sigma,
        "mean_mid": mean_mid,
        "implied_daily_vol_pct": implied_daily_vol_pct,
        "low": low,
        "high": high,
        "passed": passed,
    }


if __name__ == "__main__":
    run_all_checks()