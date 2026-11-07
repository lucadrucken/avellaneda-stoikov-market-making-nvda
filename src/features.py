from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    EVENTS_PATH, FEATURES_PATH as OUT_PATH, TICK_SIZE, K_MIN, K_MAX,
    ladder_column_name, HALF_LIFE_MINUTES, SESSION_START_UTC, SESSION_END_UTC,
    MINUTES_PER_SESSION,
)


RESAMPLE_MS = 500

EVENT_LEVEL_COLUMNS = (
    "action", "side", "price", "size",
    "depth", "flags", "ts_in_delta", "sequence",
)


def iter_days(path: Path = EVENTS_PATH):
    parquet_file = pq.ParquetFile(path)
    current_day = None
    day_chunks = []

    for row_group_index in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(row_group_index).to_pandas()
        if chunk.empty:
            continue

        chunk["ts_recv"] = pd.to_datetime(chunk["ts_recv"])
        days = chunk["trading_day"].dropna().unique()

        if len(days) != 1:
            raise ValueError(f"Eine Row-Group enthält mehrere Handelstage: {days}")

        chunk_day = days[0]
        if current_day is None:
            current_day = chunk_day

        if chunk_day != current_day:
            day_df = pd.concat(day_chunks, ignore_index=True)
            day_df = day_df.sort_values("ts_recv").reset_index(drop=True)
            yield day_df

            current_day = chunk_day
            day_chunks = []

        day_chunks.append(chunk)

    if day_chunks:
        day_df = pd.concat(day_chunks, ignore_index=True)
        day_df = day_df.sort_values("ts_recv").reset_index(drop=True)
        yield day_df


def filter_session(
    df: pd.DataFrame,
    start: str = SESSION_START_UTC,
    end: str = SESSION_END_UTC,
) -> pd.DataFrame:
    ts_recv = pd.to_datetime(df["ts_recv"], utc=True)

    start_time = pd.Timestamp(start).time()
    end_time = pd.Timestamp(end).time()
    event_times = ts_recv.dt.time

    mask = (event_times >= start_time) & (event_times <= end_time)

    removed = int((~mask).sum())
    total = len(df)
    trading_day = df["trading_day"].iloc[0] if total > 0 else "unbekannt"
    
    print(
        f"  Session-Filter Tag {trading_day}: "
        f"{removed:,} von {total:,} Events außerhalb {start}-{end} UTC entfernt"
    )

    return df.loc[mask].reset_index(drop=True)


def add_midprice_spread(df: pd.DataFrame) -> pd.DataFrame:
    df["mid"] = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
    df["spread"] = df["ask_px_00"] - df["bid_px_00"]
    df["spread_bps"] = (df["spread"] / df["mid"]) * 10_000
    return df


def add_imbalance(df: pd.DataFrame, levels=(1, 5, 10)) -> pd.DataFrame:
    for level in levels:
        bid_cols = [f"bid_sz_{i:02d}" for i in range(level)]
        ask_cols = [f"ask_sz_{i:02d}" for i in range(level)]

        bid_sum = df[bid_cols].sum(axis=1)
        ask_sum = df[ask_cols].sum(axis=1)
        total_depth = bid_sum + ask_sum

        df[f"imbalance_l{level}"] = np.where(
            total_depth > 0,
            (bid_sum - ask_sum) / total_depth,
            0.0,
        )
        df[f"depth_l{level}"] = total_depth

    return df


def _build_ladder(
    trades: pd.DataFrame,
    side: str,
    anchor_col: str,
    tick_size: float,
    direction: str,
    label: str,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> pd.DataFrame:
    columns = [ladder_column_name(label, k) for k in range(k_min, k_max + 1)]

    subset = trades.loc[trades["side"] == side, ["bin_ts", "price", anchor_col, "size"]].copy()
    subset = subset.dropna(subset=[anchor_col])

    if subset.empty:
        return pd.DataFrame(columns=columns).rename_axis("ts_recv")

    if direction == "down":
        distance = (subset[anchor_col] - subset["price"]) / tick_size
    else:
        distance = (subset["price"] - subset[anchor_col]) / tick_size

    subset["distance_ticks"] = distance.round().astype(int)
    subset = subset.loc[subset["distance_ticks"] >= k_min]

    if subset.empty:
        return pd.DataFrame(columns=columns).rename_axis("ts_recv")

    subset["distance_ticks"] = subset["distance_ticks"].clip(upper=k_max)

    histogram = (
        subset.groupby(["bin_ts", "distance_ticks"])["size"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(k_min, k_max + 1), fill_value=0.0)
    )

    ladder = histogram.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    ladder.columns = columns
    ladder.index.name = "ts_recv"

    return ladder


def compute_interval_trade_stats(
    df: pd.DataFrame,
    ms: int,
    anchor: pd.DataFrame,
    tick_size: float = TICK_SIZE,
) -> pd.DataFrame:
    ladder_columns = (
        [ladder_column_name("sell", k) for k in range(K_MIN, K_MAX + 1)]
        + [ladder_column_name("buy", k) for k in range(K_MIN, K_MAX + 1)]
    )
    result_columns = [
        "ts_recv", "trade_count", "trade_volume",
        "sell_trade_price_min", "sell_trade_volume",
        "buy_trade_price_max", "buy_trade_volume",
    ] + ladder_columns

    trades = df.loc[
        df["action"] == "T",
        ["ts_recv", "side", "price", "size"],
    ].copy()

    if trades.empty:
        return pd.DataFrame(columns=result_columns)

    trades["ts_recv"] = pd.to_datetime(trades["ts_recv"])
    trades["bin_ts"] = trades["ts_recv"].dt.floor(f"{ms}ms")
    trades = trades.merge(anchor, left_on="bin_ts", right_on="ts_recv", how="left", suffixes=("", "_bin"))

    trades_indexed = trades.set_index("ts_recv")
    frequency = f"{ms}ms"

    all_trade_stats = trades_indexed.resample(frequency).agg(
        trade_count=("price", "count"),
        trade_volume=("size", "sum"),
    )

    sell_trades = trades_indexed.loc[trades_indexed["side"] == "A"]
    sell_stats = sell_trades.resample(frequency).agg(
        sell_trade_price_min=("price", "min"),
        sell_trade_volume=("size", "sum"),
    )

    buy_trades = trades_indexed.loc[trades_indexed["side"] == "B"]
    buy_stats = buy_trades.resample(frequency).agg(
        buy_trade_price_max=("price", "max"),
        buy_trade_volume=("size", "sum"),
    )

    sell_ladder = _build_ladder(
        trades, side="A", anchor_col="anchor_bid", tick_size=tick_size,
        direction="down", label="sell",
    )
    buy_ladder = _build_ladder(
        trades, side="B", anchor_col="anchor_ask", tick_size=tick_size,
        direction="up", label="buy",
    )

    result = (
        all_trade_stats
        .join(sell_stats, how="left")
        .join(buy_stats, how="left")
        .join(sell_ladder, how="left")
        .join(buy_ladder, how="left")
        .reset_index()
    )

    for column in ladder_columns:
        if column in result.columns:
            result[column] = result[column].fillna(0.0)

    return result


def _bin_aligned_chunks(df: pd.DataFrame, ms: int, target_rows: int = 1_000_000):
    n = len(df)
    if n <= target_rows:
        yield df
        return

    bin_ids = df["ts_recv"].to_numpy("datetime64[ns]").astype("int64") // (ms * 1_000_000)
    start = 0
    while start < n:
        end = min(start + target_rows, n)
        if end < n:
            current_bin = bin_ids[end - 1]
            while end < n and bin_ids[end] == current_bin:
                end += 1
        yield df.iloc[start:end]
        start = end


def resample_day(df: pd.DataFrame, ms: int = RESAMPLE_MS, tick_size: float = TICK_SIZE) -> pd.DataFrame:
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    event_level_present = [c for c in EVENT_LEVEL_COLUMNS if c in df.columns]
    book_source = df.drop(columns=event_level_present) if event_level_present else df

    book_chunks = [
        chunk.set_index("ts_recv").resample(f"{ms}ms").last().dropna(subset=["mid"]).reset_index()
        for chunk in _bin_aligned_chunks(book_source, ms)
    ]
    book_snapshots = pd.concat(book_chunks, ignore_index=True)
    del book_chunks

    anchor = pd.DataFrame({
        "ts_recv": book_snapshots["ts_recv"],
        "anchor_bid": book_snapshots["bid_px_00"].shift(1),
        "anchor_ask": book_snapshots["ask_px_00"].shift(1),
    })

    trade_stats = compute_interval_trade_stats(df=df, ms=ms, anchor=anchor, tick_size=tick_size)

    out = book_snapshots.merge(
        trade_stats,
        on="ts_recv",
        how="left",
        validate="one_to_one",
    )

    out["trade_count"] = out["trade_count"].fillna(0).astype(int)

    volume_columns = ["trade_volume", "sell_trade_volume", "buy_trade_volume"]
    for column in volume_columns:
        out[column] = out[column].fillna(0.0).astype(float)

    ladder_columns = (
        [ladder_column_name("sell", k) for k in range(K_MIN, K_MAX + 1)]
        + [ladder_column_name("buy", k) for k in range(K_MIN, K_MAX + 1)]
    )
    for column in ladder_columns:
        if column in out.columns:
            out[column] = out[column].fillna(0.0).astype(float)

    return out


def _daily_realized_vol(day_mid: pd.Series) -> float:
    minute_mid = day_mid.resample("1min").last().dropna()
    log_returns = np.log(minute_mid).diff().dropna()
    return float(np.sqrt((log_returns ** 2).sum()))


def add_realized_vol_and_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rv_by_day = (
        df.set_index("ts_recv")
        .groupby("trading_day")["mid"]
        .apply(_daily_realized_vol)
    )

    n_days = len(rv_by_day)
    tercile_size = n_days / 3.0
    ranks = rv_by_day.rank(method="first")

    regime_by_day = pd.Series(
        np.select(
            [ranks <= tercile_size, ranks <= 2 * tercile_size],
            ["low", "medium"],
            default="high",
        ),
        index=rv_by_day.index,
    )

    df["regime_day"] = df["trading_day"].map(regime_by_day)

    return df


def add_causal_sigma_ewma(
    df: pd.DataFrame,
    half_life_minutes: float,
    sigma2_seed: float = None,
) -> pd.DataFrame:
    df = df.sort_values("ts_recv").reset_index(drop=True)
    minute_ts = df["ts_recv"].dt.floor("1min")

    minute_level = (
        pd.DataFrame({
            "minute_ts": minute_ts,
            "trading_day": df["trading_day"],
            "mid": df["mid"],
        })
        .groupby("minute_ts", as_index=True)
        .agg(mid=("mid", "last"), trading_day=("trading_day", "last"))
        .sort_index()
    )

    minute_level["delta"] = minute_level["mid"].diff()

    is_day_start = minute_level["trading_day"] != minute_level["trading_day"].shift(1)
    minute_level.loc[is_day_start, "delta"] = np.nan

    deltas = minute_level["delta"].to_numpy()
    day_start_flags = is_day_start.to_numpy()
    sigma2 = np.full(len(deltas), np.nan)

    lam = 0.5 ** (1.0 / half_life_minutes)

    if sigma2_seed is not None:
        sigma2[0] = float(sigma2_seed)
        loop_start = 1
    else:
        valid_indices = np.flatnonzero(~np.isnan(deltas))
        if len(valid_indices) == 0:
            raise ValueError("Keine gueltigen 1-Minuten-Preisdifferenzen gefunden.")

        first_valid = valid_indices[0]
        sigma2[first_valid] = deltas[first_valid] ** 2
        loop_start = first_valid + 1

    for i in range(loop_start, len(deltas)):
        if day_start_flags[i]:
            sigma2[i] = sigma2[i - 1]
        else:
            prev_delta = deltas[i - 1]
            if np.isnan(prev_delta):
                sigma2[i] = sigma2[i - 1]
            else:
                sigma2[i] = lam * sigma2[i - 1] + (1.0 - lam) * (prev_delta ** 2)

    sigma_minute_daily = pd.Series(
        np.sqrt(sigma2 * MINUTES_PER_SESSION), index=minute_level.index,
    )

    df["sigma_ewma_usd"] = minute_ts.map(sigma_minute_daily)
    df["sigma_ewma_usd"] = df["sigma_ewma_usd"].ffill()

    return df


def add_time_to_close(
    df: pd.DataFrame,
    start: str = SESSION_START_UTC,
    end: str = SESSION_END_UTC,
) -> pd.DataFrame:
    ts_recv = pd.to_datetime(df["ts_recv"], utc=True)

    start_time = pd.Timestamp(start).time()
    end_time = pd.Timestamp(end).time()

    session_start_s = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    session_end_s = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
    session_length_s = session_end_s - session_start_s

    seconds_of_day = (
        ts_recv.dt.hour * 3600
        + ts_recv.dt.minute * 60
        + ts_recv.dt.second
        + ts_recv.dt.microsecond / 1_000_000
    )

    fraction_elapsed = (seconds_of_day - session_start_s) / session_length_s
    df["time_to_close"] = (1.0 - fraction_elapsed).clip(lower=0.0, upper=1.0)

    return df


def build_features(
    events_path: Path = EVENTS_PATH,
    out_path: Path = OUT_PATH,
    resample_ms: int = RESAMPLE_MS,
    tick_size: float = TICK_SIZE,
    sigma2_seed: float = None,
) -> pd.DataFrame:
    if resample_ms <= 0:
        raise ValueError("resample_ms muss positiv sein.")

    daily_features = []

    for day_df in iter_days(events_path):
        trading_day = day_df["trading_day"].iloc[0]
        print(f"Feature-Generierung: Tag {trading_day}")

        day_df = filter_session(day_df)
        day_df = add_midprice_spread(day_df)
        day_df = add_imbalance(day_df)
        day_features = resample_day(day_df, resample_ms, tick_size)

        daily_features.append(day_features)
        del day_df
        gc.collect()

    if not daily_features:
        raise ValueError("Es konnten keine Features erzeugt werden.")

    features = pd.concat(daily_features, ignore_index=True)
    features = features.sort_values("ts_recv").reset_index(drop=True)

    duplicate_count = int(features["ts_recv"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"{duplicate_count:,} doppelte Feature-Timestamps gefunden.")

    features = add_realized_vol_and_regime(features)
    features = add_causal_sigma_ewma(
        features, half_life_minutes=HALF_LIFE_MINUTES, sigma2_seed=sigma2_seed,
    )
    features = add_time_to_close(features)

    undefined_sigma = int(features["sigma_ewma_usd"].isna().sum())
    if undefined_sigma > 0:
        print(
            f"  {undefined_sigma:,} Zeilen ohne definierte sigma_ewma_usd "
            f"(Burn-in) entfernt."
        )
        features = features.dropna(subset=["sigma_ewma_usd"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)

    final_daily_sigma = float(features["sigma_ewma_usd"].iloc[-1])
    final_sigma_variance = (final_daily_sigma ** 2) / MINUTES_PER_SESSION
    
    print(
        f"\nsigma-Endwert (unskalierte 1-Minuten-Varianz für "
        f"sigma2_seed im nächsten Monat): {final_sigma_variance!r}"
    )

    print(f"\n{len(features):,} Snapshots ({resample_ms}ms Raster) geschrieben nach {out_path}")
    print(f"Spalten gesamt: {len(features.columns)}")
    print(
        features[
            [
                "ts_recv", "mid", "spread", "bid_px_00", "ask_px_00",
                "imbalance_l1", "regime_day", "trade_count",
            ]
        ].head()
    )

    return features


if __name__ == "__main__":
    build_features()