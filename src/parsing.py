from pathlib import Path
import gc
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RAW_DIR = Path("../data")
OUT_PATH = Path("../data/processed/nvda_202606_events.parquet")

NUMERIC_PREFIXES = (
    "bid_px_", "ask_px_",
    "bid_sz_", "ask_sz_",
    "bid_ct_", "ask_ct_",
)

NUMERIC_EXACT = ("price", "size")


def find_daily_files(raw_dir: Path) -> list[Path]:
    files = sorted(Path(raw_dir).glob("*.csv.zst"))
    if not files:
        raise FileNotFoundError(f"Keine .csv.zst-Dateien in {raw_dir} gefunden.")
    return files


def peek_columns(files: list[Path]) -> None:
    first = pd.read_csv(files[0], compression="zstd", nrows=5)
    print(f"Spalten in {files[0].name} ({len(first.columns)} Stück):")
    print(list(first.columns))
    print(first.head(), "\n")


def extract_trading_day(path: Path) -> str:
    match = re.search(r"(20\d{6})", path.name)
    if match is None:
        raise ValueError(f"Kein Handelsdatum im Dateinamen gefunden: {path.name}")
    return match.group(1)


def parse_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="zstd")

    for column in ("ts_recv", "ts_event"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="raise")

    for column in df.columns:
        if column.startswith(NUMERIC_PREFIXES) or column in NUMERIC_EXACT:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    df["trading_day"] = extract_trading_day(path)
    return df


def parse_all(raw_dir: Path = RAW_DIR, out_path: Path = OUT_PATH) -> Path:
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    files = find_daily_files(raw_dir)

    peek_columns(files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer = None
    total_rows = 0

    try:
        for index, file_path in enumerate(files, start=1):
            print(f"Parse {file_path.name} ...")
            df = parse_day(file_path)

            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            elif not table.schema.equals(writer.schema):
                table = table.cast(writer.schema)

            writer.write_table(table, row_group_size=len(table))
            total_rows += len(df)

            print(f"  -> {len(df):,} Events als Tag {index}/{len(files)} geschrieben")

            del df, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print(f"\nFertig: {total_rows:,} Events aus {len(files)} Tagen in {out_path}")
    return out_path


if __name__ == "__main__":
    parse_all()