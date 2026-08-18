import yaml
import numpy as np
import torch
import pandas as pd
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
N2_EPSILON = 1e-8
REF_DEPTH_SCALE = 100.0
SPEED_SCALE = 0.1  # m/s — normalises the w_flow weighting
START_TIME = time.time()
# (typical open-ocean speed; adjust for your basin)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def log_status(message):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:6.1f}s] {message}",flush=True)

def load_config(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

# ═══════════════════════════════════════════════════════════════════════════════
# DATE PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def parse_date_column(df, date_col, date_format='%Y-%m-%d %H:%M:%S'):
    """
    Parse a combined date column into separate integer 'year' and 'month' cols.

    Supported without explicit format (dayfirst=True):
        09/06/2016  ->  year=2016, month=6
        2016-06-09  ->  year=2016, month=6
        09-Jun-2016 ->  year=2016, month=6

    WARNING: DD/MM/YYYY and MM/DD/YYYY are ambiguous when day <= 12.
    Always pass --date_format "%d/%m/%Y" to guarantee correct parsing.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available: {list(df.columns)}"
        )

    raw = df[date_col].astype(str)

    parsed = pd.to_datetime(raw, format=date_format, errors="coerce")
    fmt_used = date_format

    n_failed = parsed.isna().sum()
    if n_failed > 0:
        bad = raw[parsed.isna()].unique()[:5].tolist()
        raise ValueError(
            f"\n  Could not parse {n_failed:,} dates in '{date_col}' "
            f"using format '{fmt_used}'.\n"
            f"  Example unparseable values: {bad}\n"
            f"  Fix: pass --date_format explicitly, e.g.:\n"
            f"    --date_format '%d/%m/%Y'   for  09/06/2016\n"
            f"    --date_format '%m/%d/%Y'   for  06/09/2016\n"
            f"    --date_format '%Y-%m-%d'   for  2016-06-09\n"
            f"    --date_format '%d-%b-%Y'   for  09-Jun-2016"
        )

    df = df.copy()
    df["year"] = parsed.dt.year.astype(int)
    df["month"] = parsed.dt.month.astype(int)

    n_show = min(4, len(df))
    sample = pd.DataFrame(
        {
            "raw": raw.iloc[:n_show].values,
            "parsed": parsed.iloc[:n_show].dt.strftime("%Y-%m-%d").values,
            "year": df["year"].iloc[:n_show].values,
            "month": df["month"].iloc[:n_show].values,
        }
    )
    log_status(f"  Date format used : {fmt_used}")
    log_status("  Parse sample — verify month/day order is correct:")
    log_status(sample.to_string(index=False))
    log_status("")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_observations(
    path, lat_col,lon_col,tracer_col, temp_col, sal_col,depth_col, u_col, v_col, date_col, date_format
):
    """
    Load observation CSV.

    Velocity columns (u_col, v_col) are optional — if absent a warning is
    log_statused and the advection loss will be skipped at training time.
    All other required columns are enforced.
    """
    df = pd.read_parquet(path)
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    # ── Date ──────────────────────────────────────────────────────────────────
    if date_col is not None:
        df = parse_date_column(df, date_col, date_format)
    else:
        for col in ("year", "month"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Either include separate "
                    f"'year' and 'month' columns, or pass --date_col."
                )

    # ── Required columns ──────────────────────────────────────────────────────
    required = {
        temp_col,
        sal_col,
        depth_col,
        lon_col,
        lat_col,
        u_col,
        v_col
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if tracer_col not in df.columns:
        raise ValueError(
            f"Tracer column '{tracer_col}' not found. Available: {list(df.columns)}"
        )
    if tracer_col != "tracer":
        df = df.rename(columns={tracer_col: "tracer"})

    # ── Rename columns ──────────────────────────────────────────────────────
    df = df.rename(columns={temp_col: "temperature",
                            sal_col: "salinity",
                            depth_col: "depth",
                            lon_col: "lon",
                            lat_col: "lat",
                            date_col: "dt",
                            u_col:"u",
                            v_col:"v"
                           })


    # ── Drop NaNs ─────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[
        "temperature",
        "salinity",
        "depth",
        "lon",
        "lat",
        "tracer"
    ])
    log_status(f"  Loaded {before:,} rows, {len(df):,} after dropping NaNs")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarise_dataset(df):
    log_status("\n── Dataset summary ────────────────────────────────────────")
    log_status(f"  Observations : {len(df):,}")
    log_status(f"  Year range   : {(df['dt'].min())} - {(df['dt'].max())}")
    log_status(f"  Depth range  : {df['depth'].min():.1f} - {df['depth'].max():.1f} m")
    log_status(f"  Tracer range : {df['tracer'].min():.3f} - {df['tracer'].max():.3f}")
    log_status(f"  Lat range    : {df['lat'].min():.1f} - {df['lat'].max():.1f} deg")
    log_status(f"  Lon range    : {df['lon'].min():.1f} - {df['lon'].max():.1f} deg")
    log_status(f"  Temperature range : {df['temperature'].min():.1f} - {df['temperature'].max():.1f} degC")
    log_status(f"  Salinity range    : {df['salinity'].min():.1f} - {df['salinity'].max():.1f}")

    speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
    log_status(
        f"  Speed range  : {speed.min():.4f} - {speed.max():.4f} m/s "
        f"(mean {speed.mean():.4f})"
        )
    decade = df.groupby((df["year"] // 10) * 10).size()
    log_status("\n  Obs per decade:")
    for dec, n in decade.items():
        bar = "█" * (n // max(1, len(df) // 200))
        log_status(f"    {int(dec)}s  {n:>6,}  {bar}")
    log_status("")
