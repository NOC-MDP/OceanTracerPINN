import gsw
import numpy as np
from utilities.utils import N2_EPSILON, REF_DEPTH_SCALE

# ── 1. FEATURE ENGINEERING ─────────────────────────────────────────────────────
#
# Key idea: rotate (T, S) → (σ₀, spiciness)
#   σ₀       = potential density anomaly  → which isopycnal you're on
#   spiciness = orthogonal to σ₀ in T-S   → water mass identity ON that isopycnal
#
# This separates "across-isopycnal" from "along-isopycnal" variation,
# which is the natural coordinate system for conservative tracer physics.
def build_features(df, cast_col="cast_id"):
    """
    Build the full physics-informed feature matrix.

    Required DataFrame columns
    --------------------------
    temp, salinity, depth, lon, lat, year, month, tracer
    u, v          ← NEW: climatological velocity components (m/s)

    Optional
    --------
    cast_col      profile/cast identifier for correct per-cast N² computation

    Returns
    -------
    X           : np.ndarray  (n_valid, n_features)
    y           : np.ndarray  (n_valid,)  tracer values
    feat_names  : list[str]
    valid_mask  : boolean index array into the original df rows
    """
    for col in ("u", "v"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Velocity components u and v are required. "
                f"If you only have speed and direction, derive u and v first:\n"
                f"  df['u'] = speed * np.cos(np.deg2rad(direction))\n"
                f"  df['v'] = speed * np.sin(np.deg2rad(direction))"
            )

    # ── TEOS-10 isopycnal coordinates ─────────────────────────────────────────
    # 1. Convert depth to sea pressure (dbar)
    p = gsw.p_from_z(-df["depth"].values, df["lat"].values)
    SA = gsw.SA_from_SP(
        df["salinity"].values, p, df["lon"].values, df["lat"].values
    )
    CT = gsw.CT_from_t(SA, df["temperature"].values, p)
    sigma0 = gsw.sigma0(SA, CT)
    spice = gsw.spiciness0(SA, CT)

    # ── Depth ─────────────────────────────────────────────────────────────────
    log_depth = np.log1p(np.abs(df["depth"].values))

    # ── Geographic position (periodic encoding) ───────────────────────────────
    #
    # Avoids the International Date Line discontinuity:
    #   lon = -180° and +180° map to the same location.
    #
    # Latitude:
    #   latn = sin(lat)
    #
    # Longitude:
    #   lonn_sin = sin(lon)
    #   lonn_cos = cos(lon)

    lat_rad = np.deg2rad(df["lat"].values)
    lon_rad = np.deg2rad(df["lon"].values)

    lat_norm = np.sin(lat_rad)

    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)

    #── Velocity-derived features ─────────────────────────────────────────────
    u = np.asarray(df["u"].values, dtype=np.float64)
    v = np.asarray(df["v"].values, dtype=np.float64)

    # 1. Handle missing/fill values if u or v contain unexpected NaNs or extreme fill values
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    # 2. Compute speed safely (clip at 0 to prevent negative sqrt underflow)
    speed_sq = np.maximum(0.0, u**2 + v**2)
    speed = np.sqrt(speed_sq)  # [m/s] flow intensity

    # 3. Flow direction (sin θ, cos θ)
    eps = 1e-10
    flow_sin = np.where(speed > eps, v / (speed + eps), 0.0)
    flow_cos = np.where(speed > eps, u / (speed + eps), 0.0)

    # 4. Compute log_speed safely
    log_speed = np.log1p(np.clip(speed, a_min=0.0, a_max=None))

    # ── Assemble ───────────────────────────────────────────────────────────────
    feature_names = [
        "sigma0",
        "spice",
        "log_depth",
        "lat_norm",
        "lon_sin",
        "lon_cos",
        "log_speed",
        "flow_sin",
        "flow_cos",
    ]

    X = np.column_stack(
        [
            sigma0,
            spice,
            log_depth,
            lat_norm,
            lon_sin,
            lon_cos,
            log_speed,
            flow_sin,
            flow_cos,
        ]
    )

    # Flag and drop NaNs (bad casts, land-locked etc.)
    has_tracer = "tracer" in df.columns

    # Diagnostic check:
    for name, col in zip(feature_names, X.T):
        nan_cnt = np.isnan(col).sum()
        inf_cnt = np.isinf(col).sum()
        if nan_cnt > 0 or inf_cnt > 0:
            print(f"Feature '{name}' generated {nan_cnt} NaNs and {inf_cnt} Infs")

    valid = np.isfinite(X).all(axis=1)
    if has_tracer:
        valid = valid & np.isfinite(df["tracer"].values)

    print(
        f"  Speed range : {speed[valid].min():.4f} – "
        f"{speed[valid].max():.4f} m/s  "
        f"(mean {speed[valid].mean():.4f})"
    )
    y = df["tracer"].values[valid] if has_tracer else None

    return X[valid], y, feature_names, valid
