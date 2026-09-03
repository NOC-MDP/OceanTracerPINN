# ── 8. INFERENCE ON OCEAN MODEL ───────────────────────────────────────────────
import gsw
import numpy as np
from pyproj import Transformer
import xarray as xr
import torch
import os
from models.architecture import TracerPINN
from sklearn.preprocessing import StandardScaler

def load_checkpoint(path, device):
    """
    Load a .pt checkpoint saved by main.py and reconstruct the model + scaler.

    Checkpoint schema (written by main.py):
        model_state   : nn.Module state_dict
        feat_names    : list[str]
        hidden_dim    : int
        n_blocks      : int
        n_features    : int
        scaler_mean   : np.ndarray
        scaler_scale  : np.ndarray
    """
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Checkpoint not found: {path}")

    ckpt = torch.load(path, weights_only=False, map_location=device)

    required_keys = {
        "model_state",
        "feat_names",
        "hidden_dim",
        "n_blocks",
        "n_features",
        "scaler_mean",
        "scaler_scale",
    }
    missing = required_keys - set(ckpt.keys())
    if missing:
        sys.exit(
            f"[ERROR] Checkpoint is missing keys: {missing}\n"
            f"        Make sure you are loading a checkpoint saved by main.py."
        )

    # Reconstruct model
    model = TracerPINN(
        n_features=ckpt["n_features"],
        hidden_dim=ckpt["hidden_dim"],
        n_blocks=ckpt["n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Reconstruct scaler (sklearn StandardScaler shell, no re-fitting needed)
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = ckpt["n_features"]

    feat_names = ckpt["feat_names"]

    print(f"  Checkpoint   : {path}")
    print(
        f"  Architecture : hidden_dim={ckpt['hidden_dim']}, "
        f"n_blocks={ckpt['n_blocks']}, n_features={ckpt['n_features']}"
    )
    print(f"  Features     : {feat_names}")
    return model, scaler, feat_names

def infer_on_model_field(
    cfg,
    ds,
    model,
    scaler,
    feat_names,
    target_year,
    target_month,
    device,
    n_mc,
    target_time_index=0
):
    """
    Run the trained PINN on a gridded ocean model Dataset.
    """

    # # ── Candidate variable names ───────────────────────────────────────────────
    # _TEMP_VARS = ["temp", "temperature", "thetao", "votemper", "potential_temperature", "THETA"]
    # _SALT_VARS = ["salinity", "salt", "so", "vosaline", "practical_salinity", "SALT"]
    # _UVEL_VARS = ["vxo", "EVEL"]
    # _VVEL_VARS = ["vyo", "NVEL"]
    # _DEPTH_NAMES = ["depth", "deptht", "depthu", "depthv", "Z", "z", "lev", "level"]
    # _LAT_NAMES = ["lat", "latitude", "nav_lat", "yt_ocean", "nlat", "y"]
    # _LON_NAMES = ["lon", "longitude", "nav_lon", "xt_ocean", "nlon", "x"]
    # _TIME_NAMES = ["time", "time_counter", "t", "time_0", "time_centered"]

    # def _find_var(ds, candidates, label):
    #     low = {v.lower(): v for v in ds.data_vars}
    #     for c in candidates:
    #         if c.lower() in low:
    #             return low[c.lower()]
    #     raise ValueError(
    #         f"Cannot find {label} variable. "
    #         f"Available: {list(ds.data_vars)}. "
    #         f"Pass {label.lower()}_var='your_name' to override."
    #     )

    # def _find_dim(da, candidates, label):
    #     low = {d.lower(): d for d in da.dims}
    #     for c in candidates:
    #         if c.lower() in low:
    #             return low[c.lower()]
    #     raise ValueError(f"Cannot find {label} dimension in {list(da.dims)}.")

    def _to_3d(da, dep_dim, lat_dim, lon_dim):
        """Squeeze extra dims then transpose to available spatial dims (handles 2D transects or 3D grids)."""
        # Filter out None values and only keep dimensions that actually exist in the DataArray
        target_dims = [d for d in [dep_dim, lat_dim, lon_dim] if d is not None and d in da.dims]

        # Squeeze any extra dimensions (like remaining time dimensions)
        extra = [d for d in da.dims if d not in target_dims]
        for d in extra:
            da = da.isel({d: 0}) if da.sizes[d] > 1 else da.squeeze(d)

        # Transpose dynamically to whatever spatial dimensions are present
        return da.transpose(*target_dims).values

    # # ── Variable detection ────────────────────────────────────────────────────
    # temp_var = _find_var(ds, _TEMP_VARS, "temperature")
    # salt_var = _find_var(ds, _SALT_VARS, "salinity")
    # uvel_var = _find_var(ds, _UVEL_VARS, "u-velocity")
    # vvel_var = _find_var(ds, _VVEL_VARS, "v-velocity")
    # print(f"  [infer] temp='{temp_var}', salt='{salt_var}', u='{uvel_var}', v='{vvel_var}'")

    # # # ── Dimension detection ───────────────────────────────────────────────────
    # ref = ds[cfg['temp_var']]
    # print(ref)
    # # dep_dim = _find_dim(ref, _DEPTH_NAMES, "depth")
    # # lat_dim = _find_dim(ref, _LAT_NAMES, "latitude")
    # # lon_dim = _find_dim(ref, _LON_NAMES, "longitude")

    # # Handle time
    # # time_dim = next((d for d in _TIME_NAMES if d in ref.dims), None)
    # if cfg['time_dim'] and ref.sizes[cfg['time_dim']] > 1:
    #     print(f"  [infer] Slicing time index {target_time_index} from dim '{cfg['time_dim']}' (size {ref.sizes[cfg['time_dim']]})")
    #     ds = ds.isel({cfg['time_dim']: target_time_index})
    #     ref = ds[cfg['temp_var']]

    # ── Extract 3-D arrays ────────────────────────────────────────────────────
    temp_3d = _to_3d(ds[cfg['temp_var']], cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])
    salt_3d = _to_3d(ds[cfg['salt_var']], cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])
    v_3d = _to_3d(ds[cfg['v_var']],cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])
    if cfg['u_var'] is None:
        u_3d = np.zeros_like(v_3d)
    else:
        u_3d = _to_3d(ds[cfg['u_var']], cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])


    if temp_3d.ndim == 3:
        n_dep, n_lat, n_lon = temp_3d.shape
    elif temp_3d.ndim == 2:
        n_dep, n_lon = temp_3d.shape
        n_lat = 1  # Dummy or singleton size for latitude if needed downstream

    depth_vals = ds[cfg['dep_dim']].values

    if temp_3d.ndim == 2:
        # --- 2D TRANSECT CASE (Depth x Lon) ---
        n_dep, n_lon = temp_3d.shape

        # Broadcast depth and lon to 2D
        depth_3d = np.broadcast_to(depth_vals[:, np.newaxis], temp_3d.shape).copy()

        lon_1d = ds.coords[cfg['lon_dim']].values if cfg['lon_dim'] in ds.coords else np.arange(n_lon)
        lon_3d = np.broadcast_to(lon_1d[np.newaxis, :], temp_3d.shape).copy()

        # Handle scalar latitude for the transect
        lat_val = ds.coords[cfg['lat_dim']].values if cfg['lat_dim'] in ds.coords else 0.0
        lat_3d = np.full(temp_3d.shape, lat_val)

    else:
        # --- 3D GRID CASE (Depth x Lat x Lon) ---
        n_dep, n_lat, n_lon = temp_3d.shape

        depth_3d = np.broadcast_to(depth_vals[:, np.newaxis, np.newaxis], temp_3d.shape).copy()

        lat_1d = ds.coords[cfg['lat_dim']].values if cfg['lat_dim'] in ds.coords else np.zeros(n_lat)
        lon_1d = ds.coords[cfg['lon_dim']].values if cfg['lon_dim'] in ds.coords else np.zeros(n_lon)

        if lat_1d.ndim == 2:  # curvilinear grid
            lat_3d = np.broadcast_to(lat_1d[np.newaxis], temp_3d.shape).copy()
            lon_3d = np.broadcast_to(lon_1d[np.newaxis], temp_3d.shape).copy()
        else:
            lat_3d = np.broadcast_to(lat_1d[np.newaxis, :, np.newaxis], temp_3d.shape).copy()
            lon_3d = np.broadcast_to(lon_1d[np.newaxis, np.newaxis, :], temp_3d.shape).copy()


    # ── Define Target Polar Stereographic Projection (EPSG:3413) ──────────────
    lon_0 = -45.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    x_grid, y_grid = transformer.transform(lon_3d, lat_3d)

    gamma = np.radians(lon_3d - lon_0)

    # Fixed bug: avoided self-overwriting u_3d before calculating v_3d
    u_orig = u_3d.copy()
    v_orig = v_3d.copy()
    u_3d = u_orig * np.cos(gamma) - v_orig * np.sin(gamma)
    v_3d = u_orig * np.sin(gamma) + v_orig * np.cos(gamma)

    # ── Flatten ───────────────────────────────────────────────────────────────
    T_flat = temp_3d.ravel()
    S_flat = salt_3d.ravel()
    z_flat = depth_3d.ravel()
    lat_f = lat_3d.ravel()
    lon_f = lon_3d.ravel()
    u_flat = u_3d.ravel()
    v_flat = v_3d.ravel()

    # ── TEOS-10 ───────────────────────────────────────────────────────────────
    p_flat = gsw.p_from_z(-z_flat, lat_f)
    if not cfg['conservativeT']:
        SA = gsw.SA_from_SP(S_flat, p_flat, lon_f, lat_f)
    else:
        SA = S_flat
    if not cfg['absoluteS']:
        CT = gsw.CT_from_pt(SA,T_flat)
    else:
        CT = T_flat
    sigma0 = gsw.sigma0(SA, CT)
    spice = gsw.spiciness0(SA, CT)
    sigma2 = gsw.sigma2(SA, CT)

    # ── N² proxy / stratification feature ────────────────────────────────
    alpha = gsw.alpha(SA, CT, p_flat)
    beta = gsw.beta(SA, CT, p_flat)
    rho = gsw.rho(SA, CT, p_flat)

    REF_DEPTH_SCALE = 100.0
    N2_EPSILON = 1e-8

    n2_raw = (
        (9.7963 / rho)
        * (alpha + beta)
        / REF_DEPTH_SCALE
    )

    n2_clean = np.nan_to_num(
        n2_raw,
        nan=N2_EPSILON,
        posinf=N2_EPSILON,
        neginf=N2_EPSILON,
    )

    n2 = np.maximum(
        n2_clean,
        N2_EPSILON
    )

    log_n2 = np.log10(n2)

    # ── Velocity features ─────────────────────────────────────────────────────
    eps_v = 1e-10
    speed = np.sqrt(u_flat**2 + v_flat**2)
    flow_sin = np.where(speed > eps_v, v_flat / (speed + eps_v), 0.0)
    flow_cos = np.where(speed > eps_v, u_flat / (speed + eps_v), 0.0)
    log_speed = np.log1p(speed)

    # ── Scalar temporal/seasonal features ────────────────────────────────────
    n_pts = T_flat.shape[0]
    year_norm = (target_year - 1970) / 53.0
    season_sin = np.sin(2 * np.pi * target_month / 12)
    season_cos = np.cos(2 * np.pi * target_month / 12)

    # ── Feature map — must match training order ───────────────────────────────
    feature_map = {
        "sigma0": sigma0,
        "spice": spice,
        "sigma2": sigma2,
        "log_depth": np.log1p(z_flat),
        "log_n2": log_n2,
        "year_norm": np.full(n_pts, year_norm),
        "season_sin": np.full(n_pts, season_sin),
        "season_cos": np.full(n_pts, season_cos),
        "lat_norm": np.sin(np.deg2rad(lat_f)),
        "lon_sin": np.sin(np.deg2rad(lon_f)),
        "lon_cos": np.cos(np.deg2rad(lon_f)),
        "log_speed": log_speed,
        "flow_sin": flow_sin,
        "flow_cos": flow_cos,
    }

    missing = [f for f in feat_names if f not in feature_map]
    if missing:
        raise ValueError(
            f"Features {missing} not computed. Add them to feature_map in infer_on_model_field."
        )

    X_model = np.column_stack([feature_map[f] for f in feat_names])

    # ── Mask land / ice ───────────────────────────────────────────────────────
    valid = np.isfinite(X_model).all(axis=1)
    n_valid = valid.sum()
    print(f"  [infer] Valid ocean points: {n_valid:,} / {valid.size:,} ({100 * n_valid / valid.size:.1f}%)")

    print("\n========== INFERENCE FEATURE DIAGNOSTICS ==========")

    print("Feature names:")
    for i, name in enumerate(feat_names):
        print(
            f"  {i:2d}: {name:20s} "
            f"raw min={np.nanmin(X_model[:, i]):12.5g} "
            f"max={np.nanmax(X_model[:, i]):12.5g} "
            f"mean={np.nanmean(X_model[:, i]):12.5g} "
            f"std={np.nanstd(X_model[:, i]):12.5g}"
        )

    X_scaled = scaler.transform(X_model[valid])

    print("\nScaled features:")
    for i, name in enumerate(feat_names):
        print(
            f"  {i:2d}: {name:20s} "
            f"min={np.nanmin(X_scaled[:, i]):12.5g} "
            f"max={np.nanmax(X_scaled[:, i]):12.5g} "
            f"mean={np.nanmean(X_scaled[:, i]):12.5g} "
            f"std={np.nanstd(X_scaled[:, i]):12.5g}"
        )

    print("\nScaler:")
    print("  mean :", scaler.mean_)
    print("  scale:", scaler.scale_)

    print("====================================================\n")

    X_valid = scaler.transform(X_model[valid])

    # ── Vectorized MC-Dropout inference ───────────────────────────────────────
    model.to(device)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    # Adjusted batch size to account for the n_mc dimension expansion
    batch_size = 50_000
    n = X_valid.shape[0]

    mean = np.zeros(n, dtype=np.float32)
    sq_mean = np.zeros(n, dtype=np.float32)

    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    use_autocast = "cuda" in str(device) and torch.cuda.is_available()

    for i in range(0, n, batch_size):
        X_batch_np = X_valid[i : i + batch_size]
        curr_batch_len = X_batch_np.shape[0]

        # Transfer batch to device once
        X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)

        # Replicate batch n_mc times: shape (n_mc * curr_batch_len, num_features)
        X_expanded = X_batch.repeat_interleave(n_mc, dim=0)

        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_autocast):
            y_expanded = model(X_expanded)  # (curr_batch_len * n_mc, out_dim)

        # Reshape to (curr_batch_len, n_mc)
        mc_preds = y_expanded.view(curr_batch_len, n_mc)

        # Compute statistics directly on GPU, then bring scalar outputs to CPU
        batch_mean = mc_preds.mean(dim=1)
        batch_sq_mean = (mc_preds**2).mean(dim=1)

        mean[i : i + curr_batch_len] = batch_mean.cpu().numpy().squeeze()
        sq_mean[i : i + curr_batch_len] = batch_sq_mean.cpu().numpy().squeeze()

        if i == 0:
            y_test = (
                y_expanded
                .detach()
                .float()
                .cpu()
                .numpy()
                .squeeze()
            )

            print("\n========== MODEL OUTPUT CHECK ==========")
            print(f"min  = {np.nanmin(y_test):.6f}")
            print(f"max  = {np.nanmax(y_test):.6f}")
            print(f"mean = {np.nanmean(y_test):.6f}")
            print(f"std  = {np.nanstd(y_test):.6f}")
            print("========================================")

    tracer_mean = mean
    tracer_std = np.sqrt(np.maximum(sq_mean - mean**2, 0.0))

    tracer_field = np.full(valid.size, np.nan)
    uncert_field = np.full(valid.size, np.nan)
    tracer_field[valid] = tracer_mean
    uncert_field[valid] = tracer_std

    # ── Output Dataset ────────────────────────────────────────────────────────
    if temp_3d.ndim == 2:
        # Treat 2D transect as 3D with a singleton latitude dimension of size 1
        n_lat = 1
        out_dims = (cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])

        # Extract the scalar latitude value (or default to 0.0) and make it a 1D list/array
        lat_val = ds.coords[cfg['lat_dim']].values.item() if cfg['lat_dim'] in ds.coords else 0.0

        out_coords = {
            cfg['dep_dim']: ds.coords.get(cfg['dep_dim'], np.arange(n_dep)),
            cfg['lat_dim']: [lat_val],  # 1D coordinate of size 1 matching the dimension
            cfg['lon_dim']: ds.coords.get(cfg['lon_dim'], np.arange(n_lon)),
        }
    else:
        # 3D Grid Case (Depth x Lat x Lon)
        out_dims = (cfg['dep_dim'], cfg['lat_dim'], cfg['lon_dim'])
        out_coords = {
            cfg['dep_dim']: ds.coords.get(cfg['dep_dim'], np.arange(n_dep)),
            cfg['lat_dim']: ds.coords.get(cfg['lat_dim'], np.arange(n_lat)),
            cfg['lon_dim']: ds.coords.get(cfg['lon_dim'], np.arange(n_lon)),
        }

    # netcdf attributes can't be None so convert to String None
    if cfg['u_var'] is None:
        u_var_used = "None"
    else:
        u_var_used = cfg['u_var']

    ds_out = xr.Dataset(
        {
            "tracer_pred": (out_dims, tracer_field.reshape(n_dep, n_lat, n_lon)),
            "tracer_uncertainty": (out_dims, uncert_field.reshape(n_dep, n_lat, n_lon)),
        },
        coords=out_coords,
        attrs={
            "description": "PINN tracer inference with velocity loss",
            "target_year": target_year,
            "target_month": target_month,
            "temp_var_used": cfg['temp_var'],
            "salt_var_used": cfg['salt_var'],
            "u_var_used": u_var_used,
            "v_var_used": cfg['v_var'],
        },
    )
    return ds_out
