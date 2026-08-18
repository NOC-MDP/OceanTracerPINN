from inference.inference import infer_on_model_field
import os
import sys
import glob
import matplotlib
import cmocean
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models.architecture import TracerPINN
import cartopy.crs as ccrs
import cartopy.feature as cfeature

cfg = {"model_year": 1991,
    "model_month": 2,
    "seed": 42,
    "nc_path":"/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-04-01.nc",
    "output_dir": "/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN/outputs/inference",
    "mc_samples": 100,
    "checkpoint": "/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN/outputs/oxygen18/experiment1/pinn_oxygen18.pt",

}

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


np.random.seed(cfg['seed'])
torch.manual_seed(cfg['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg['seed'])

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(cfg['output_dir'], exist_ok=True)

print(f"\n{'=' * 60}")
print(" PINN Tracer Inference — pre-trained model")
print(f"{'=' * 60}")
print(f"  Device       : {device}")
print(f"  Checkpoint   : {cfg['checkpoint']}")
print(f"  Ocean model  : {cfg['nc_path']}")
print(f"  Target time  : {cfg['model_year']}-{cfg['model_month']:02d}")
print(f"  MC samples   : {cfg['mc_samples']}")
print(f"  Output dir   : {cfg['output_dir']}")
print()

# ── Load checkpoint ───────────────────────────────────────────────────────
print("── Loading checkpoint ──────────────────────────────────────")
model, scaler, feat_names = load_checkpoint(cfg['checkpoint'], device)

#Define your directory path instead of a single file
print("\n── Ocean model inference ────────────────────────────────────")
print(f"  Directory : {cfg['nc_path']}")
print(f"  Target    : {cfg['model_year']}-{cfg['model_month']:02d}")
# ################################################
# This commented code is for using a set of netcdf files as input e.g. monthly files such as ECCO
# if not os.path.isdir(cfg['nc_path']):
#     sys.exit(f"[ERROR] Directory not found: {cfg['nc_path']}")

# # Use glob to grab all NetCDF files in the folder
# file_pattern = os.path.join(cfg['nc_path'], f"*.nc")
# file_list = glob.glob(file_pattern)

# if not file_list:
#     sys.exit(f"[ERROR] No NetCDF files found in: {cfg['nc_path']}")

# Open and combine all datasets
# chunks = {'time': 1}
# ds = xr.open_mfdataset(file_list, combine='by_coords', data_vars='minimal', coords='minimal',chunks=chunks)
#
# ################################################
ds = xr.open_dataset(cfg['nc_path'])
# Subset to the target time step before any heavy operations
time_str = f"{cfg['model_year']}-{cfg['model_month']:02d}-01"
ds = ds.sel(time=time_str, method="nearest")
print(
    f"  Nearest time slice selected: "
    f"{str(ds['time'].values)[:10] if 'time' in ds else 'N/A'}"
)
if "time" in ds.dims:
    ds = ds.squeeze("time")


ds_out = infer_on_model_field(
    ds,
    model,
    scaler,
    feat_names,
    target_year=cfg['model_year'],
    target_month=cfg['model_month'],
    device=device,
    n_mc=cfg['mc_samples']
)

# ── Save NetCDF ───────────────────────────────────────────────────────────
nc_out = os.path.join(cfg['output_dir'], f"tracer_predicted_{cfg['model_year']}_{cfg['model_month']:02d}.nc")
ds_out.to_netcdf(nc_out)
print(f"  Saved NetCDF: {nc_out}")

# ── Surface map ───────────────────────────────────────────────────────────
depth_dim = next(
    (d for d in ("depth", "z", "lev", "level", "deptht","Z") if d in ds_out.dims), None
)
sel_kwargs = {depth_dim: 0} if depth_dim else {}

# Extract surface fields
da_pred = ds_out["tracer_pred"].isel(**sel_kwargs)
da_unc = ds_out["tracer_uncertainty"].isel(**sel_kwargs)

# Get coordinate names
lat_name = [c for c in da_pred.coords if "lat" in c.lower()][0]
lon_name = [c for c in da_pred.coords if "lon" in c.lower()][0]

lats = da_pred[lat_name]
lons = da_pred[lon_name]

# Create figure with Arctic projection
proj = ccrs.NorthPolarStereo()
data_crs = ccrs.PlateCarree()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})

for ax in axes:
    ax.set_extent([-180, 180, 50, 90], crs=data_crs)  # Arctic view
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)

# ── Plot predicted tracer ─────────────────────────────────────────────────
pcm1 = axes[0].pcolormesh(
    lons,
    lats,
    da_pred,
    transform=data_crs,
    cmap=cmocean.cm.haline,   # reversed so fresh = light, salty = dark,
    shading="auto",
    vmin=-4,
    vmax=0.5,
)
axes[0].set_title(f"Predicted tracer — {cfg['model_year']}-{cfg['model_month']:02d} (10m depth)")
plt.colorbar(pcm1, ax=axes[0], orientation="horizontal", pad=0.05)

# ── Plot uncertainty ──────────────────────────────────────────────────────
pcm2 = axes[1].pcolormesh(
    lons, lats, da_unc, transform=data_crs, cmap="Oranges", shading="auto"
)
axes[1].set_title("Prediction uncertainty σ (MC-Dropout, 10m depth)")
plt.colorbar(pcm2, ax=axes[1], orientation="horizontal", pad=0.05)

plt.tight_layout()

map_path = os.path.join(cfg['output_dir'], f"model_10m_map_{cfg['model_year']}_{cfg['model_month']:02d}.png")
plt.savefig(map_path, dpi=150)
plt.close()

print(f"  Saved: {map_path}")

# ── Zonal-mean cross-section ───────────────────────────────────────────────
if depth_dim and "latitude" in ds_out.coords:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ds_out["tracer_pred"].where(ds_out[depth_dim] <= 300, drop=True).mean(
        "longitude"
    ).plot(
        ax=axes[0],
        cmap = cmocean.cm.haline,
        yincrease=False,
        vmin=-4,
        vmax=0.5,
    )
    axes[0].set_title("Zonal-mean predicted tracer")
    ds_out["tracer_uncertainty"].where(ds_out[depth_dim] <= 300, drop=True).mean(
        "longitude"
    ).plot(
        ax=axes[1],
        cmap="Oranges",
        yincrease=False,
        # vmin=-4,
        # vmax=0.5,
    )
    axes[1].set_title("Zonal-mean uncertainty σ")
    plt.tight_layout()
    xsec_path = os.path.join(
        cfg['output_dir'], f"model_zonal_mean_{cfg['model_year']}_{cfg['model_month']:02d}.png"
    )
    plt.savefig(xsec_path, dpi=150)
    plt.close()
    print(f"  Saved: {xsec_path}")

print(f"\n{'=' * 60}")
print(f"  All outputs written to: {cfg['output_dir']}/")
print(f"{'=' * 60}\n")
