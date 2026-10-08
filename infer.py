from inference.inference import infer_on_model_field, load_checkpoint
from utilities.utils import load_config,set_seed,log_status
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
import argparse


def main(cfg_path,model_year,model_month):
    log_status(f"Loading config from: {cfg_path}")
    cfg = load_config(path=cfg_path)
    cfg = cfg['inference']
    set_seed(cfg['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg['output_dir'], exist_ok=True)

    log_status(f"\n{'=' * 60}")
    log_status(" PINN Tracer Inference")
    log_status(f"{'=' * 60}")
    log_status(f"  Device       : {device}")
    log_status(f"  Checkpoint   : {cfg['checkpoint']}")
    log_status(f"  Ocean model  : {cfg['nc_path']}")
    log_status(f"  Target time  : {model_year}-{model_month:02d}")
    log_status(f"  MC samples   : {cfg['mc_samples']}")
    log_status(f"  Output dir   : {cfg['output_dir']}")
    log_status(f"  Seed         : {cfg['seed']}")
    log_status("")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log_status("── Loading checkpoint ──────────────────────────────────────")
    model, scaler, feat_names = load_checkpoint(cfg['checkpoint'], device)

    #Define your directory path instead of a single file
    log_status("\n── Ocean model inference ────────────────────────────────────")
    log_status(f"  Directory : {cfg['nc_path']}")
    log_status(f"  Target    : {model_year}-{model_month:02d}")
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
    time_str = f"{model_year}-{model_month:02d}-01"
    ds = ds.sel(time=time_str, method="nearest")
    log_status(
        f"  Nearest time slice selected: "
        f"{str(ds['time'].values)[:10] if 'time' in ds else 'N/A'}"
    )
    if "time" in ds.dims:
        ds = ds.squeeze("time")

    ds_out = infer_on_model_field(
        cfg,
        ds,
        model,
        scaler,
        feat_names,
        target_year=model_year,
        target_month=model_month,
        device=device,
        n_mc=cfg['mc_samples']
    )

    # ── Save NetCDF ───────────────────────────────────────────────────────────
    nc_out = os.path.join(cfg['output_dir'], f"netcdf/tracer_predicted_{model_year}_{model_month:02d}.nc")
    ds_out.to_netcdf(nc_out)
    log_status(f"  Saved NetCDF: {nc_out}")

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
    axes[0].set_title(f"Predicted tracer — {model_year}-{model_month:02d} (10m depth)")
    plt.colorbar(pcm1, ax=axes[0], orientation="horizontal", pad=0.05)

    # ── Plot uncertainty ──────────────────────────────────────────────────────
    pcm2 = axes[1].pcolormesh(
        lons, lats, da_unc, transform=data_crs, cmap="Oranges", shading="auto"
    )
    axes[1].set_title("Prediction uncertainty σ (MC-Dropout, 10m depth)")
    plt.colorbar(pcm2, ax=axes[1], orientation="horizontal", pad=0.05)

    plt.tight_layout()

    map_path = os.path.join(cfg['output_dir'], f"images/model_10m_map_{model_year}_{model_month:02d}.png")
    plt.savefig(map_path, dpi=150)
    plt.close()

    log_status(f"  Saved: {map_path}")

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
            cfg['output_dir'], f"images/model_zonal_mean_{model_year}_{model_month:02d}.png"
        )
        plt.savefig(xsec_path, dpi=150)
        plt.close()
        log_status(f"  Saved: {xsec_path}")

    log_status(f"\n{'=' * 60}")
    log_status(f"  All outputs written to: {cfg['output_dir']}/")
    log_status(f"{'=' * 60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Physics-Informed Neural Network")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2007,
        help="Year to target inference"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=9,
        help="Year to target inference"
    )
    args = parser.parse_args()
    main(cfg_path=args.config,model_year=args.year,model_month=args.month)
