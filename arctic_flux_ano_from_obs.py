import os
# Must be set before netCDF4/h5py/xarray import — HDF5 reads this at library
# init. JASMIN's GWS/scratch filesystems don't reliably support HDF5's native
# file locking, and with parallel=True multiple Dask workers open files
# concurrently, which trips it and raises "Can't open HDF5 attribute".
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
import cmocean
import glob

cfg = {
    "t_name": "Fram_Strait",
    "work_dir":"outputs/anomaly/images_fram",
    "mw_output_path" : "outputs/inventory/netcdf/arctic_meteoric_inventory_FRAM_2005_2023.nc",
    "sim_output_path" : "outputs/inventory/netcdf/arctic_seaicemelt_inventory_FRAM_2005_2023.nc",
    "frac_output_path" : "outputs/inventory/netcdf/arctic_fractions_FRAM_2005_2023.nc",
    "ML_model_dir": "outputs/oxygen18/experiment2",
    "inference_target": "/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN/FramStrait_adjusted_v_fulldepth.nc",
    "baseline_start": "2005",
    "baseline_end": "2015",
    "anomaly_start": "2023",
    "anomaly_end": "2023",
    "salinity": "SA",
    "depth": "depth",
    "tracer_pred": "tracer_pred",
    "time": "time",
    "tracer_uncertainty": "tracer_uncertainty",
}

start_time = time.time()

def log_status(message):
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {message}")


def main():
    # --- Calculate Meteoric Flux through a Davis Strait Transect ---
    log_status("Setting up transects...")
    R = 6371000.0
    # Dedicated Dask cluster for the flux section. This is what was OOM-killing the
    # job: ds_v was opened with no chunks (so xarray/dask picks one big chunk per
    # file) and then .interp() forces the *entire* lat/lon extent of whatever it
    # touches into a single chunk to do the scipy-based interpolation. Under the
    # default threaded scheduler that all happens in the same process with no
    # memory accounting, so a big enough interp step just OOM-kills the whole
    # script. Running it under a distributed LocalCluster with a memory_limit per
    # worker means Dask will spill to disk instead of blowing up the process.
    flux_cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="16GB")
    flux_client = Client(flux_cluster)
    print(f"Dask Dashboard link for flux calc monitoring: {flux_client.dashboard_link}")

    chunks = {"time": 1}
    ds_frac = xr.open_dataset(f"{cfg['frac_output_path']}", chunks=chunks)

    ds_v = xr.open_dataset(cfg['inference_target'],chunks=chunks)

    log_status(f"[{cfg['t_name']}] Extracting grid parameters along diagonal path...")

    # 2D Transect Dataset (e.g., Depth x Lon across FRAM Strait)
    f_sim_trans = ds_frac["f_sim"]
    f_met_trans = ds_frac["f_met"]

    # Velocity is already the normal component across the transect
    v_normal = ds_v["v"]

    # Calculate cell horizontal width (dx) directly from longitude coordinates
    lon_vals = ds_v["lon"].values
    lat_val = ds_v["lat"].values.item()

    # Distance in meters between adjacent longitudes at this latitude
    lon_rad = np.deg2rad(lon_vals)
    dlon = np.abs(np.diff(lon_rad))
    # Pad or construct dx array matching lon dimension
    dlon_padded = np.append(dlon, dlon[-1])
    dx_meters = R * np.cos(np.deg2rad(lat_val)) * dlon_padded

    dx_width = xr.DataArray(dx_meters, coords={"lon": ds_v["lon"]}, dims=["lon"])
    spatial_sum_dims = ["depth", "lon"]

    # Clean duplicates along the time dimension for your input variables
    v_normal = v_normal.drop_duplicates(dim='time')
    f_sim_trans = f_sim_trans.drop_duplicates(dim='time')
    f_met_trans = f_met_trans.drop_duplicates(dim='time')

    #1. Extract depth array from the TOPAZ dataset (typically named 'depth')
    depths = ds_v["depth"].values

    # 2. Compute interface boundaries between adjacent layers
    # Midpoints between consecutive depth levels
    interfaces = (depths[:-1] + depths[1:]) / 2.0

    # Define full boundaries including surface (0) and bottom interface
    bounds = np.concatenate(([0.0], interfaces, [depths[-1] + (depths[-1] - interfaces[-1])]))

    # 3. Calculate layer thicknesses (dz = bottom_bound - top_bound)
    dz_values = np.diff(bounds)

    # 4. Package into an xarray DataArray matching TOPAZ coordinates
    dz2 = xr.DataArray(dz_values, coords={"depth": ds_v["depth"]}, dims=["depth"])

    # 5. Select matching depths for your interpolated transect
    dz_trans = dz2.sel(depth=f_sim_trans["depth"])

    # ── 2. Compute Fluxes ──────────────────────────────────────────────────
    sim_cell_flux_m3s = v_normal * f_sim_trans * dz_trans * dx_width * -1
    met_cell_flux_m3s = v_normal * f_met_trans * dz_trans * dx_width

    # ── 3. Sum across correct spatial dimensions ───────────────────────────
    sim_total_flux_m3s = sim_cell_flux_m3s.sum(dim=spatial_sum_dims)
    met_total_flux_m3s = met_cell_flux_m3s.sum(dim=spatial_sum_dims)

    sim_flux_mSv = sim_total_flux_m3s / 1000.0 * -1
    sim_flux_mSv.name = f"seaicemelt_flux_{cfg['t_name']}"
    sim_flux_mSv.attrs["units"] = "mSv"
    sim_flux_mSv.attrs["description"] = (
        "Seaice melt liquid water volume transport flux in milliSverdrups"
    )

    met_flux_mSv = met_total_flux_m3s / 1000.0 * -1
    met_flux_mSv.name = f"met_flux_{cfg['t_name']}"
    met_flux_mSv.attrs["units"] = "mSv"
    met_flux_mSv.attrs["description"] = (
        "Meteoric melt liquid water volume transport flux in milliSverdrups"
    )



    log_status("Submitting flux graphs to the Dask cluster...")
    (met_computed_fluxes,) = dask.compute(met_flux_mSv)
    (sim_computed_fluxes,) = dask.compute(sim_flux_mSv)
    log_status("Flux pipeline computation completed!")

    # Conversion factor: 1 mSv -> km^3/yr (365 days/year)
    MSV_TO_KM3_YR = 31.536

    # Extract Series and convert units to km^3/yr
    sim_raw = sim_computed_fluxes * MSV_TO_KM3_YR
    met_raw = met_computed_fluxes * MSV_TO_KM3_YR

    # Calculate 12-month rolling means
    # center=True aligns the window symmetrically around each date
    sim_smoothed = sim_raw.rolling(time=12, center=True).mean()
    met_smoothed = met_raw.rolling(time=12, center=True).mean()


    # Define distinct colors
    color_sim = "#1f77b4"  # Blue shade for Sea Ice Melt / Brine
    color_met = "#d62728"  # Red/Coral shade for Meteoric

    plt.figure(figsize=(10, 4), dpi=150)

    # Plot raw unsmoothed data faintly in background
    sim_raw.plot(color=color_sim, alpha=0.25, linewidth=0.8)#, add_legend=False)
    met_raw.plot(color=color_met, alpha=0.25, linewidth=0.8)#, add_legend=False)

    # Plot 12-month smoothed data
    sim_smoothed.plot(
        color=color_sim,
        linewidth=2.0,
        label="Sea Ice Melt (12-mo mean)",
    )
    met_smoothed.plot(
        color=color_met,
        linewidth=2.0,
        label="Meteoric (12-mo mean)",
    )

    # Plot net liquid freshwater sum
    net_smoothed = sim_smoothed + met_smoothed
    net_smoothed.plot(
        color="black",
        linewidth=1.5,
        linestyle=":",
        label="Net Liquid Freshwater",
    )

    def plot_trendline(series, color, linestyle="--", label=None):
        valid_data = series.to_series().dropna()
        if valid_data.empty:
            return None

        # Extract the time index explicitly and cast to DatetimeIndex
        time_index = pd.to_datetime(valid_data.index.get_level_values('time') if isinstance(valid_data.index, pd.MultiIndex) else valid_data.index)

        # Convert to fractional decimal years (e.g. 2004.8)
        x = (time_index.year + (time_index.dayofyear - 1) / 365.25).to_numpy()

        # Fit linear polynomial (slope will be in units/year)
        slope_per_year, intercept = np.polyfit(x, valid_data.values, 1)
        trend_y = slope_per_year * x + intercept

        # Format the legend label
        if label:
            sign = "+" if slope_per_year >= 0 else ""
            label = f"{label} ({sign}{slope_per_year:.2f} km³/yr²)"

        # Plot trend line directly
        plt.plot(
            time_index,
            trend_y,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
        return pd.Series(trend_y, index=time_index)

    # Add trend lines for each series
    trend_sim = plot_trendline(sim_smoothed, color=color_sim, label="Sea Ice Melt Trend")
    trend_met = plot_trendline(met_smoothed, color=color_met, label="Meteoric Trend")
    trend_net = plot_trendline(net_smoothed, color="black", label="Net Liquid Trend")

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(
        f"Freshwater Volume Fluxes through {cfg['t_name']}",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Flux (km3/yr)", fontsize=11)
    plt.xlabel("Year", fontsize=11)
    plt.grid(alpha=0.3)
    # Position the legend below the x-axis
    # bbox_to_anchor=(0.5, -0.22) centers it horizontally below the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,  # Place items side-by-side in 3 columns
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    plt.savefig(f"{cfg['t_name']}_combined_flux_from_obs.png", bbox_inches="tight")
    plt.show()

    flux_client.close()
    flux_cluster.close()


    # --- 1. SET UP THE POLAR PROJECTED MAP ---
    fig2 = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.NorthPolarStereo()})
    ax2 = fig2[1]

    ax2.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

    ax2.add_feature(cfeature.LAND, facecolor="#f4f4f4", edgecolor="gray", zorder=1)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555", zorder=2)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", zorder=2)

    gl = ax2.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=3,
    )
    gl.top_labels = False
    gl.right_labels = False

    ax2.plot(
        [lon_vals[0], lon_vals[-1]],
        [lat_val, lat_val],
        color="#e63946",
        linewidth=3,
        linestyle="-",
        transform=ccrs.Geodetic(),
        zorder=4,
        label="Flux Transect Line",
    )

    ax2.scatter(
        [lon_vals[0], lon_vals[-1]],
        [lat_val, lat_val],
        color="#1d3557",
        s=60,
        edgecolor="white",
        linewidth=1,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    ax2.text(
        lon_vals[0] - 2,
        lat_val + 0.5,
        cfg['t_name'],
        transform=ccrs.PlateCarree(),
        fontsize=12,
        weight="bold",
        zorder=6,
    )

    ax2.set_title(
        f"Transect Locations", fontsize=14, pad=20, weight="bold"
    )
    ax2.legend(loc="lower left", framealpha=0.9)

    plt.savefig("transect_location_map.png", bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
