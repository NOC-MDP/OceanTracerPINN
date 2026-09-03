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
    "work_dir":"outputs/anomaly/images_fram",
    "mw_output_path" : "outputs/inventory/netcdf/arctic_meteoric_inventory_FRAM_2004_2024.nc",
    "sim_output_path" : "outputs/inventory/netcdf/arctic_seaicemelt_inventory_FRAM_2004_2024.nc",
    "frac_output_path" : "outputs/inventory/netcdf/arctic_fractions_FRAM_2004_2024.nc",
    "ML_model_dir": "outputs/oxygen18/experiment2",
    "inference_target": "/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN/FramStrait_adjusted_v_fulldepth.nc",
    "baseline_start": "2004",
    "baseline_end": "2014",
    "anomaly_start": "2024",
    "anomaly_end": "2024",
    "salinity": "SA",
    "depth": "depth",
    "tracer_pred": "tracer_pred",
    "time": "time",
    "tracer_uncertainty": "tracer_uncertainty",
}

# Flux Transects
transects = {
# "Davis_Strait" : {'lat1':66.665,"lon1":-61.646,"lat2":67.117,"lon2":-53.641,"num_points": 50},
# "N_Baffin_Bay" : {'lat1':73.447,"lon1":-77.783,"lat2":76.448,"lon2":-68.575,"num_points": 50},
"Fram_Strait" : {'lat1':78.83125,"lon1":-20.6,"lat2":78.83125,"lon2":11.9,"num_points": 50},
# "Bering_Strait" : {'lat1':67.866,"lon1":-164.361,"lat2":66.337,"lon2":-171.163,"num_points": 50},
# "Barents_Sea": {'lat1':70.245,"lon1":20.724,"lat2":76.622,"lon2":16.443,"num_points": 50},
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

    # How far (in degrees) to pad around each transect's lat/lon box before
    # interpolating. Subsetting to this small window first means the "single
    # chunk" interp() forces onto the interpolation dims is small, rather than
    # spanning the whole Arctic domain — this is the actual fix for the memory
    # blow-up, and the cluster's memory_limit/spill is the safety net behind it.
    transect_pad_deg = 2.0

    log_status("Setting up diagonal transect calculation...")
    sim_flux_results = {}
    met_flux_results = {}
    for t_name, transect in transects.items():
        transect_lats = np.linspace(transect['lat1'], transect['lat2'], transect['num_points'])
        transect_lons = np.linspace(transect['lon1'], transect['lon2'], transect['num_points'])

        lat_rad = np.deg2rad(transect_lats)
        lon_rad = np.deg2rad(transect_lons)

        dlat_trans = np.diff(lat_rad)
        dlon_trans = np.diff(lon_rad)
        mid_lats = (lat_rad[:-1] + lat_rad[1:]) / 2.0

        dx = R * np.cos(mid_lats) * dlon_trans
        dy = R * dlat_trans
        ds_segment = np.sqrt(dx**2 + dy**2)

        sample_lats = np.rad2deg(mid_lats)
        sample_lons = np.rad2deg((lon_rad[:-1] + lon_rad[1:]) / 2.0)

        da_lats = xr.DataArray(sample_lats, dims="segment")
        da_lons = xr.DataArray(sample_lons, dims="segment")
        da_ds = xr.DataArray(ds_segment, dims="segment")
        da_dx = xr.DataArray(dx, dims="segment")
        da_dy = xr.DataArray(dy, dims="segment")

        log_status(f"[{t_name}] Subsetting to a padded bounding box around the transect...")

        lat_lo = min(sample_lats.min(), transect['lat1'], transect['lat2']) - transect_pad_deg
        lat_hi = max(sample_lats.max(), transect['lat1'], transect['lat2']) + transect_pad_deg
        lon_lo = min(sample_lons.min(), transect['lon1'], transect['lon2']) - transect_pad_deg
        lon_hi = max(sample_lons.max(), transect['lon1'], transect['lon2']) + transect_pad_deg

        ds_frac_sub = ds_frac#.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        ds_v_sub = ds_v#.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))

        log_status(f"[{t_name}] Extracting grid parameters along diagonal path...")

        # ── 1. Handle Interpolation vs Direct Selection ──────────────────────────
        if "lat" in ds_frac.dims and ds_frac.sizes["lat"] > 1:
            # 3D Model Grid: Interpolate to diagonal segment points
            f_sim_trans = ds_frac_sub["f_sim"].interp(lat=da_lats, lon=da_lons)
            f_met_trans = ds_frac_sub["f_met"].interp(lat=da_lats, lon=da_lons)

            # Extract velocities and project normal vector
            vyo_trans = ds_v_sub["v"].interp(lat=da_lats, lon=da_lons)
            vxo_trans = ds_v_sub["u"].interp(lat=da_lats, lon=da_lons) if "u" in ds_v_sub else xr.zeros_like(vyo_trans)
            v_normal = (-vxo_trans * da_dy + vyo_trans * da_dx) / da_ds

            # Width of each segment
            dx_width = da_ds
            spatial_sum_dims = ["depth", "segment"]

        else:
            # 2D Transect Dataset (e.g., Depth x Lon across FRAM Strait)
            f_sim_trans = ds_frac_sub["f_sim"]
            f_met_trans = ds_frac_sub["f_met"]

            # Velocity is already the normal component across the transect
            v_normal = ds_v_sub["v"]

            # Calculate cell horizontal width (dx) directly from longitude coordinates
            lon_vals = ds_v_sub["lon"].values
            lat_val = ds_v_sub["lat"].values.item() if "lat" in ds_v_sub.coords else transect['lat1']

            # Distance in meters between adjacent longitudes at this latitude
            lon_rad = np.deg2rad(lon_vals)
            dlon = np.abs(np.diff(lon_rad))
            # Pad or construct dx array matching lon dimension
            dlon_padded = np.append(dlon, dlon[-1])
            dx_meters = R * np.cos(np.deg2rad(lat_val)) * dlon_padded

            dx_width = xr.DataArray(dx_meters, coords={"lon": ds_v_sub["lon"]}, dims=["lon"])
            spatial_sum_dims = ["depth", "lon"]

        # Clean duplicates along the time dimension for your input variables
        v_normal = v_normal.drop_duplicates(dim='time')
        f_sim_trans = f_sim_trans.drop_duplicates(dim='time')
        f_met_trans = f_met_trans.drop_duplicates(dim='time')

        #1. Extract depth array from the TOPAZ dataset (typically named 'depth')
        depths = ds_v_sub["depth"].values

        # 2. Compute interface boundaries between adjacent layers
        # Midpoints between consecutive depth levels
        interfaces = (depths[:-1] + depths[1:]) / 2.0

        # Define full boundaries including surface (0) and bottom interface
        bounds = np.concatenate(([0.0], interfaces, [depths[-1] + (depths[-1] - interfaces[-1])]))

        # 3. Calculate layer thicknesses (dz = bottom_bound - top_bound)
        dz_values = np.diff(bounds)

        # 4. Package into an xarray DataArray matching TOPAZ coordinates
        dz2 = xr.DataArray(dz_values, coords={"depth": ds_v_sub["depth"]}, dims=["depth"])

        # 5. Select matching depths for your interpolated transect
        dz_trans = dz2.sel(depth=f_sim_trans["depth"])

        # ── 2. Compute Fluxes ──────────────────────────────────────────────────
        sim_cell_flux_m3s = v_normal * f_sim_trans * dz_trans * dx_width * -1
        met_cell_flux_m3s = v_normal * f_met_trans * dz_trans * dx_width

        # ── 3. Sum across correct spatial dimensions ───────────────────────────
        sim_total_flux_m3s = sim_cell_flux_m3s.sum(dim=spatial_sum_dims)
        met_total_flux_m3s = met_cell_flux_m3s.sum(dim=spatial_sum_dims)

        sim_flux_mSv = sim_total_flux_m3s / 1000.0 * -1
        sim_flux_mSv.name = f"seaicemelt_flux_{t_name}"
        sim_flux_mSv.attrs["units"] = "mSv"
        sim_flux_mSv.attrs["description"] = (
            "Seaice melt liquid water volume transport flux in milliSverdrups"
        )

        met_flux_mSv = met_total_flux_m3s / 1000.0 * -1
        met_flux_mSv.name = f"met_flux_{t_name}"
        met_flux_mSv.attrs["units"] = "mSv"
        met_flux_mSv.attrs["description"] = (
            "Meteoric melt liquid water volume transport flux in milliSverdrups"
        )

        # Graph is still lazy at this point — stash it and submit everything to
        # the cluster together so the workers can process transects in parallel
        # rather than the loop computing (via .plot()) one at a time.
        sim_flux_results[t_name] = sim_flux_mSv
        met_flux_results[t_name] = met_flux_mSv

    log_status("Submitting flux graphs to the Dask cluster...")
    (met_computed_fluxes,) = dask.compute(met_flux_results)
    (sim_computed_fluxes,) = dask.compute(sim_flux_results)
    log_status("Flux pipeline computation completed!")

    # Conversion factor: 1 mSv -> km^3/yr (365 days/year)
    MSV_TO_KM3_YR = 31.536
    # Loop over each transect assuming matching keys in both dictionaries
    for t_name in sim_computed_fluxes.keys():
        log_status(f"[{t_name}] Plotting combined flux time series...")

        # Extract Series and convert units to km^3/yr
        sim_raw = sim_computed_fluxes[t_name] * MSV_TO_KM3_YR
        met_raw = met_computed_fluxes[t_name] * MSV_TO_KM3_YR

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
            f"Freshwater Volume Fluxes through {t_name}",
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

        plt.savefig(f"{t_name}_combined_flux_from_obs.png", bbox_inches="tight")
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

    for t_name, transect in transects.items():

        ax2.plot(
            [transect['lon1'], transect['lon2']],
            [transect['lat1'], transect['lat2']],
            color="#e63946",
            linewidth=3,
            linestyle="-",
            transform=ccrs.Geodetic(),
            zorder=4,
            label="Flux Transect Line",
        )

        ax2.scatter(
            [transect['lon1'], transect['lon2']],
            [transect['lat1'], transect['lat2']],
            color="#1d3557",
            s=60,
            edgecolor="white",
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        ax2.text(
            transect['lon1'] - 2,
            transect['lat1'] + 0.5,
            t_name,
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
