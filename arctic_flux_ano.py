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
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
import cmocean
import glob

cfg = {
    "work_dir":"outputs/anomaly/images",
    "mw_output_path" : "outputs/inventory/netcdf/arctic_meteoric_inventory_TOPAZ4_1991_2025.nc",
    "sim_output_path" : "outputs/inventory/netcdf/arctic_seaicemelt_inventory_TOPAZ4_1991_2025.nc",
    "frac_output_path" : "outputs/inventory/netcdf/arctic_fractions_TOPAZ4_1991_2025.nc",
    "ML_model_dir": "outputs/oxygen18/experiment1",
    "inference_target": "/work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-04-01.nc",
    "baseline_start": "1991",
    "baseline_end": "2001",
    "anomaly_start": "2023",
    "anomaly_end": "2025",
    "salinity": "so",
    "depth": "depth",
    "tracer_pred": "tracer_pred",
    "time": "time",
    "tracer_uncertainty": "tracer_uncertainty",
}

# # Volume Boxes # NOT implemented yet
# Beaufort_Gyre = {"lats": [70, 70, 80, 80, 70], "lons":  [-160, -130, -130, -160, -160] }
# Arctic = {}

# Flux Transects
transects = {
"Davis_Strait" : {'lat1':66.665,"lon1":-61.646,"lat2":67.117,"lon2":-53.641,"num_points": 50},
"N_Baffin_Bay" : {'lat1':73.447,"lon1":-77.783,"lat2":76.448,"lon2":-68.575,"num_points": 50},
"Fram_Strait" : {'lat1':81.432,"lon1":-12.256,"lat2":79.551,"lon2":11.428,"num_points": 50},
"Bering_Strait" : {'lat1':67.866,"lon1":-164.361,"lat2":66.337,"lon2":-171.163,"num_points": 50},
"Barents_Sea": {'lat1':70.245,"lon1":20.724,"lat2":76.622,"lon2":16.443,"num_points": 50},
}

start_time = time.time()

def log_status(message):
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {message}")


def main():
    # --- 7. PLOTTING ANOMALY MAP ---

    log_status("Loading generated inventory file for plotting...")
    ds_map = xr.open_dataset(f"{cfg['mw_output_path']}")

    log_status("Computing baseline vs recent climatology values and uncertainties...")

    # 1. Slice your inventory value and uncertainty data arrays
    mw_baseline_data = ds_map["meteoric_freshwater_thickness"].sel(
        time=slice(cfg['baseline_start'], cfg['baseline_end'])
    )
    mw_baseline_unc = ds_map["meteoric_freshwater_thickness_uncertainty"].sel(
        time=slice(cfg['baseline_start'], cfg['baseline_end'])
    )

    mw_recent_data = ds_map["meteoric_freshwater_thickness"].sel(
        time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
    )
    mw_recent_unc = ds_map["meteoric_freshwater_thickness_uncertainty"].sel(
        time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
    )

    # 2. Compute the standard means for the values
    baseline = mw_baseline_data.mean(dim=cfg["time"])
    recent = mw_recent_data.mean(dim=cfg["time"])
    mw_anomaly = recent - baseline

    # 3. Propagate uncertainty through the TIME MEAN
    # Formula: sqrt(sum(unc^2)) / N
    N_baseline = mw_baseline_unc.sizes[cfg["time"]]
    baseline_uncertainty = np.sqrt((mw_baseline_unc**2).sum(dim=cfg["time"])) / N_baseline

    N_recent = mw_recent_unc.sizes[cfg["time"]]
    recent_uncertainty = np.sqrt((mw_recent_unc**2).sum(dim=cfg["time"])) / N_recent

    # 4. Propagate uncertainty through the SUBTRACTION (recent - baseline)
    # Formula: sqrt(unc_recent^2 + unc_baseline^2)
    mw_anomaly_uncertainty = np.sqrt(recent_uncertainty**2 + baseline_uncertainty**2)

    log_status("Generating Polar Stereographic Plot...")
    projection = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.axes(projection=projection)
    ax.set_extent([-180, 180, 50, 90], crs=data_crs)

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    # --- Draw the Beaufort Gyre Bounding Box ---
    box_lons = [-160, -130, -130, -160, -160]
    box_lats = [70, 70, 80, 80, 70]

    ax.plot(
        box_lons,
        box_lats,
        color="black",
        linewidth=2.0,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=3,
        label="Beaufort Gyre Domain"
    )

    vmax = 1.0
    im = mw_anomaly.plot.pcolormesh(
        ax=ax,
        x="longitude",
        y="latitude",
        transform=data_crs,
        cmap=cmocean.cm.balance,
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=False,
        zorder=1,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both"
    )
    cbar.set_label("Meteoric Freshwater Inventory Anomaly (meters)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.title(
        f"Arctic Meteoric Freshwater Inventory Anomaly\n(Mean {cfg['anomaly_start']}/{cfg['anomaly_end']} minus Baseline {cfg['baseline_start']}–{cfg['baseline_end']})",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plot_output = f"{cfg['work_dir']}/arctic_meteoric_anomaly_map.png"
    plt.savefig(plot_output, bbox_inches="tight")
    plt.show()
    log_status(f"Map plotted and saved to: {plot_output}")

    ##################################
    # --- 7. PLOTTING ANOMALY MAP ---
    ##################################
    log_status("Loading generated inventory file for plotting...")
    ds_map = xr.open_dataset(f"{cfg['sim_output_path']}")

    log_status("Computing baseline vs recent climatology values and uncertainties...")

    sim_baseline_data = ds_map["seaicemelt_freshwater_thickness"].sel(
        time=slice(cfg['baseline_start'], cfg['baseline_end'])
    )
    sim_baseline_unc = ds_map["seaicemelt_freshwater_thickness_uncertainty"].sel(
        time=slice(cfg['baseline_start'], cfg['baseline_end'])
    )

    sim_recent_data = ds_map["seaicemelt_freshwater_thickness"].sel(
        time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
    )
    sim_recent_unc = ds_map["seaicemelt_freshwater_thickness_uncertainty"].sel(
        time=slice(cfg['anomaly_start'], cfg['anomaly_end'])
    )

    baseline = sim_baseline_data.mean(dim="time")
    recent = sim_recent_data.mean(dim="time")
    sim_anomaly = recent - baseline

    N_baseline = sim_baseline_unc.sizes["time"]
    baseline_uncertainty = np.sqrt((sim_baseline_unc**2).sum(dim="time")) / N_baseline

    N_recent = sim_recent_unc.sizes["time"]
    recent_uncertainty = np.sqrt((sim_recent_unc**2).sum(dim="time")) / N_recent

    sim_anomaly_uncertainty = np.sqrt(recent_uncertainty**2 + baseline_uncertainty**2)

    log_status("Generating Polar Stereographic Plot...")
    projection = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.axes(projection=projection)
    ax.set_extent([-180, 180, 50, 90], crs=data_crs)

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    box_lons = [-160, -130, -130, -160, -160]
    box_lats = [70, 70, 80, 80, 70]

    ax.plot(
        box_lons,
        box_lats,
        color="black",
        linewidth=2.0,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=3,
        label="Beaufort Gyre Domain"
    )

    vmax = 2.0
    im = sim_anomaly.plot.pcolormesh(
        ax=ax,
        x="longitude",
        y="latitude",
        transform=data_crs,
        cmap=cmocean.cm.balance,
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=False,
        zorder=1,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both"
    )
    cbar.set_label("SeaIce Melt Freshwater Inventory Anomaly (meters)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.title(
        f"Arctic Sea Ice Melt Freshwater Inventory Anomaly\n Mean {cfg['anomaly_start']}/{cfg['anomaly_end']} minus Baseline {cfg['baseline_start']}–{cfg['baseline_end']})",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plot_output = f"{cfg['work_dir']}/arctic_seaicemelt_anomaly_map.png"
    plt.savefig(plot_output, bbox_inches="tight")
    plt.show()
    log_status(f"Map plotted and saved to: {plot_output}")
    log_status("All tasks successfully completed!")

    # ==============================================================================
    # 1. METEORIC WATER VOLUME & UNCERTAINTY PROPAGATION
    # ==============================================================================
    log_status("Calculating grid cell areas for Meteoric volume conversion...")

    R = 6371000.0
    lats = np.deg2rad(mw_anomaly["latitude"].values)
    lons = np.deg2rad(mw_anomaly["longitude"].values)
    dlat = np.abs(np.diff(lats)[0])
    dlon = np.abs(np.diff(lons)[0])

    lat_mesh, _ = np.meshgrid(
        mw_anomaly["latitude"].values, mw_anomaly["longitude"].values, indexing="ij"
    )
    cell_areas_m2 = (R**2) * np.cos(np.deg2rad(lat_mesh)) * dlat * dlon

    da_area = xr.DataArray(
        cell_areas_m2,
        coords=[mw_anomaly["latitude"], mw_anomaly["longitude"]],
        dims=["latitude", "longitude"],
    )

    volume_km3_grid = (mw_anomaly * da_area) / 1e9
    volume_uncertainty_km3_grid = (mw_anomaly_uncertainty * da_area) / 1e9

    bg_volume = volume_km3_grid.sel(
        latitude=slice(70, 80), longitude=slice(-160, -130)
    )
    bg_volume_uncertainty = volume_uncertainty_km3_grid.sel(
        latitude=slice(70, 80), longitude=slice(-160, -130)
    )

    total_bg_mw_increase = bg_volume.sum().values
    total_bg_mw_increase_uncertainty = np.sqrt(
        (bg_volume_uncertainty**2).sum()
    ).values

    print(f"--- Meteoric Volumetric Analysis ---")
    print(
        f"Total Net Meteoric Freshwater Accumulated in Beaufort Gyre: {total_bg_mw_increase:.3f} ± {total_bg_mw_increase_uncertainty:.3f} km³"
    )

    arctic_volume = volume_km3_grid.sel(latitude=slice(60, 90))
    arctic_volume_uncertainty = volume_uncertainty_km3_grid.sel(
        latitude=slice(60, 90))

    total_arctic_mw_increase = arctic_volume.sum().values
    total_arctic_mw_increase_uncertainty = np.sqrt(
        (arctic_volume_uncertainty**2).sum()
    ).values

    print(
        f"Total Arctic-wide (North of 60°N) Meteoric Freshwater Change: {total_arctic_mw_increase:.3f} ± {total_arctic_mw_increase_uncertainty:.3f} km³"
    )


    # ==============================================================================
    # 2. SEA ICE MELT VOLUME & UNCERTAINTY PROPAGATION
    # ==============================================================================
    log_status("Calculating grid cell areas for Sea Ice Melt volume conversion...")

    lats = np.deg2rad(sim_anomaly["latitude"].values)
    lons = np.deg2rad(sim_anomaly["longitude"].values)
    dlat = np.abs(np.diff(lats)[0])
    dlon = np.abs(np.diff(lons)[0])

    lat_mesh, _ = np.meshgrid(
        sim_anomaly["latitude"].values, sim_anomaly["longitude"].values, indexing="ij"
    )
    cell_areas_m2 = (R**2) * np.cos(np.deg2rad(lat_mesh)) * dlat * dlon

    da_area = xr.DataArray(
        cell_areas_m2,
        coords=[sim_anomaly["latitude"], sim_anomaly["longitude"]],
        dims=["latitude", "longitude"],
    )

    volume_km3_grid = (sim_anomaly * da_area) / 1e9
    volume_uncertainty_km3_grid = (sim_anomaly_uncertainty * da_area) / 1e9

    bg_volume = volume_km3_grid.sel(
        latitude=slice(70, 80), longitude=slice(-160, -130)
    )
    bg_volume_uncertainty = volume_uncertainty_km3_grid.sel(
        latitude=slice(70, 80), longitude=slice(-160, -130)
    )

    total_bg_sim_increase = bg_volume.sum().values
    total_bg_sim_increase_uncertainty = np.sqrt(
        (bg_volume_uncertainty**2).sum()
    ).values

    print(f"\n--- Sea Ice Melt Volumetric Analysis ---")
    print(
        f"Total Net Sea Ice Melt Freshwater Accumulated in Beaufort Gyre: {total_bg_sim_increase:.3f} ± {total_bg_sim_increase_uncertainty:.3f} km³"
    )

    arctic_volume = volume_km3_grid.sel(latitude=slice(60, 90))
    arctic_volume_uncertainty = volume_uncertainty_km3_grid.sel(
        latitude=slice(60, 90))

    total_arctic_sim_increase = arctic_volume.sum().values
    total_arctic_sim_increase_uncertainty = np.sqrt(
        (arctic_volume_uncertainty**2).sum()
    ).values

    print(
        f"Total Arctic-wide (North of 60°N) Sea Ice Melt Freshwater Change: {total_arctic_sim_increase:.3f} ± {total_arctic_sim_increase_uncertainty:.3f} km³"
    )

    # --- Calculate Meteoric Flux through a Davis Strait Transect ---
    log_status("Setting up transects...")

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
    # ecco_paths2 = glob.glob(os.path.join(cfg['inference_target'], "*.nc"))
    # log_status("Loading ECCO v4r4 dataset lazily...")

    # ds_v = xr.open_mfdataset(
    #     ecco_paths2, combine='by_coords', data_vars='minimal', coords='minimal',
    #     compat='override', join='override', chunks=chunks,
    # )
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

        ds_frac_sub = ds_frac.sel(latitude=slice(lat_lo, lat_hi), longitude=slice(lon_lo, lon_hi))
        ds_v_sub = ds_v.sel(latitude=slice(lat_lo, lat_hi), longitude=slice(lon_lo, lon_hi))

        log_status(f"[{t_name}] Extracting grid parameters along diagonal path...")

        f_sim_trans = ds_frac_sub["f_sim"].interp(latitude=da_lats, longitude=da_lons)
        f_met_trans = ds_frac_sub["f_met"].interp(latitude=da_lats, longitude=da_lons)
        vxo_trans = ds_v_sub["vxo"].interp(latitude=da_lats, longitude=da_lons)
        vyo_trans = ds_v_sub["vyo"].interp(latitude=da_lats, longitude=da_lons)

        # # 1. Compute exact layer thicknesses (dz) from Z_bnds
        # # Z_bnds has shape (len(Z), 2) for lower and upper layer interfaces
        # dz_values = np.abs(ds_v_sub["Z_bnds"][:, 1] - ds_v_sub["Z_bnds"][:, 0]).values

        # # 2. Package into a DataArray aligned with coordinate Z
        # dz2 = xr.DataArray(dz_values, coords={"Z": ds_v_sub["Z"]}, dims=["Z"])

        # # 3. Select matching depths for your transect
        # dz_trans = dz2.sel(Z=f_sim_trans["Z"])

        # 1. Extract depth array from the TOPAZ dataset (typically named 'depth')
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

        log_status(f"[{t_name}] Projecting velocity vectors perpendicular to transect...")

        v_normal = (-vxo_trans * da_dy + vyo_trans * da_dx) / da_ds

        log_status(f"[{t_name}] Integrating total column flux...")

        sim_cell_flux_m3s = v_normal * f_sim_trans * dz_trans * da_ds * -1

        sim_total_flux_m3s = sim_cell_flux_m3s.sum(dim=["depth", "segment"])
        sim_flux_mSv = sim_total_flux_m3s / 1000.0 * -1
        sim_flux_mSv.name = f"seaicemelt_flux_{t_name}"
        sim_flux_mSv.attrs["units"] = "mSv"
        sim_flux_mSv.attrs["description"] = (
            "Seaice melt liquid water volume transport flux in milliSverdrups"
        )

        met_cell_flux_m3s = v_normal * f_met_trans * dz_trans * da_ds

        met_total_flux_m3s = met_cell_flux_m3s.sum(dim=["depth", "segment"])
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
        sim_raw.plot(color=color_sim, alpha=0.25, linewidth=0.8, add_legend=False)
        met_raw.plot(color=color_met, alpha=0.25, linewidth=0.8, add_legend=False)

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

        plt.savefig(f"{t_name}_combined_flux.png", bbox_inches="tight")
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
