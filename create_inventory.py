import os
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
    "work_dir":"/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN",
    "mw_output_path" : "outputs/inventory/netcdf/arctic_meteoric_inventory_FRAM_2004_2024.nc",
    "sim_output_path" : "outputs/inventory/netcdf/arctic_seaicemelt_inventory_FRAM_2004_2024.nc",
    "frac_output_path" : "outputs/inventory/netcdf/arctic_fractions_FRAM_2004_2024.nc",
    "ML_model_dir": "outputs/inference_fram/netcdf",
    "inventory_start": 2004,
    "inventory_end": 2024,
    "inference_target": "/gws/ssde/j25a/nemo/vol4/thopri/OceanTracerPINN/FramStrait_adjusted_v_fulldepth.nc",
    "salinity": "SA",
    "depth": "depth",
    "tracer_pred": "tracer_pred",
    "time": "time",
    "tracer_uncertainty": "tracer_uncertainty",
}
# Masks
Baltic = {"lat_min":54.0,"lat_max":66.0,"lon_min":10.0,"lon_max":30.0}
Pacific = {"lat_max":65.0,"lon_min":-120.0,"lon_max":140.0}

# Mass Balance
MassBal = {
    "S_mar" : 34.8,
    "delta_mar" : 0.3,
    "S_met" : 0.0,
    "delta_met" : -20.0,
    "S_sim" : 4.0,
    "delta_sim" : -2.0,
    "S_ref": 34.8,
}

start_time = time.time()

def log_status(message):
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {message}")


def main():
    log_status(f"creating invectory based on this configuration:")
    log_status(f"{cfg}")
    # --- 0. INITIALIZE DASK CLUSTER ---
    log_status("Initializing Dask cluster and client...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="16GB")
    client = Client(cluster)
    print(f"Dask Dashboard link for monitoring: {client.dashboard_link}")

    # --- 1. SET UP PATHS ---
    log_status("Generating file paths...")
    start_year = cfg['inventory_start']
    end_year = cfg['inventory_end']
    model_years = [year for year in range(start_year, end_year + 1)]
    paths = []
    # ecco_paths = []
    for model_year in model_years:
        for i in range(12):
            paths.append(
                f"{cfg['work_dir']}/{cfg['ML_model_dir']}/tracer_predicted_{model_year}_{i + 1:02d}.nc"
            )


    # --- 2. LOAD DATASETS WITH CHUNKS ---
    chunks = {"time": 1}
    # ecco_paths = ecco_paths + glob.glob(os.path.join(cfg['inference_target'], "*.nc"))
    log_status("Loading dataset lazily...")
    # Use glob to grab all NetCDF files in the folder

    # if not file_list:
    #     sys.exit(f"[ERROR] No NetCDF files found in: {nc_path}")

    # Open and combine all datasets
    # data_vars='minimal' merges variables that don't share all dimensions without bloating memory
    # coords='minimal' does the same for coordinates
    # ds_s_raw = xr.open_mfdataset(
    #     ecco_paths, combine='by_coords', data_vars='minimal', coords='minimal',
    #     compat='override', join='override',chunks=chunks
    # )
    ds_s_raw = xr.open_dataset(cfg['inference_target'],
        chunks=chunks,
    )

    log_status(
        f"Loading {len(paths)} PINN predicted tracer files via open_mfdataset..."
    )
    ds_o = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", chunks=chunks
    )

    log_status("Aligning timelines between physics and tracer arrays...")
    ds_s = ds_s_raw.sel(time=ds_o[cfg['time']], method="nearest")

    log_status("Masking out the Baltic Sea and Pacific Ocean regions...")

    # Create Baltic Mask
    is_baltic = (
        (ds_s["lat"] >= Baltic["lat_min"])
        & (ds_s["lat"] <= Baltic['lat_max'])
        & (ds_s["lon"] >= Baltic['lon_min'])
        & (ds_s["lon"] <= Baltic['lon_max'])
    )

    # 2. Define the boundary for the North Pacific / Bering Sea
    # Bering Strait is at roughly 66°N. We mask everything south of 66°N
    # in the Pacific longitude sector (roughly 100°E to 100°W / -100° to 180° & -180° to -100°)
    pacific_lat_max = Pacific['lat_max']

    # In a -180 to 180 longitude setup, the North Pacific covers:
    # From 140°E (which is 140) to -120°W (which is -120)
    is_pacific = (ds_s["lat"] < pacific_lat_max) & (
        (ds_s["lon"] >= Pacific['lon_max']) | (ds_s["lon"] <= Pacific['lon_min'])
    )

    # 3. Combine masks: We want data where it is NOT Baltic and NOT Pacific
    valid_ocean_mask = ~(is_baltic | is_pacific)

    # Apply the combined mask to your input variables
    ds_s[cfg['salinity']] = ds_s[cfg['salinity']].where(valid_ocean_mask)
    ds_o[cfg["tracer_pred"]] = ds_o[cfg["tracer_pred"]].where(valid_ocean_mask)


    ds_o[cfg['tracer_uncertainty']] = ds_o[cfg['tracer_uncertainty']].where(valid_ocean_mask)

    # --- 3. MASS BALANCE & ERROR PROPAGATION (Lazy) ---
    log_status("Building mass balance and error propagation graphs...")

    denom = (MassBal['S_mar'] - MassBal['S_sim']) * (MassBal['delta_mar'] - MassBal['delta_met']) \
        - (MassBal['S_mar'] - MassBal['S_met']) * (MassBal['delta_mar'] - MassBal['delta_sim'])

    ds = xr.Dataset()

    # Base fractions
    ds["f_met"] = (
        (ds_s[cfg['salinity']] - MassBal['S_mar']) * (MassBal['delta_mar'] - MassBal['delta_sim'])
        - (ds_o["tracer_pred"] - MassBal['delta_mar']) * (MassBal['S_mar'] - MassBal['S_sim'])
    ) / denom
    ds["f_sim"] = (
        (ds_o["tracer_pred"] - MassBal['delta_mar']) * (MassBal['S_mar'] - MassBal['S_met'])
        - (ds_s[cfg['salinity']] - MassBal['S_mar']) * (MassBal['delta_mar'] - MassBal['delta_met'])
    ) / denom

    # Propagated Fraction Uncertainties (Analytical Derivatives * Tracer Uncertainty)
    ds["f_met_unc"] = np.abs(-(MassBal['S_mar'] - MassBal['S_sim']) / denom) * ds_o[cfg['tracer_uncertainty']]
    ds["f_sim_unc"] = np.abs((MassBal['S_mar'] - MassBal['S_met']) / denom) * ds_o[cfg['tracer_uncertainty']]

    # --- 4. PREPARE VERTICAL INTEGRATION ---
    log_status("Calculating depth layer thicknesses (dz)...")

    depths = ds_s[cfg['depth']].values
    bounds = np.zeros(len(depths) + 1)

    # 1. Calculate the midpoints between layers (this works regardless of orientation)
    bounds[1:-1] = (depths[:-1] + depths[1:]) / 2

    # 2. Determine orientation and set the boundary conditions
    if depths[0] > depths[-1]:
        # Case A: Sorted Surface-to-Bottom (e.g., 0, -10, -50, -100)
        bounds[0] = 0  # Surface is at the start
        # Extrapolate the bottom boundary downwards
        bounds[-1] = depths[-1] + (depths[-1] - bounds[-2])
    else:
        # Case B: Sorted Bottom-to-Surface (e.g., -100, -50, -10, 0)
        bounds[-1] = 0  # Surface is at the end
        # Extrapolate the bottom boundary downwards
        bounds[0] = depths[0] - (bounds[1] - depths[0])

    # 3. Calculate thickness and force it to be positive using np.abs
    dz_values = np.abs(np.diff(bounds))

    # 4. Wrap back into xarray DataArray
    dz = xr.DataArray(dz_values, coords=[ds_s[cfg['depth']]], dims=[cfg['depth']])


    valid_mask = (ds_s[cfg['salinity']] < MassBal['S_ref']) & (ds_s[cfg['depth']].notnull())
    fw_weight = (MassBal['S_ref'] - ds_s[cfg['salinity']]) / MassBal['S_ref']

    # Core Value Components
    meteoric_fw_fraction = fw_weight * ds["f_met"]
    seaicemelt_fw_fraction = fw_weight * ds["f_sim"]

    # Uncertainty Components (per layer)
    met_layer_unc = fw_weight * ds["f_met_unc"] * dz
    sim_layer_unc = fw_weight * ds["f_sim_unc"] * dz

    log_status("Assembling vertical integration and error graphs...")
    # Standard Integration for inventories
    meteoric_inventory = (meteoric_fw_fraction * dz).where(valid_mask).sum(dim=cfg['depth'])
    seaicemelt_inventory = (
        (seaicemelt_fw_fraction * dz).where(valid_mask).sum(dim=cfg['depth'])
    )

    # Root-Sum-of-Squares Integration for uncertainties
    meteoric_inventory_unc = (
        np.sqrt((met_layer_unc**2).where(valid_mask).sum(dim=cfg['depth']))
    )
    seaicemelt_inventory_unc = (
        np.sqrt((sim_layer_unc**2).where(valid_mask).sum(dim=cfg['depth']))
    )

    # --- 5. COMBINE FIELDS INTO FINAL DATASETS ---
    mw_ds = xr.Dataset(
        {
            "meteoric_freshwater_thickness": meteoric_inventory,
            "meteoric_freshwater_thickness_uncertainty": meteoric_inventory_unc,
        }
    )
    mw_ds["meteoric_freshwater_thickness"].attrs = {
        "units": "meters",
        "description": f"Meteoric liquid freshwater inventory integrated down to Salinity = {MassBal['S_ref']}",
    }
    mw_ds["meteoric_freshwater_thickness_uncertainty"].attrs = {
        "units": "meters",
        "description": f"Propagated 1-sigma uncertainty for meteoric inventory",
    }

    sim_ds = xr.Dataset(
        {
            "seaicemelt_freshwater_thickness": seaicemelt_inventory,
            "seaicemelt_freshwater_thickness_uncertainty": seaicemelt_inventory_unc,
        }
    )
    sim_ds["seaicemelt_freshwater_thickness"].attrs = {
        "units": "meters",
        "description": f"Sea Ice Melt liquid freshwater inventory integrated down to Salinity = {MassBal['S_ref']}",
    }
    sim_ds["seaicemelt_freshwater_thickness_uncertainty"].attrs = {
        "units": "meters",
        "description": f"Propagated 1-sigma uncertainty for sea ice melt inventory",
    }

    # --- 6. TRIGGER COMPUTATION VIA DASK ---
    log_status("Handing execution over to workers...")
    mw_ds = mw_ds.compute()
    sim_ds = sim_ds.compute()
    log_status("Dask computation completed successfully!")

    # --- 7. SAVE TO DISK ---
    log_status("Saving computation to disk...")
    ds.to_netcdf(f"{cfg['work_dir']}/{cfg['frac_output_path']}")
    log_status(f"Saved meteoric and seaicemet fractions to: {cfg['frac_output_path']}")
    mw_ds.to_netcdf(f"{cfg['work_dir']}/{cfg['mw_output_path']}")
    log_status(f"Saved meteoric dataset with uncertainties to: {cfg['mw_output_path']}")
    sim_ds.to_netcdf(f"{cfg['work_dir']}/{cfg['sim_output_path']}")
    log_status(
        f"Saved sea ice melt dataset with uncertainties to: {cfg['sim_output_path']}"
    )

    client.close()
    cluster.close()
    log_status("Inventory creation complete.")

if __name__ == "__main__":
    main()
