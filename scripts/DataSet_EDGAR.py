import xarray as xr
ds = xr.open_dataset("bkl_TRANSPORT_emi_nc/EDGAR_2024_GHG_CO2_2023_bkl_TRANSPORT_emi.nc")
print(ds)
print(ds.data_vars)
print(ds.coords)
print('----------------------------------')
print(ds.dims)
print(ds["time"].values[:12])  # first 12 timestamps
print('----------------------------------')
print(len(ds["lat"]), len(ds["lon"]))
print('----------------------------------')
import numpy as np
print(np.isnan(ds["emissions"]).sum().item())
print('----------------------------------')
print(ds["emissions"].min().item(), ds["emissions"].max().item())
print('----------------------------------')
print(ds["emissions"].attrs)
print('----------------------------------')
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

DATA_DIR = Path("bkl_TRANSPORT_emi_nc")
files = sorted(DATA_DIR.glob("EDGAR_*_bkl_TRANSPORT_emi.nc"))
assert files, "No NetCDF files found."

USE_DASK = False  # set to True after installing dask

if USE_DASK:
    # requires: python3 -m pip install "dask[array]>=2024.1.0" netcdf4
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})
else:
    # Manual concat WITHOUT dask: open each file and concatenate a long time
    datasets = []
    for p in files:
        # Prefer h5netcdf for NetCDF4 files; fall back to default auto-detect if needed
        try:
            dsi = xr.open_dataset(p, engine="h5netcdf")
        except Exception:
            dsi = xr.open_dataset(p)
        datasets.append(dsi)
        # Concatenate by time without triggering dask
        ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        )

var = "emissions"
da = ds[var]

print("Opened", len(files), "files without dask; concatenated along 'time'.")


# Units say Tonnes → totals per grid cell per month; no area-weight
units = (da.attrs.get("units") or "").lower()
needs_area = False
print(f"Units: {units!r} | Area-weight needed? {needs_area}")

da_tot = da.astype("float64")
out_units = "tonnes"

# Global monthly & annual totals
global_monthly = da_tot.sum(dim=["lat","lon"])  # (time)
global_monthly_df = global_monthly.to_series().rename("value").to_frame()
global_monthly_df.index.name = "time"
global_monthly_df["units"] = out_units

global_annual_df = (
    global_monthly_df
    .assign(year=lambda d: d.index.year)
    .groupby("year")["value"].sum()
    .to_frame()
)
global_annual_df["units"] = out_units

print("Global monthly (head):")
print(global_monthly_df.head())
print("\nGlobal annual:")
print(global_annual_df.head(3))

# Save CSVs (and also MtCO2 for reporting convenience)
global_monthly_df.to_csv("transport_global_monthly_total_tonnes.csv")
global_annual_df.to_csv("transport_global_annual_total_tonnes.csv")

global_monthly_mt = global_monthly_df.assign(value=lambda d: d["value"]/1e6)
global_annual_mt  = global_annual_df.assign(value=lambda d: d["value"]/1e6)
global_monthly_mt["units"] = "MtCO2"
global_annual_mt["units"]  = "MtCO2"
global_monthly_mt.to_csv("transport_global_monthly_total_MtCO2.csv")
global_annual_mt.to_csv("transport_global_annual_total_MtCO2.csv")

print("Wrote CSVs: *_tonnes.csv and *_MtCO2.csv")

# === Slide-friendly dataset stats ===
print("\n=== DATASET STATS ===")
# variables (features)
vars_list = list(ds.data_vars.keys())
coords_list = list(ds.coords.keys())

print(f"Data variables ({len(vars_list)}): {vars_list}")
print(f"Coordinates ({len(coords_list)}): {coords_list}")

# dimensions & shape
dims = {k: int(v) for k, v in ds.dims.items()}
print(f"Dimensions: {dims}")

# “rows/columns” notion for NetCDF:
# treat (time, lat, lon) grid as a table with columns=[emissions plus IDs] and rows=time*lat*lon
time_dim = dims.get("time", 1)
lat_dim  = dims.get("lat", 1)
lon_dim  = dims.get("lon", 1)
approx_rows = time_dim * lat_dim * lon_dim
approx_cols = 1  # emissions (add more if there are multiple pollutants/vars)

print(f"Approx tabular rows (time*lat*lon): {approx_rows:,}")
print(f"Approx value columns (data vars): {len(vars_list)}")

# temporal coverage
if "time" in ds:
    tmin = pd.to_datetime(ds["time"].values.min())
    tmax = pd.to_datetime(ds["time"].values.max())
    print(f"Temporal coverage: {tmin.date()} → {tmax.date()} ({time_dim} steps)")

# spatial coverage
lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
if lat_name:
    lat_vals = ds[lat_name].values
    print(f"Latitude range: {float(lat_vals.min())} to {float(lat_vals.max())} ({lat_dim} points)")
if lon_name:
    lon_vals = ds[lon_name].values
    print(f"Longitude range: {float(lon_vals.min())} to {float(lon_vals.max())} ({lon_dim} points)")

# missingness and range for "emissions"
nan_count = int(np.isnan(da).sum().item())
total_vals = int(np.product([time_dim, lat_dim, lon_dim]))
nan_pct = 100.0 * nan_count / max(total_vals, 1)
vmin = float(da.min().item())
vmax = float(da.max().item())
units = str(da.attrs.get("units", "unknown"))

print(f"Missing values in 'emissions': {nan_count:,} ({nan_pct:.3f}%)")
print(f"'emissions' range: min={vmin}, max={vmax} (units={units})")