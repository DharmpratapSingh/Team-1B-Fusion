"""
EDGAR Power Industry Data Preprocessing
======================================

This script processes EDGAR Power Industry emissions data (2000-2023) following the same
pattern as the transport data preprocessing. It combines yearly NetCDF files into a
single dataset and prepares it for city/country aggregation.

Based on: EDGAR_Transport.ipynb
Sector: Power Industry - Power and heat generation plants (public & autoproducers)
"""

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import time

# ------------ config ------------
RAW_DIR = Path("bkl_POWER_INDUSTRY_emi_nc")  # folder containing yearly *_POWER_INDUSTRY_emi.nc
CURATED_DIR = Path("data/curated")
CURATED_DIR.mkdir(parents=True, exist_ok=True)
OUT_NC = CURATED_DIR / "edgar_power_industry_2000_2024_rawcombined.nc"

# ------------ small timer helper ------------
def section_timer():
    t = {"last": time.time(), "start": None}

    def tick(label):
        now = time.time()
        if t["start"] is None:
            t["start"] = now
            t["last"] = now
            print(f"[{label}] start")
        else:
            print(f"[{label}] +{now - t['last']:.2f}s (elapsed {now - t['start']:.2f}s)")
            t["last"] = now

    return tick

tick = section_timer()

# ------------ discover files ------------
tick("discover")
files = sorted(RAW_DIR.glob("*POWER_INDUSTRY_emi.nc"))
print(f"Found {len(files)} files")
for f in files: 
    print("  -", f.name)
if not files:
    raise FileNotFoundError(f"No files matched in {RAW_DIR}/'*POWER_INDUSTRY_emi.nc'")

# ------------ open each file (fast: no time decode) ------------
tick("open_files")
datasets = []
lat0 = lon0 = None
emi_name = None

for i, fp in enumerate(files, 1):
    t0 = time.time()
    # Try h5netcdf first, fall back to netcdf4
    # Use decode_times=True to properly handle time dimension
    try:
        ds_y = xr.open_dataset(fp, engine="h5netcdf", decode_times=True)
    except ValueError:
        ds_y = xr.open_dataset(fp, engine="netcdf4", decode_times=True)

    # Minimal coord harmonization (NO sorting/reindexing)
    if "longitude" in ds_y.coords and "lon" not in ds_y.coords:
        ds_y = ds_y.rename({"longitude": "lon"})
    if "latitude" in ds_y.coords and "lat" not in ds_y.coords:
        ds_y = ds_y.rename({"latitude": "lat"})

    # Fail-fast: all years must share identical grid arrays (preserves overlap math later)
    if i == 1:
        lat0 = ds_y["lat"].values
        lon0 = ds_y["lon"].values
        emi_name = "emi_co2" if "emi_co2" in ds_y.data_vars else list(ds_y.data_vars)[0]
    else:
        if not (np.array_equal(lat0, ds_y["lat"].values) and np.array_equal(lon0, ds_y["lon"].values)):
            raise ValueError(f"Grid mismatch in {fp.name}. Normalize coords before combining.")
        cur_var = "emi_co2" if "emi_co2" in ds_y.data_vars else list(ds_y.data_vars)[0]
        if cur_var != emi_name:
            ds_y = ds_y.rename({cur_var: emi_name})

    datasets.append(ds_y)
    print(f"[open] {i:02d}/{len(files)} {fp.name} in {time.time() - t0:.2f}s")

# ------------ concatenate datasets ------------
tick("concat")
# Be explicit to avoid the compat/coords conflict:
# - data_vars="minimal": only concat variables that actually vary by 'time'
# - coords="minimal": keep only coords along the concat dim (time); lat/lon are identical, so this is fine
# - compat="no_conflicts": allow concat as long as there is no *overlap* conflict
# - combine_attrs="override": skip slow/strict attribute merging
ds = xr.concat(
    datasets,
    dim="time",
    data_vars="minimal",
    coords="minimal",
    compat="no_conflicts",
    combine_attrs="override",
)

# Debug: Check time dimension after concatenation
print(f"Time dimension after concat: {ds.time.shape}")
print(f"Time values (first 5): {ds.time.values[:5]}")
print(f"Time values (last 5): {ds.time.values[-5:]}")
print("Dims:", dict(ds.dims))
print("Data variables:", list(ds.data_vars))

# Quick QC (cheap since we didn't reorder)
v = emi_name
vmin = float(ds[v].min().values)
vmax = float(ds[v].max().values)
print(f"[QC] {v}: min={vmin:.3f}, max={vmax:.3f}")

# ------------ save uncompressed NetCDF (fast) ------------
tick("save_uncompressed_netcdf")
# Disable compression for ALL data vars & common coords to avoid slow write
all_write_vars = list(ds.data_vars) + [c for c in ["lat", "lon", "time", "year", "month"] if c in ds.variables]
encoding = {name: {"zlib": False} for name in all_write_vars}

# Use netcdf4 engine (more reliable)
ds.to_netcdf(OUT_NC, engine="netcdf4", encoding=encoding)
print(f"Saved combined file: {OUT_NC}")

# ------------ sidecars for city/county phase ------------
tick("sidecars")
# Stable cell_id map (one-time)
lat_vals = ds["lat"].values
lon_vals = ds["lon"].values
grid = (
    pd.DataFrame({
        "lat": np.repeat(lat_vals, len(lon_vals)),
        "lon": np.tile(lon_vals, len(lat_vals)),
    })
    .assign(cell_id=lambda d: np.arange(len(d), dtype=np.int64))
)
grid_path = CURATED_DIR / "grid_cells_power.parquet"
grid.to_parquet(grid_path, index=False)

# Provenance (raw time values as stored; decode later in normalization pass)
prov = pd.DataFrame({
    "index_in_concat": np.arange(ds.dims["time"], dtype=np.int32),
    "time_raw": ds["time"].values.astype("object"),
})
prov_path = CURATED_DIR / "provenance_power.parquet"
prov.to_parquet(prov_path, index=False)

print(f"Wrote sidecars:\n  - {grid_path}\n  - {prov_path}")
tick("done")

# ------------ brief log you can keep ------------
print("\n[LOG]")
print("• Combined EDGAR Power Industry yearly NetCDFs with decode_times=False (fast).")
print("• No normalization yet (lon sorting, month-end coercion deferred to Phase 1B/2).")
print("• Wrote uncompressed NetCDF (fast to write; larger on disk).")
print("• Exported sidecars for later city/county aggregation (grid cell_id map, per-time provenance).")
print(f"• Power Industry data covers {len(files)} years: {files[0].name.split('_')[3]} to {files[-1].name.split('_')[3]}")
