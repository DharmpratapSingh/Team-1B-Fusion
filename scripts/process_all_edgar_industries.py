"""
EDGAR All Industries Data Processing & Aggregation
=================================================

This script processes ALL 8 EDGAR industry emissions datasets (2000-2024) in one go:
1. Agriculture
2. Buildings  
3. Fuel Exploitation
4. Industrial Combustion
5. Industrial Processes
6. Power Industry (already processed)
7. Transport (already processed)
8. Waste

It combines yearly NetCDF files into single datasets and aggregates them to 
country, admin1 (state/province), and city levels using spatial joins.

Based on: EDGAR_Transport.ipynb and individual industry processing scripts
"""

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import time
import zipfile
import shutil
import geopandas as gpd
from shapely.geometry import Point

# ------------ config ------------
INDUSTRIES = [
    "AGRICULTURE",
    "BUILDINGS", 
    "FUEL_EXPLOITATION",
    "IND_COMBUSTION",
    "IND_PROCESSES",
    "WASTE"
]

# Skip Power Industry and Transport (already processed)
ZIP_FILES = {
    "AGRICULTURE": "ClimateGPT_All/bkl_AGRICULTURE_emi_nc.zip",
    "BUILDINGS": "ClimateGPT_All/bkl_BUILDINGS_emi_nc.zip",
    "FUEL_EXPLOITATION": "ClimateGPT_All/bkl_FUEL_EXPLOITATION_emi_nc.zip",
    "IND_COMBUSTION": "ClimateGPT_All/bkl_IND_COMBUSTION_emi_nc.zip",
    "IND_PROCESSES": "ClimateGPT_All/bkl_IND_PROCESSES_emi_nc.zip",
    "WASTE": "ClimateGPT_All/bkl_WASTE_emi_nc.zip"
}

RAW_DIR = Path(".")
CURATED_DIR = Path("data/curated")
GEO_DIR = Path("data/geo")
CURATED_DIR.mkdir(parents=True, exist_ok=True)

# ------------ timer helper ------------
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

# ------------ process each industry ------------
def process_industry(industry):
    """Process a single industry: extract, combine NetCDF files, and aggregate"""
    print(f"\n{'='*60}")
    print(f"🏭 PROCESSING {industry} INDUSTRY")
    print(f"{'='*60}")
    
    tick(f"process_{industry.lower()}")
    
    # Extract zip file
    zip_path = Path(ZIP_FILES[industry])
    if not zip_path.exists():
        print(f"⚠️  Zip file not found: {zip_path}")
        return False
    
    # Check if files already exist
    existing_files = list(RAW_DIR.glob(f"EDGAR_*_bkl_{industry}_emi.nc"))
    if not existing_files:
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"✅ Extracted {industry} files to root directory")
    else:
        print(f"✅ {industry} files already exist ({len(existing_files)} files)")
    
    # Discover files
    files = sorted(RAW_DIR.glob(f"EDGAR_*_bkl_{industry}_emi.nc"))
    print(f"📁 Found {len(files)} {industry} files")
    if not files:
        print(f"❌ No files matched for {industry}")
        return False
    
    # Combine NetCDF files
    print(f"🔄 Combining {len(files)} NetCDF files...")
    datasets = []
    lat0 = lon0 = None
    emi_name = None
    
    for i, fp in enumerate(files, 1):
        t0 = time.time()
        # Try different engines, fall back to default
        try:
            ds_y = xr.open_dataset(fp, engine="h5netcdf", decode_times=True)
        except ValueError:
            try:
                ds_y = xr.open_dataset(fp, engine="netcdf4", decode_times=True)
            except ValueError:
                ds_y = xr.open_dataset(fp, decode_times=True)
        
        # Coordinate harmonization
        if "longitude" in ds_y.coords and "lon" not in ds_y.coords:
            ds_y = ds_y.rename({"longitude": "lon"})
        if "latitude" in ds_y.coords and "lat" not in ds_y.coords:
            ds_y = ds_y.rename({"latitude": "lat"})
        
        # Grid validation
        if i == 1:
            lat0 = ds_y["lat"].values
            lon0 = ds_y["lon"].values
            emi_name = "emi_co2" if "emi_co2" in ds_y.data_vars else list(ds_y.data_vars)[0]
        else:
            if not (np.array_equal(lat0, ds_y["lat"].values) and np.array_equal(lon0, ds_y["lon"].values)):
                raise ValueError(f"Grid mismatch in {fp.name}")
            cur_var = "emi_co2" if "emi_co2" in ds_y.data_vars else list(ds_y.data_vars)[0]
            if cur_var != emi_name:
                ds_y = ds_y.rename({cur_var: emi_name})
        
        datasets.append(ds_y)
        print(f"  📄 {i:02d}/{len(files)} {fp.name} ({time.time() - t0:.2f}s)")
    
    # Concatenate datasets
    print(f"🔗 Concatenating {len(datasets)} datasets...")
    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="no_conflicts",
        combine_attrs="override",
    )
    
    # Save combined NetCDF
    out_nc = CURATED_DIR / f"edgar_{industry.lower()}_2000_2024_rawcombined.nc"
    print(f"💾 Saving combined dataset: {out_nc}")
    
    all_write_vars = list(ds.data_vars) + [c for c in ["lat", "lon", "time", "year", "month"] if c in ds.variables]
    encoding = {name: {"zlib": False} for name in all_write_vars}
    ds.to_netcdf(out_nc, encoding=encoding)
    
    # Create sidecar files
    print(f"📋 Creating sidecar files...")
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    grid = (
        pd.DataFrame({
            "lat": np.repeat(lat_vals, len(lon_vals)),
            "lon": np.tile(lon_vals, len(lat_vals)),
        })
        .assign(cell_id=lambda d: np.arange(len(d), dtype=np.int64))
    )
    grid_path = CURATED_DIR / f"grid_cells_{industry.lower()}.csv"
    grid.to_csv(grid_path, index=False)
    
    prov = pd.DataFrame({
        "index_in_concat": np.arange(ds.sizes["time"], dtype=np.int32),
        "time_raw": ds["time"].values.astype("object"),
    })
    prov_path = CURATED_DIR / f"provenance_{industry.lower()}.csv"
    prov.to_csv(prov_path, index=False)
    
    # Clean up individual files
    print(f"🧹 Cleaning up {len(files)} individual NetCDF files...")
    for file_path in files:
        if file_path.exists():
            file_path.unlink()
    
    # Aggregate to geographic levels
    print(f"🌍 Aggregating {industry} to country/state/city levels...")
    aggregate_industry(industry, ds)
    
    print("✅ " + industry + " processing complete!")
    return True

def aggregate_industry(industry, ds):
    """Aggregate industry data to country, admin1, and city levels"""
    
    # Create output directories
    country_dir = CURATED_DIR / f"{industry.lower()}-country-year"
    admin1_dir = CURATED_DIR / f"{industry.lower()}-admin1-year"
    city_dir = CURATED_DIR / f"{industry.lower()}-city-year"
    
    for dir_path in [country_dir, admin1_dir, city_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load geographic data
    countries_gdf = gpd.read_file(GEO_DIR / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp")
    admin1_gdf = gpd.read_file(GEO_DIR / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp")
    cities_gdf = gpd.read_file(GEO_DIR / "GHS_UCDB_GLOBE_R2024A_V1_1" / "GHS_UCDB_GLOBE_R2024A.gpkg",
                              layer="GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A")
    cities_gdf = cities_gdf.to_crs('EPSG:4326')
    cities_gdf.columns = [col.strip('\ufeff') for col in cities_gdf.columns]
    
    # Create grid points
    lats = ds.lat.values
    lons = ds.lon.values
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append(Point(lon, lat))
    
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_points}, crs='EPSG:4326')
    
    # Country aggregation
    print(f"  🏴 Country aggregation...")
    grid_countries = gpd.sjoin(grid_gdf, countries_gdf, how='left', predicate='within')
    country_mapping = {}
    for idx, row in grid_countries.iterrows():
        if pd.notna(row['ADM0_A3']):
            lat_idx = idx // len(lons)
            lon_idx = idx % len(lons)
            country_mapping[(lat_idx, lon_idx)] = {
                'iso3': row['ADM0_A3'],
                'country_name': row['NAME']
            }
    
    # Admin1 aggregation
    print(f"  🏛️  Admin1 aggregation...")
    grid_admin1 = gpd.sjoin(grid_gdf, admin1_gdf, how='left', predicate='within')
    country_name_map = {row['ADM0_A3']: row['NAME'] for _, row in countries_gdf.iterrows()}
    admin1_mapping = {}
    for idx, row in grid_admin1.iterrows():
        if pd.notna(row['adm1_code']):
            lat_idx = idx // len(lons)
            lon_idx = idx % len(lons)
            country_name = country_name_map.get(row['adm0_a3'], 'Unknown')
            admin1_mapping[(lat_idx, lon_idx)] = {
                'admin1_geoid': row['adm1_code'],
                'admin1_name': row['name'],
                'country_name': country_name,
                'iso3': row['adm0_a3']
            }
    
    # City aggregation
    print(f"  🏙️  City aggregation...")
    grid_cities = gpd.sjoin(grid_gdf, cities_gdf, how='left', predicate='within')
    city_mapping = {}
    for idx, row in grid_cities.iterrows():
        if pd.notna(row['ID_UC_G0']):
            lat_idx = idx // len(lons)
            lon_idx = idx % len(lons)
            city_name = str(row['GC_UCN_MAI_2025']).strip().replace('\ufeff', '') if pd.notna(row['GC_UCN_MAI_2025']) else 'Unknown'
            country_name_raw = str(row['GC_CNT_GAD_2025']).strip().replace('\ufeff', '') if pd.notna(row['GC_CNT_GAD_2025']) else 'Unknown'
            iso3_raw = str(row['GC_CNT_UNN_2025']).strip().replace('\ufeff', '') if pd.notna(row['GC_CNT_UNN_2025']) else 'Unknown'
            country_name = country_name_map.get(iso3_raw, country_name_raw) if pd.notna(iso3_raw) and iso3_raw != 'Unknown' else country_name_raw
            
            admin1_info = admin1_mapping.get((lat_idx, lon_idx), {})
            admin1_name = admin1_info.get('admin1_name', 'Unknown')
            
            city_mapping[(lat_idx, lon_idx)] = {
                'city_id': row['ID_UC_G0'],
                'city_name': city_name,
                'admin1_name': admin1_name,
                'country_name': country_name,
                'iso3': iso3_raw
            }
    
    # Process emissions by year
    print(f"  📊 Processing emissions by year...")
    
    # Rename emissions variable if needed
    if 'emissions' not in ds.data_vars:
        emission_var_candidates = ['emi_co2', 'emissions']
        found_emission_var = None
        for var_name in emission_var_candidates:
            if var_name in ds.data_vars:
                found_emission_var = var_name
                break
        if found_emission_var:
            ds = ds.rename({found_emission_var: 'emissions'})
    
    # Aggregate emissions
    country_emissions = {}
    admin1_emissions = {}
    city_emissions = {}
    
    for year in range(2000, 2025):  # 2000-2024
        year_data = ds.sel(time=ds.time.dt.year == year)
        if len(year_data.time) == 0:
            continue
        
        annual_emissions = year_data.emissions.sum(dim='time')
        
        # Country emissions
        for (lat_idx, lon_idx), country_info in country_mapping.items():
            iso3 = country_info['iso3']
            country_name = country_info['country_name']
            
            if iso3 not in country_emissions:
                country_emissions[iso3] = {
                    'country_name': country_name,
                    'emissions_by_year': {}
                }
            
            emission_value = float(annual_emissions.isel(lat=lat_idx, lon=lon_idx).values)
            if pd.notna(emission_value) and emission_value > 0:
                if year not in country_emissions[iso3]['emissions_by_year']:
                    country_emissions[iso3]['emissions_by_year'][year] = 0
                country_emissions[iso3]['emissions_by_year'][year] += emission_value
        
        # Admin1 emissions
        for (lat_idx, lon_idx), admin1_info in admin1_mapping.items():
            geoid = admin1_info['admin1_geoid']
            
            if geoid not in admin1_emissions:
                admin1_emissions[geoid] = {
                    'admin1_name': admin1_info['admin1_name'],
                    'country_name': admin1_info['country_name'],
                    'iso3': admin1_info['iso3'],
                    'emissions_by_year': {}
                }
            
            emission_value = float(annual_emissions.isel(lat=lat_idx, lon=lon_idx).values)
            if pd.notna(emission_value) and emission_value > 0:
                if year not in admin1_emissions[geoid]['emissions_by_year']:
                    admin1_emissions[geoid]['emissions_by_year'][year] = 0
                admin1_emissions[geoid]['emissions_by_year'][year] += emission_value
        
        # City emissions
        for (lat_idx, lon_idx), city_info in city_mapping.items():
            city_id = city_info['city_id']
            
            if city_id not in city_emissions:
                city_emissions[city_id] = {
                    'city_name': city_info['city_name'],
                    'admin1_name': city_info['admin1_name'],
                    'country_name': city_info['country_name'],
                    'iso3': city_info['iso3'],
                    'emissions_by_year': {}
                }
            
            emission_value = float(annual_emissions.isel(lat=lat_idx, lon=lon_idx).values)
            if pd.notna(emission_value) and emission_value > 0:
                if year not in city_emissions[city_id]['emissions_by_year']:
                    city_emissions[city_id]['emissions_by_year'][year] = 0
                city_emissions[city_id]['emissions_by_year'][year] += emission_value
    
    # Create DataFrames and save
    print(f"  💾 Saving aggregated data...")
    
    # Country data
    country_data = []
    for iso3, data in country_emissions.items():
        for year, emissions in data['emissions_by_year'].items():
            country_data.append({
                'iso3': iso3,
                'country_name': data['country_name'],
                'year': year,
                'emissions_tonnes': emissions,
                'MtCO2': emissions / 1e6,
                'units': 'tonnes CO2',
                'source': f'EDGAR v2024 {industry.lower()}',
                'spatial_res': '0.1°',
                'temporal_res': 'annual'
            })
    
    country_df = pd.DataFrame(country_data)
    country_df = country_df.sort_values(['year', 'emissions_tonnes'], ascending=[True, False])
    country_csv = country_dir / f"{industry.lower()}_country_year.csv"
    country_df.to_csv(country_csv, index=False)
    
    # Admin1 data
    admin1_data = []
    for geoid, data in admin1_emissions.items():
        for year, emissions in data['emissions_by_year'].items():
            admin1_data.append({
                'admin1_geoid': geoid,
                'admin1_name': data['admin1_name'],
                'country_name': data['country_name'],
                'iso3': data['iso3'],
                'year': year,
                'emissions_tonnes': emissions,
                'MtCO2': emissions / 1e6,
                'units': 'tonnes CO2',
                'source': f'EDGAR v2024 {industry.lower()}',
                'spatial_res': '0.1°',
                'temporal_res': 'annual'
            })
    
    admin1_df = pd.DataFrame(admin1_data)
    admin1_df = admin1_df.sort_values(['year', 'emissions_tonnes'], ascending=[True, False])
    admin1_csv = admin1_dir / f"{industry.lower()}_admin1_year.csv"
    admin1_df.to_csv(admin1_csv, index=False)
    
    # City data
    city_data = []
    for city_id, data in city_emissions.items():
        for year, emissions in data['emissions_by_year'].items():
            city_data.append({
                'city_id': city_id,
                'city_name': data['city_name'],
                'admin1_name': data['admin1_name'],
                'country_name': data['country_name'],
                'iso3': data['iso3'],
                'year': year,
                'emissions_tonnes': emissions,
                'MtCO2': emissions / 1e6,
                'units': 'tonnes CO2',
                'source': f'EDGAR v2024 {industry.lower()}',
                'spatial_res': '0.1°',
                'temporal_res': 'annual'
            })
    
    city_df = pd.DataFrame(city_data)
    city_df = city_df.sort_values(['year', 'emissions_tonnes'], ascending=[True, False])
    city_csv = city_dir / f"{industry.lower()}_city_year.csv"
    city_df.to_csv(city_csv, index=False)
    
    print(f"    ✅ Country: {len(country_df)} rows")
    print(f"    ✅ Admin1: {len(admin1_df)} rows")
    print(f"    ✅ City: {len(city_df)} rows")

# ------------ main execution ------------
def main():
    print("🚀 EDGAR ALL INDUSTRIES PROCESSING")
    print("=" * 60)
    print("Processing 6 industries (Power & Transport already done):")
    for industry in INDUSTRIES:
        print(f"  • {industry}")
    print("=" * 60)
    
    success_count = 0
    total_count = len(INDUSTRIES)
    
    for industry in INDUSTRIES:
        try:
            if process_industry(industry):
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing {industry}: {e}")
    
    tick("final_summary")
    
    print(f"\n{'='*60}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {success_count}/{total_count} industries")
    print(f"📁 All data saved to: {CURATED_DIR}")
    print(f"🌍 Geographic levels: Country, Admin1 (State/Province), City")
    print(f"📊 Time range: 2000-2024 (25 years)")
    print(f"🎯 Next step: Update MCP server manifest with all new datasets")
    
    if success_count == total_count:
        print(f"\n🏆 ALL INDUSTRIES PROCESSED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {total_count - success_count} industries failed - check logs above")

if __name__ == "__main__":
    main()
