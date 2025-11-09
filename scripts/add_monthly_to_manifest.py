#!/usr/bin/env python3
"""
Add monthly tables to manifest_mcp_duckdb.json
"""
import json
from pathlib import Path

MANIFEST_PATH = Path("data/curated-2/manifest_mcp_duckdb.json")

# Load existing manifest
with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

# Sector definitions matching the annual tables
sectors = [
    ("transport", "Transport", "transport"),
    ("power", "Power Industry", "power industry"),
    ("agriculture", "Agriculture", "agriculture"),
    ("buildings", "Buildings", "buildings"),
    ("fuel-exploitation", "Fuel Exploitation", "fuel exploitation"),
    ("ind-combustion", "Industrial Combustion", "industrial combustion"),
    ("ind-processes", "Industrial Processes", "industrial processes"),
    ("waste", "Waste", "waste")
]

monthly_entries = []

for sector_id, sector_name, sector_source in sectors:
    # For each resolution (country, admin1, city)
    for resolution, resolution_name, table_suffix in [
        ("country", "Country", "country"),
        ("admin1", "Admin1", "admin1"),
        ("city", "City", "city")
    ]:
        # Build table name (e.g., transport_country_month)
        table_name = f"{sector_id.replace('-', '_')}_{table_suffix}_month"
        
        # Build file_id (e.g., transport-country-month)
        file_id = f"{sector_id}-{resolution}-month"
        
        # Find corresponding annual table to copy structure
        annual_file_id = f"{sector_id}-{resolution}-year"
        annual_entry = None
        for f in manifest["files"]:
            if f.get("file_id") == annual_file_id:
                annual_entry = f
                break
        
        if not annual_entry:
            print(f"⚠️  Warning: Annual entry not found for {annual_file_id}")
            continue
        
        # Create monthly entry based on annual entry
        monthly_entry = {
            "file_id": file_id,
            "name": f"{sector_name} Emissions by {resolution_name} (Monthly)",
            "description": f"Monthly {sector_name.lower()} CO₂ emissions by {resolution_name.lower()} from 2000-2023",
            "source": f"EDGAR v2024 {sector_source}",
            "units": "tonnes CO₂",
            "resolution": resolution,
            "temporal_coverage": "2000-2023 (monthly)",
            "engine": "duckdb",
            "path": f"duckdb://data/warehouse/climategpt.duckdb#{table_name}",
            "columns": []
        }
        
        # Copy columns from annual entry and add month column
        for col in annual_entry.get("columns", []):
            monthly_entry["columns"].append(col)
        
        # Add month column (insert after year column)
        year_index = next((i for i, col in enumerate(monthly_entry["columns"]) if col.get("name") == "year"), -1)
        if year_index >= 0:
            monthly_entry["columns"].insert(year_index + 1, {
                "name": "month",
                "type": "INTEGER",
                "description": "Month (1-12)"
            })
        else:
            # If no year column found, add month at the end
            monthly_entry["columns"].append({
                "name": "month",
                "type": "INTEGER",
                "description": "Month (1-12)"
            })
        
        monthly_entries.append(monthly_entry)
        print(f"✅ Added {file_id} -> {table_name}")

# Add monthly entries to manifest
manifest["files"].extend(monthly_entries)

# Sort files by file_id for consistency
manifest["files"].sort(key=lambda x: x.get("file_id", ""))

# Save updated manifest
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n🎉 Added {len(monthly_entries)} monthly table entries to manifest")
print(f"📊 Total files in manifest: {len(manifest['files'])}")
print(f"   - Annual: 24")
print(f"   - Monthly: {len(monthly_entries)}")


