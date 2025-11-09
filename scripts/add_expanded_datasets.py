#!/usr/bin/env python3
"""
Add expanded datasets to the MCP server manifest.
These datasets contain additional metadata and enhanced information.
"""

import json
from pathlib import Path

def add_expanded_datasets():
    """Add expanded transport datasets to manifest"""
    
    print("📝 Adding Expanded Datasets to Manifest...")
    print("=" * 45)
    
    # Load existing manifest
    manifest_path = Path("data/curated/manifest_mcp_multi.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Define expanded datasets
    expanded_datasets = [
        {
            "id": "transport-country-year-expanded",
            "title": "Country transport CO₂ totals (yearly) - Expanded",
            "description": "Enhanced country totals with additional metadata and quality indicators.",
            "level": "country",
            "period": "year",
            "sector": "transport",
            "path": "data/curated/country-year/transport_country_year_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"iso3","type":"string","role":"key"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","iso3"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "year"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "annual", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.862} },
            "query_capabilities": {
                "filterable": ["year","iso3","country_name"],
                "groupable": ["year","iso3"],
                "sortable": ["emissions_tonnes","MtCO2","year","country_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        },
        {
            "id": "transport-country-month-expanded",
            "title": "Country transport CO₂ totals (monthly) - Expanded",
            "description": "Enhanced country monthly totals with additional metadata and quality indicators.",
            "level": "country",
            "period": "month",
            "sector": "transport",
            "path": "data/curated/country-month/transport_country_month_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"iso3","type":"string","role":"key"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"month","type":"int8","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","month","iso3"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "month"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "monthly", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.862} },
            "query_capabilities": {
                "filterable": ["year","month","iso3","country_name"],
                "groupable": ["year","month","iso3"],
                "sortable": ["emissions_tonnes","MtCO2","year","month","country_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        },
        {
            "id": "transport-admin1-year-expanded",
            "title": "State/Province transport CO₂ totals (yearly) - Expanded",
            "description": "Enhanced admin-1 yearly totals with additional metadata and quality indicators.",
            "level": "admin1",
            "period": "year",
            "sector": "transport",
            "path": "data/curated/admin1-year/transport_admin1_year_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"admin1_geoid","type":"string","role":"key"},
                {"name":"admin1_name","type":"string","role":"label"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"iso3","type":"string","role":"code"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","admin1_geoid"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "year"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "annual", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.975, "min": 0.973, "max": 0.977} },
            "provenance": { "join_strategy": "Admin-1 intersects, nearest fallback up to 250km" },
            "query_capabilities": {
                "filterable": ["year","iso3","country_name","admin1_name","admin1_geoid"],
                "groupable": ["year","iso3","country_name"],
                "sortable": ["emissions_tonnes","MtCO2","year","admin1_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        },
        {
            "id": "transport-admin1-month-expanded",
            "title": "State/Province transport CO₂ totals (monthly) - Expanded",
            "description": "Enhanced admin-1 monthly totals with additional metadata and quality indicators.",
            "level": "admin1",
            "period": "month",
            "sector": "transport",
            "path": "data/curated/admin1-month/transport_admin1_month_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"admin1_geoid","type":"string","role":"key"},
                {"name":"admin1_name","type":"string","role":"label"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"iso3","type":"string","role":"code"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"month","type":"int8","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","month","admin1_geoid"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "month"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "monthly", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.975, "min": 0.973, "max": 0.977} },
            "provenance": { "join_strategy": "Admin-1 intersects, nearest fallback up to 250km" },
            "query_capabilities": {
                "filterable": ["year","month","iso3","country_name","admin1_name","admin1_geoid"],
                "groupable": ["year","month","iso3","country_name"],
                "sortable": ["emissions_tonnes","MtCO2","year","month","admin1_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        },
        {
            "id": "transport-city-year-expanded",
            "title": "City transport CO₂ totals (yearly) - Expanded",
            "description": "Enhanced city yearly totals with additional metadata and quality indicators.",
            "level": "city",
            "period": "year",
            "sector": "transport",
            "path": "data/curated/city-year/transport_city_year_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"city_id","type":"string","role":"key"},
                {"name":"city_name","type":"string","role":"label"},
                {"name":"admin1_name","type":"string","role":"label"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"iso3","type":"string","role":"code"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","city_id"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "year"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "annual", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.045, "min": 0.040, "max": 0.048}, "cities_count": 1305 },
            "provenance": { "generated_from": "EDGAR v2024 transport gridded 0.1°", "city_mapping": "UCDB polygons (strict) + nearest-centroid fallback" },
            "query_capabilities": {
                "filterable": ["year","iso3","country_name","admin1_name","city_name","city_id"],
                "groupable": ["year","iso3","country_name","admin1_name"],
                "sortable": ["emissions_tonnes","MtCO2","year","city_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        },
        {
            "id": "transport-city-month-expanded",
            "title": "City transport CO₂ totals (monthly) - Expanded",
            "description": "Enhanced city monthly totals with additional metadata and quality indicators.",
            "level": "city",
            "period": "month",
            "sector": "transport",
            "path": "data/curated/city-month/transport_city_month_expanded.parquet",
            "file_format": "parquet",
            "exists": True,
            "priority": 3,
            "columns": [
                {"name":"city_id","type":"string","role":"key"},
                {"name":"city_name","type":"string","role":"label"},
                {"name":"admin1_name","type":"string","role":"label"},
                {"name":"country_name","type":"string","role":"label"},
                {"name":"iso3","type":"string","role":"code"},
                {"name":"year","type":"int16","role":"time"},
                {"name":"month","type":"int8","role":"time"},
                {"name":"emissions_tonnes","type":"float64","role":"measure","units":"tonnes CO2"},
                {"name":"MtCO2","type":"float64","role":"measure","units":"Mt CO2"}
            ],
            "primary_keys": ["year","month","city_id"],
            "temporal_extent": {"start": 2000, "end": 2023, "grain": "month"},
            "spatial_extent": {"type":"global"},
            "semantics": { "units": ["tonnes CO2","Mt CO2"], "spatial_res": "0.1°", "temporal_res": "monthly", "source": "EDGAR v2024 transport" },
            "quality": { "coverage_ratio_vs_global": {"mean": 0.045, "min": 0.040, "max": 0.048}, "cities_count": 1305 },
            "provenance": { "generated_from": "EDGAR v2024 transport gridded 0.1°", "city_mapping": "UCDB polygons (strict) + nearest-centroid fallback" },
            "query_capabilities": {
                "filterable": ["year","month","iso3","country_name","admin1_name","city_name","city_id"],
                "groupable": ["year","month","iso3","country_name","admin1_name"],
                "sortable": ["emissions_tonnes","MtCO2","year","month","city_name"]
            },
            "enhanced_features": ["additional_metadata", "quality_indicators", "optimized_storage"]
        }
    ]
    
    # Add expanded datasets to manifest
    manifest["files"].extend(expanded_datasets)
    
    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Added {len(expanded_datasets)} expanded datasets to manifest")
    print(f"📁 Updated manifest: {manifest_path}")
    print(f"🚀 Expanded datasets now available via MCP server")
    print(f"💾 Enhanced features: additional metadata, quality indicators, optimized storage")

if __name__ == "__main__":
    try:
        add_expanded_datasets()
        print(f"\n🎉 SUCCESS!")
        print(f"📊 All expanded datasets added to manifest")
        print(f"🌐 Enhanced datasets now accessible via MCP server")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()



