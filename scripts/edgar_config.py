"""
EDGAR Processing Configuration
=============================

Configuration file for EDGAR data processing pipeline.
Customize these settings for your specific system and requirements.
"""

from pathlib import Path

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Base directory - UPDATE THIS PATH FOR YOUR ALIENWARE SYSTEM
BASE_DIR = Path("/Users/dharmpratapsingh/Downloads/DataSets_ClimateGPT")

# Memory configuration (adjust based on your system)
MEMORY_LIMIT_GB = 100  # Leave some RAM for system (128GB total)
MAX_WORKERS = 6  # Conservative for memory usage, increase if you have more RAM

# Processing configuration
BATCH_SIZE = 500000  # Grid cells per batch
CHUNK_SIZE = 1000000  # For large array operations

# =============================================================================
# INDUSTRY CONFIGURATION
# =============================================================================

# Industries to process (comment out any you don't want to process)
INDUSTRIES_TO_PROCESS = [
    'AGRICULTURE',      # Agricultural activities and livestock
    'BUILDINGS',        # Residential and commercial buildings
    'FUEL_EXPLOITATION', # Oil and gas extraction and processing
    'IND_COMBUSTION',   # Industrial combustion processes
    'IND_PROCESSES',    # Industrial processes (non-combustion)
    'POWER_INDUSTRY',   # Power and heat generation plants
    'TRANSPORT',        # Transportation emissions
    'WASTE',           # Waste management and disposal
]

# Industry descriptions
INDUSTRY_DESCRIPTIONS = {
    'AGRICULTURE': 'Agricultural activities and livestock emissions',
    'BUILDINGS': 'Residential and commercial buildings emissions',
    'FUEL_EXPLOITATION': 'Oil and gas extraction and processing emissions',
    'IND_COMBUSTION': 'Industrial combustion processes emissions',
    'IND_PROCESSES': 'Industrial processes (non-combustion) emissions',
    'POWER_INDUSTRY': 'Power and heat generation plants emissions',
    'TRANSPORT': 'Transportation emissions',
    'WASTE': 'Waste management and disposal emissions',
}

# =============================================================================
# FILE PATHS
# =============================================================================

# Input directories
CLIMATE_ALL_DIR = BASE_DIR / "ClimateGPT_All"
CURATED_DIR = BASE_DIR / "data" / "curated"
GEO_DIR = BASE_DIR / "data" / "geo"

# Geographic data paths
CITY_GPKG = GEO_DIR / "GHS_UCDB_GLOBE_R2024A_V1_1" / "GHS_UCDB_GLOBE_R2024A.gpkg"
ADMIN1_PATH = GEO_DIR / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp"
ADMIN0_PATH = GEO_DIR / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"

# Output directories
OUTPUT_DIRS = {
    'country': CURATED_DIR / "country-year",
    'admin1': CURATED_DIR / "admin1-year",
    'city': CURATED_DIR / "city-year",
    'monthly': CURATED_DIR / "monthly",  # For future monthly aggregation
}

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================

# NetCDF processing options
NETCDF_OPTIONS = {
    'engine': 'netcdf4',  # Primary engine
    'fallback_engine': 'h5netcdf',  # Fallback engine
    'decode_times': True,
    'compression': False,  # Disable compression for faster writing
}

# Spatial processing options
SPATIAL_OPTIONS = {
    'crs': 'EPSG:4326',  # WGS84
    'predicate': 'within',  # Spatial join predicate
    'max_distance': 0.25,  # Maximum distance for nearest neighbor joins (degrees)
}

# Aggregation options
AGGREGATION_OPTIONS = {
    'years': list(range(2000, 2024)),  # Years to process
    'temporal_resolution': 'annual',  # 'annual' or 'monthly'
    'spatial_resolution': '0.1°',  # Grid resolution
    'units': 'tonnes CO2',
    'source_prefix': 'EDGAR v2024',
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_files': {
        'processing': 'edgar_processing.log',
        'geocoding': 'edgar_geocoding.log',
        'pipeline': 'edgar_full_pipeline.log',
    }
}

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Performance tuning parameters
PERFORMANCE_CONFIG = {
    'use_multiprocessing': True,
    'use_threading': False,  # Set to True for I/O bound operations
    'memory_efficient': True,
    'progress_bars': True,
    'verbose_logging': True,
}

# Memory management
MEMORY_CONFIG = {
    'chunk_size': CHUNK_SIZE,
    'batch_size': BATCH_SIZE,
    'gc_frequency': 100,  # Run garbage collection every N operations
    'max_memory_usage': MEMORY_LIMIT_GB * 1024**3,  # Convert to bytes
}

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Data validation options
VALIDATION_CONFIG = {
    'check_grid_consistency': True,
    'validate_emissions_range': True,
    'min_emission_threshold': 1e-6,  # Minimum emission value to include
    'max_emission_threshold': 1e12,  # Maximum emission value (sanity check)
    'check_geographic_coverage': True,
}

# =============================================================================
# ALIENWARE SYSTEM OPTIMIZATIONS
# =============================================================================

# Optimizations for high-memory systems
ALIENWARE_OPTIMIZATIONS = {
    'parallel_extraction': True,  # Extract multiple zip files in parallel
    'parallel_processing': True,  # Process multiple industries in parallel
    'memory_mapping': True,  # Use memory mapping for large files
    'fast_io': True,  # Use fast I/O operations
    'gpu_acceleration': False,  # Set to True if you have compatible GPU libraries
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_industry_config(industry: str) -> dict:
    """Get configuration for a specific industry."""
    return {
        'name': industry,
        'description': INDUSTRY_DESCRIPTIONS.get(industry, 'Unknown industry'),
        'zip_file': f'bkl_{industry}_emi_nc.zip',
        'folder_name': f'bkl_{industry}_emi_nc',
        'output_prefix': industry.lower(),
    }

def get_output_paths(industry: str) -> dict:
    """Get output file paths for a specific industry."""
    prefix = industry.lower()
    return {
        'netcdf': CURATED_DIR / f"edgar_{prefix}_2000_2024_rawcombined.nc",
        'metadata': CURATED_DIR / f"metadata_{prefix}.json",
        'grid': CURATED_DIR / f"grid_cells_{prefix}.parquet",
        'provenance': CURATED_DIR / f"provenance_{prefix}.parquet",
        'country_csv': OUTPUT_DIRS['country'] / f"{prefix}_country_year.csv",
        'admin1_csv': OUTPUT_DIRS['admin1'] / f"{prefix}_admin1_year.csv",
        'city_csv': OUTPUT_DIRS['city'] / f"{prefix}_city_year.csv",
    }

def validate_config() -> bool:
    """Validate the configuration."""
    errors = []
    
    # Check if base directory exists
    if not BASE_DIR.exists():
        errors.append(f"Base directory does not exist: {BASE_DIR}")
    
    # Check if ClimateGPT_All directory exists
    if not CLIMATE_ALL_DIR.exists():
        errors.append(f"ClimateGPT_All directory does not exist: {CLIMATE_ALL_DIR}")
    
    # Check if geographic data exists
    if not CITY_GPKG.exists():
        errors.append(f"City data not found: {CITY_GPKG}")
    
    if not ADMIN1_PATH.exists():
        errors.append(f"Admin1 data not found: {ADMIN1_PATH}")
    
    if not ADMIN0_PATH.exists():
        errors.append(f"Admin0 data not found: {ADMIN0_PATH}")
    
    # Check memory configuration
    if MEMORY_LIMIT_GB > 120:
        errors.append(f"Memory limit too high: {MEMORY_LIMIT_GB}GB (max recommended: 120GB)")
    
    if MAX_WORKERS > 16:
        errors.append(f"Too many workers: {MAX_WORKERS} (max recommended: 16)")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    print("EDGAR Processing Configuration")
    print("=" * 40)
    print(f"Base directory: {BASE_DIR}")
    print(f"Memory limit: {MEMORY_LIMIT_GB}GB")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Industries to process: {len(INDUSTRIES_TO_PROCESS)}")
    print(f"Industries: {', '.join(INDUSTRIES_TO_PROCESS)}")
    
    if validate_config():
        print("\n✅ Configuration is valid")
    else:
        print("\n❌ Configuration has errors")













