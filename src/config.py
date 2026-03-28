"""
Configuration module for managing project paths and settings.
"""
from pathlib import Path

# Project root (src folder is inside project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

# Model paths
MODELS_DIR = PROJECT_ROOT / 'models'

# Output paths
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
SHAPEFILE_DIR = PROJECT_ROOT / 'shapefile'

# Dashboard data file
DASHBOARD_FILE = DATA_PROCESSED / 'kenya_road_accidents_dashboard.csv'

# Raw data file
RAW_DATA_FILE = DATA_RAW / 'ma3route_crashes_manualcode.csv'

# Create directories if they don't exist
def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, SHAPEFILE_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Auto-create directories when module is imported
ensure_directories()