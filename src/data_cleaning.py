"""
Data cleaning and preprocessing functions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .config import DATA_RAW, DATA_PROCESSED

def load_raw_data():
    """Load raw accident data from CSV."""
    file_path = DATA_RAW / 'ma3route_crashes_manualcode.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data not found at {file_path}")
    return pd.read_csv(file_path)

def clean_column_names(df):
    """Clean column names: lowercase, replace spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def convert_datetime(df):
    """Convert date columns to datetime objects."""
    df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    return df

def extract_time_features(df):
    """Extract hour, month, day_of_week, year from datetime."""
    df['hour'] = df['crash_datetime'].dt.hour
    df['month'] = df['crash_datetime'].dt.month
    df['year'] = df['crash_datetime'].dt.year
    df['day_of_week'] = df['crash_datetime'].dt.day_name()
    return df

def create_severity_label(df):
    """Create human-readable severity labels."""
    df['severity'] = df['contains_fatality_words'].map({1: 'Fatal', 0: 'Non-Fatal'})
    return df

def classify_road_user(row):
    """Classify road user type based on indicator columns."""
    if row['contains_pedestrian_words'] == 1:
        return 'Pedestrian'
    elif row['contains_motorcycle_words'] == 1:
        return 'Motorcycle'
    elif row['contains_matatu_words'] == 1:
        return 'Matatu'
    else:
        return 'Other'

def add_road_user_type(df):
    """Add road_user_type column."""
    df['road_user_type'] = df.apply(classify_road_user, axis=1)
    return df

def prepare_dashboard_data(df):
    """Prepare data for dashboard export."""
    # Make a copy to avoid modifying original
    df_dashboard = df.copy()
    
    # Apply transformations
    df_dashboard = clean_column_names(df_dashboard)
    df_dashboard = convert_datetime(df_dashboard)
    df_dashboard = extract_time_features(df_dashboard)
    df_dashboard = create_severity_label(df_dashboard)
    df_dashboard = add_road_user_type(df_dashboard)
    
    return df_dashboard

def export_dashboard_data(df, filename=None):
    """Export dashboard data to CSV."""
    if filename is None:
        filename = DATA_PROCESSED / 'kenya_road_accidents_dashboard.csv'
    else:
        filename = Path(filename)
    
    df.to_csv(filename, index=False)
    print(f"✅ Dashboard data exported to: {filename}")
    return filename

def get_dashboard_columns():
    """Return the standard columns for dashboard export."""
    return [
        'crash_id', 'crash_datetime', 'crash_date', 'hour', 'month', 'year',
        'day_of_week', 'latitude', 'longitude', 'severity', 'contains_fatality_words',
        'contains_pedestrian_words', 'contains_motorcycle_words', 'contains_matatu_words',
        'road_user_type', 'n_crash_reports'
    ]