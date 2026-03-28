"""
Visualization functions for maps and charts.
"""
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config import FIGURES_DIR

def create_accident_map(df, center=None, zoom_start=7):
    """
    Create an interactive Folium map of accident locations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with latitude, longitude, and severity columns
    center : tuple, optional
        Map center (lat, lon). Defaults to mean of data
    zoom_start : int
        Initial zoom level
    
    Returns:
    --------
    folium.Map
        Interactive map
    """
    if center is None:
        center = [df['latitude'].mean(), df['longitude'].mean()]
    
    m = folium.Map(location=center, zoom_start=zoom_start)
    
    for _, row in df.iterrows():
        color = 'red' if row.get('contains_fatality_words', 0) == 1 else 'blue'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Severity: {'Fatal' if row.get('contains_fatality_words', 0) == 1 else 'Non-Fatal'}<br>Reports: {row.get('n_crash_reports', 1)}"
        ).add_to(m)
    
    return m

def save_map(m, filename='accidents_hotspots.html'):
    """Save map to HTML file."""
    filepath = FIGURES_DIR / filename
    m.save(filepath)
    print(f"✅ Map saved to: {filepath}")
    return filepath

def plot_hour_distribution(df, save=False):
    """Plot accident distribution by hour."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour', data=df)
    plt.title('Road Accidents by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Accidents')
    
    if save:
        plt.savefig(FIGURES_DIR / 'accidents_by_hour.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_day_distribution(df, save=False):
    """Plot accident distribution by day of week."""
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 6))
    sns.countplot(x='day_of_week', data=df, order=order)
    plt.title('Road Accidents by Day of Week')
    plt.xticks(rotation=45)
    
    if save:
        plt.savefig(FIGURES_DIR / 'accidents_by_day.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_severity_pie(df, save=False):
    """Plot pie chart of fatal vs non-fatal accidents."""
    severity_counts = df['contains_fatality_words'].value_counts()
    labels = ['Non-Fatal', 'Fatal']
    
    plt.figure(figsize=(8, 8))
    plt.pie(severity_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Fatal vs Non-Fatal Crashes')
    
    if save:
        plt.savefig(FIGURES_DIR / 'severity_pie.png', dpi=300, bbox_inches='tight')
    plt.show()