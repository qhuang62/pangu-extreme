#!/usr/bin/env python3
"""
Hurricane Sandy 7-Day Track Prediction using Pangu-Weather Model
Extended version with 168-hour forecast capability
Updated for reorganized folder structure
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import onnxruntime as ort
import matplotlib.pyplot as plt

print("=== Hurricane Sandy 7-Day Track Prediction - Pangu-Weather ===")
print("Extended forecast: 168 hours (7 days)")

# Set up GPU environment check
if 'CUDA_HOME' not in os.environ:
    print("⚠️  CUDA_HOME not found. Please run:")
    print("   source ../setup_gpu_environment.sh")
    print("   before running this script")
    exit(1)
else:
    print(f"✅ CUDA_HOME: {os.environ['CUDA_HOME']}")

print(f"✅ ONNX Runtime version: {ort.__version__}")
print(f"✅ Available providers: {ort.get_available_providers()}")

# Hurricane Tracker Class
class HurricaneTracker:
    """Enhanced hurricane tracker for extended Pangu predictions"""
    
    def __init__(self, init_lat, init_lon, init_time):
        self.tracked_times = [init_time]
        self.tracked_lats = [init_lat]
        self.tracked_lons = [init_lon]
        
    def find_pressure_minimum(self, mslp_field, lat_grid, lon_grid, search_radius=8.0):
        """Find pressure minimum near last known position"""
        last_lat, last_lon = self.tracked_lats[-1], self.tracked_lons[-1]
        
        # Create search mask - both lat_grid and lon_grid are 2D
        lat_mask = (np.abs(lat_grid - last_lat) <= search_radius)
        lon_mask = (np.abs(lon_grid - last_lon) <= search_radius)
        
        # Handle longitude wraparound
        if last_lon < search_radius:
            lon_mask = lon_mask | (lon_grid >= (360 - search_radius))
        elif last_lon > (360 - search_radius):
            lon_mask = lon_mask | (lon_grid <= search_radius)
            
        # Combine masks - only search where both lat and lon are in range
        search_mask = lat_mask & lon_mask
        
        # Find minimum pressure in search region
        search_region = mslp_field.copy()
        search_region[~search_mask] = np.inf
        
        min_idx = np.unravel_index(np.argmin(search_region), search_region.shape)
        new_lat = lat_grid[min_idx[0], min_idx[1]]
        new_lon = lon_grid[min_idx[0], min_idx[1]]
        
        return new_lat, new_lon
        
    def step(self, prediction_output, current_time):
        """Update track with new prediction"""
        # Extract MSL pressure from Pangu output
        mslp = prediction_output[1][0]  # Surface variables, first variable (MSLP)
        
        # Create lat/lon grids (ERA5 0.25° resolution)
        lats = np.arange(90, -90.25, -0.25)  # 90 to -90, 721 points
        lons = np.arange(0, 360, 0.25)       # 0 to 359.75, 1440 points
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Find new hurricane center
        new_lat, new_lon = self.find_pressure_minimum(mslp, lat_grid, lon_grid)
        
        # Update track
        self.tracked_times.append(current_time)
        self.tracked_lats.append(float(new_lat))
        self.tracked_lons.append(float(new_lon))
        
    def results(self):
        """Return track as DataFrame"""
        return pd.DataFrame({
            'time': self.tracked_times,
            'lat': self.tracked_lats, 
            'lon': self.tracked_lons
        })

# Set up ONNX Runtime for GPU
print("\n=== Setting up ONNX Runtime ===")
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 1

cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

# Load Pangu-Weather models (24h, 6h, 3h, 1h) - updated paths
print("Loading Pangu-Weather models...")
try:
    ort_session_24 = ort.InferenceSession('../model/pangu_weather_24.onnx', sess_options=options, 
                                         providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
    ort_session_6 = ort.InferenceSession('../model/pangu_weather_6.onnx', sess_options=options,
                                        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
    ort_session_3 = ort.InferenceSession('../model/pangu_weather_3.onnx', sess_options=options,
                                        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
    ort_session_1 = ort.InferenceSession('../model/pangu_weather_1.onnx', sess_options=options,
                                        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
    
    print("✅ All Pangu-Weather models loaded successfully!")
    
    # Check if GPU is actually being used
    if 'CUDAExecutionProvider' in ort_session_24.get_providers():
        print("🚀 GPU acceleration is ACTIVE!")
    else:
        print("⚠️  Running on CPU (GPU acceleration not available)")
    
    print(f"Active providers: {ort_session_24.get_providers()}")
    
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    exit(1)

# Load initial conditions - updated paths
print("\n=== Loading Initial Conditions ===")
try:
    input_upper = np.load('sandy_input_data/input_upper.npy').astype(np.float32)
    input_surface = np.load('sandy_input_data/input_surface.npy').astype(np.float32)
    
    print(f"✅ Initial conditions loaded:")
    print(f"   Upper shape: {input_upper.shape}")
    print(f"   Surface shape: {input_surface.shape}")
except Exception as e:
    print(f"❌ Failed to load initial conditions: {e}")
    print("Make sure ERA5 data has been processed and saved to sandy_input_data/")
    exit(1)

# Initialize hurricane tracker
print("\n=== Initializing Hurricane Tracker ===")
tracker = HurricaneTracker(
    init_lat=16.6, 
    init_lon=283.1, 
    init_time=datetime(2012, 10, 24, 12, 0)
)
print(f"Tracker initialized at: {tracker.tracked_times[0]}")
print(f"Initial position: {tracker.tracked_lats[0]}°N, {360-tracker.tracked_lons[0]}°W")

# Run Extended Pangu prediction (7 days = 168 hours)
print("\n=== Running Pangu-Weather 7-Day Hurricane Prediction ===")
print("Forecast strategy:")
print("  - 24h model: Every 24 hours (primary backbone)")
print("  - 6h model: Fill 6-hour intervals")
print("  - 3h and 1h models: For sub-daily resolution")

predictions = []
current_time = datetime(2012, 10, 24, 12, 0)

# Initialize inputs for different models
input_upper_24, input_surface_24 = input_upper.copy(), input_surface.copy()
input_upper_6, input_surface_6 = input_upper.copy(), input_surface.copy()
input_upper_3, input_surface_3 = input_upper.copy(), input_surface.copy()
input_upper_1, input_surface_1 = input_upper.copy(), input_surface.copy()

# Extended forecast: 168 hours = 28 steps of 6 hours each
total_steps = 28  # 7 days * 4 steps per day (6-hour intervals)

for step in range(total_steps):
    forecast_hour = (step + 1) * 6  # 6, 12, 18, 24, 30, 36, ..., 168
    current_time += timedelta(hours=6)
    
    # Enhanced iterative strategy for 7-day forecast
    if forecast_hour % 24 == 0:  # Use 24-hour model every 24 hours
        day = forecast_hour // 24
        print(f"Day {day} (Step {step+1}): Using 24h model for {forecast_hour}h forecast")
        output_upper, output_surface = ort_session_24.run(
            None, {'input': input_upper_24, 'input_surface': input_surface_24}
        )
        # Update all model inputs from 24h prediction
        input_upper_24, input_surface_24 = output_upper, output_surface
        input_upper_6, input_surface_6 = output_upper, output_surface
        input_upper_3, input_surface_3 = output_upper, output_surface
        input_upper_1, input_surface_1 = output_upper, output_surface
        
    elif forecast_hour % 6 == 0:  # Use 6-hour model for intermediate steps
        print(f"Step {step+1}: Using 6h model for {forecast_hour}h forecast")
        output_upper, output_surface = ort_session_6.run(
            None, {'input': input_upper_6, 'input_surface': input_surface_6}
        )
        # Update 6h and shorter-term model inputs
        input_upper_6, input_surface_6 = output_upper, output_surface
        input_upper_3, input_surface_3 = output_upper, output_surface
        input_upper_1, input_surface_1 = output_upper, output_surface
        
    elif forecast_hour % 3 == 0:  # Use 3-hour model if needed
        print(f"Step {step+1}: Using 3h model for {forecast_hour}h forecast")
        output_upper, output_surface = ort_session_3.run(
            None, {'input': input_upper_3, 'input_surface': input_surface_3}
        )
        input_upper_3, input_surface_3 = output_upper, output_surface
        input_upper_1, input_surface_1 = output_upper, output_surface
        
    else:  # Use 1-hour model for highest resolution
        print(f"Step {step+1}: Using 1h model for {forecast_hour}h forecast")
        output_upper, output_surface = ort_session_1.run(
            None, {'input': input_upper_1, 'input_surface': input_surface_1}
        )
        input_upper_1, input_surface_1 = output_upper, output_surface
    
    # Store prediction
    predictions.append((output_upper, output_surface, current_time))
    
    # Update hurricane track
    tracker.step((output_upper, output_surface), current_time)
    
    # Print progress every 12 hours
    if forecast_hour % 12 == 0:
        print(f"  → {current_time}: Hurricane at {tracker.tracked_lats[-1]:.1f}°N, {360-tracker.tracked_lons[-1]:.1f}°W")

print(f"\n✅ Pangu-Weather 7-day prediction completed!")
print(f"Generated {len(predictions)} forecast steps over {total_steps*6} hours")

# Get results
track = tracker.results()
print(f"\n=== Final 7-Day Track Summary ===")
print("Key forecast positions:")
for day in range(8):  # Days 0-7
    day_time = datetime(2012, 10, 24, 12, 0) + timedelta(days=day)
    closest_point = track[track['time'] <= day_time].iloc[-1] if len(track[track['time'] <= day_time]) > 0 else track.iloc[0]
    print(f"Day {day}: {closest_point['time'].strftime('%Y-%m-%d %H:%M')} - {closest_point['lat']:.1f}°N, {360-closest_point['lon']:.1f}°W")

# Observational track from IBTrACS - complete lifecycle (within our 7-day forecast range)
obs_data_extended = [
    ("2012-10-24 12:00", 16.6, 283.1),  # Initialization point
    ("2012-10-24 18:00", 17.7, 283.3),
    ("2012-10-25 00:00", 18.9, 283.6),
    ("2012-10-25 06:00", 20.1, 284.0),
    ("2012-10-25 12:00", 21.7, 284.5),
    ("2012-10-25 18:00", 23.3, 284.7),
    ("2012-10-26 00:00", 24.8, 284.1),
    ("2012-10-26 06:00", 25.7, 283.6),
    ("2012-10-26 12:00", 26.4, 283.1),
    ("2012-10-26 18:00", 27.0, 282.8),
    ("2012-10-27 00:00", 27.5, 282.9),
    ("2012-10-27 06:00", 28.1, 283.1),
    ("2012-10-27 12:00", 28.8, 283.5),
    ("2012-10-27 18:00", 29.7, 284.4),
    ("2012-10-28 00:00", 30.5, 285.3),
    ("2012-10-28 06:00", 31.3, 286.1),
    ("2012-10-28 12:00", 32.0, 287.0),
    ("2012-10-28 18:00", 32.8, 288.0),
    ("2012-10-29 00:00", 33.9, 289.0),  # Final approach
    ("2012-10-29 06:00", 35.3, 289.5),
    ("2012-10-29 12:00", 36.9, 289.0),
    ("2012-10-29 18:00", 38.3, 286.8),  # Near landfall
    ("2012-10-30 00:00", 39.5, 285.5),  # US Landfall
    ("2012-10-30 06:00", 39.9, 283.8),
    ("2012-10-30 12:00", 40.1, 282.2),
    ("2012-10-30 18:00", 40.4, 281.1),
    ("2012-10-31 00:00", 40.7, 280.2),  # Final position
]

obs_df_extended = pd.DataFrame(obs_data_extended, columns=['datetime', 'lat', 'lon'])
obs_df_extended['datetime'] = pd.to_datetime(obs_df_extended['datetime'])

# Calculate errors at different time intervals
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km using Haversine formula"""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

print(f"\n=== Track Error Analysis ===")
error_times = [24, 48, 72, 96, 120, 144]  # Hours
for hours in error_times:
    forecast_time = datetime(2012, 10, 24, 12, 0) + timedelta(hours=hours)
    
    # Find closest predicted point
    pred_point = track[track['time'] <= forecast_time]
    if len(pred_point) > 0:
        pred_point = pred_point.iloc[-1]
        
        # Find closest observed point
        obs_point = obs_df_extended[obs_df_extended['datetime'] <= forecast_time]
        if len(obs_point) > 0:
            obs_point = obs_point.iloc[-1]
            
            error = calculate_distance(pred_point.lat, pred_point.lon, obs_point.lat, obs_point.lon)
            print(f"{hours:3d}h: {error:6.1f} km - Obs: {obs_point.lat:.1f}°N,{360-obs_point.lon:.1f}°W | Pred: {pred_point.lat:.1f}°N,{360-pred_point.lon:.1f}°W")

# Save results
print(f"\n=== Saving 7-Day Results ===")
os.makedirs('sandy_results_7day', exist_ok=True)
track.to_csv('sandy_results_7day/hurricane_sandy_7day_track_prediction.csv', index=False)
print("✅ 7-day track saved to: sandy_results_7day/hurricane_sandy_7day_track_prediction.csv")

# Save prediction fields at key times
key_times = [24, 72, 120, 168]  # 1, 3, 5, 7 days
for hours in key_times:
    if hours//6 <= len(predictions):
        idx = hours//6 - 1
        np.save(f'sandy_results_7day/prediction_{hours}h_upper.npy', predictions[idx][0])
        np.save(f'sandy_results_7day/prediction_{hours}h_surface.npy', predictions[idx][1])

print("✅ Key prediction fields saved to: sandy_results_7day/")

# Create extended visualization
print(f"\n=== Creating 7-Day Track Visualization ===")
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.lines import Line2D
    
    # Create comprehensive 7-day track comparison - exact same layout as Aurora
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Use exact same extent as Aurora version
    ax.set_extent([270, 310, 10, 50], crs=ccrs.PlateCarree())  # Full Atlantic basin
    
    # Geographic features - exact same as Aurora
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.8)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5dc', edgecolor='0.5')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, color="0.4")
    ax.coastlines('50m', linewidth=0.8)
    
    # Add gridlines - exact same as Aurora
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='0.5', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot observed track (black solid line) - exact same style as Aurora
    ax.plot(obs_df_extended.lon, obs_df_extended.lat, 'k-', linewidth=4, 
            transform=ccrs.PlateCarree(), label='Observed Track', zorder=6)
    ax.scatter(obs_df_extended.lon, obs_df_extended.lat, c='black', s=25, 
              edgecolors='white', linewidth=1, transform=ccrs.PlateCarree(), zorder=7)
    
    # Plot predicted track (GREEN) - full 7 days
    ax.plot(track.lon, track.lat, color='green', linestyle='-', linewidth=3, marker='o', markersize=3,
            transform=ccrs.PlateCarree(), label='Pangu 7-Day Prediction', zorder=5)
    
    # Mark initialization point - same style as Aurora
    init_lon, init_lat = 283.1, 16.6
    ax.plot(init_lon, init_lat, '^', markersize=15, color='green', markeredgecolor='white', 
            markeredgewidth=2, transform=ccrs.PlateCarree(), zorder=8)
    
    # Mark final prediction - same style as Aurora
    final_point = track.iloc[-1]
    ax.plot(final_point.lon, final_point.lat, 's', markersize=12, color='green', 
            markeredgecolor='black', markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=8)
    
    # Add major US cities and landmarks - exact same as Aurora
    cities = [
        ("Miami", 25.7617, 279.8269),
        ("New York", 40.7128, 286.0060), 
        ("DC", 38.9072, 282.9401),
        ("Boston", 42.3601, 288.9740),
        ("Norfolk", 36.8485, 283.8839),
        ("Charleston", 32.7765, 280.0316),
        ("Bermuda", 32.3078, 295.2361),
        ("Kingston, Jamaica", 17.9712, 283.2064),
        ("Havana, Cuba", 23.1136, 277.6334)
    ]
    
    for name, lat, lon in cities:
        ax.plot(lon, lat, marker='*', markersize=8, color='darkred', 
                markeredgecolor='black', markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=6)
        ax.text(lon + 0.8, lat, name, transform=ccrs.PlateCarree(),
                fontsize=9, ha='left', va='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='0.4', alpha=0.9), zorder=6)
    
    # Create legend - same style as Aurora
    legend_handles = [
        Line2D([0], [0], color='black', linewidth=4, label='Observed Track'),
        Line2D([0], [0], color='green', linewidth=3, linestyle='-', 
               marker='o', markersize=3, label='Pangu 7-Day Prediction')
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=12, frameon=True,
             fancybox=True, shadow=True)
    
    # Set title - same style as Aurora but for Pangu
    ax.set_title('Pangu 7-Day Forecast: Hurricane Sandy 2012 (Init: Oct 24)', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('sandy_results_7day/hurricane_sandy_7day_track_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 7-day track visualization saved to: sandy_results_7day/hurricane_sandy_7day_track_comparison.png")
    
except ImportError as e:
    print(f"⚠️  Cartopy not available: {e}")
    # Create simple matplotlib fallback
    plt.figure(figsize=(14, 10))
    plt.plot([360-lon for lon in obs_df_extended.lon], obs_df_extended.lat, 'k-', 
             linewidth=3, label='Observed Track', marker='o')
    plt.plot([360-lon for lon in track.lon], track.lat, 'b--', 
             linewidth=3, label='Pangu 7-Day Prediction', marker='x')
    plt.xlabel('Longitude (°W)')
    plt.ylabel('Latitude (°N)')
    plt.title('Hurricane Sandy 7-Day Track Prediction\nPangu-Weather vs Observations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.savefig('sandy_results_7day/hurricane_sandy_7day_track_simple.png', dpi=300, bbox_inches='tight')
    print("✅ Simple 7-day track plot saved")

print(f"\n🎉 Hurricane Sandy 7-day prediction completed successfully!")
print(f"   - Used GPU acceleration: {'✅' if 'CUDAExecutionProvider' in ort_session_24.get_providers() else '❌'}")
print(f"   - Total forecast length: 168 hours (7 days)")
print(f"   - Prediction steps: {len(predictions)}")
print(f"   - Results saved to: sandy_results_7day/")
print(f"   - Covers Sandy's full lifecycle including US landfall")