# Hurricane Sandy Analysis - Pangu-Weather

## Quick Start

Navigate to the Sandy analysis folder:
```bash
cd /scratch/qhuang62/pangu-extreme/2012_Sandy
```

### 1. Setup GPU Environment
```bash
source ../setup_gpu_environment.sh
```

### 2. Choose Your Forecast Length

**48-hour forecast (original):**
```bash
python run_pangu_sandy.py
```

**7-day forecast (extended):**
```bash
python run_pangu_sandy_7day.py
```

## Folder Structure

```
2012_Sandy/
├── run_pangu_sandy.py          # 48-hour forecast
├── run_pangu_sandy_7day.py     # 168-hour (7-day) forecast
├── sandy_input_data/           # ERA5 initial conditions
│   ├── input_upper.npy
│   └── input_surface.npy
├── sandy_results/              # 48-hour results
└── sandy_results_7day/         # 7-day results
```

## Expected Results

### 48-Hour Forecast
- **Track error**: ~200 km
- **Covers**: Oct 24-26, 2012
- **Results**: Basic hurricane track through Atlantic

### 7-Day Forecast  
- **Track error**: Varies by day
- **Covers**: Oct 24-31, 2012
- **Results**: Complete Sandy lifecycle including:
  - Atlantic tropical phase
  - US East Coast approach  
  - New Jersey/New York landfall
  - Post-tropical transition

## Model Performance

The Pangu-Weather model uses:
- **GPU acceleration** on NVIDIA A100
- **Multi-model strategy**: 1h, 3h, 6h, 24h models
- **Iterative prediction**: Each model updates the next
- **ERA5 reanalysis**: High-quality initial conditions

## Files Generated

### Both forecasts create:
- `hurricane_sandy_track_*.csv` - Track coordinates
- `hurricane_sandy_track_*.png` - Visualization maps
- `prediction_*h_*.npy` - Atmospheric fields

### 7-day forecast additionally provides:
- Daily position markers
- Extended error analysis  
- Post-landfall tracking
- Key forecast milestone fields