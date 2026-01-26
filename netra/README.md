# Project NETRA
## Networked Environmental Tracking for Radar Awareness

**Project NETRA** is an advanced radar simulation and visualization system that uses computer vision and AI-powered object detection to create a professional-grade radar interface.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for accurate object detection from camera feed
- **Radar Visualization**: Professional Patriot-style radar display with:
  - Centroid-based multi-object tracking with persistent IDs
  - Exponential smoothing to reduce jitter
  - Velocity computation and predicted ghost positions
  - Threat scoring based on confidence, speed, and stability
  - Color-coded threat levels
  - Fading trails and uncertainty halos
  
- **Advanced UI Features**:
  - Track details panel
  - Event log system
  - Elevation strip visualization
  - Altitude stack display
  - Mini-zoom for selected sectors
  - Real-time camera preview

- **Interactive Controls**:
  - **ESC** - Quit application
  - **T** - Toggle trails
  - **P** - Toggle prediction paths
  - **L** - Toggle labels
  - **R** - Toggle replay mode
  - **3** - Toggle 3D mode
  - **Left Click** - Select track

- **Audio Feedback**: Spatial audio cues for track detection and warnings

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a camera connected (default: camera index 0)

## Usage

Run the radar visualization:
```bash
python netra/vision/radar_cv.py
```

Or run the main simulation:
```bash
python netra/main.py
```

## Project Structure

- `vision/` - Computer vision and radar visualization modules
- `engine/` - Physics simulation engine
- `visuals/` - Rendering components
- `controls/` - Input handling
- `assets/` - Audio and visual assets
- `config/` - Configuration files
- `logs/` - Track logs and data exports

## Output

Track logs are automatically saved to `track_logs.csv` upon exit, containing:
- Track ID
- Timestamp
- Position (x, y)
- Speed
- Confidence
- Threat score

## Requirements

- Python 3.8+
- OpenCV
- PyGame
- Ultralytics YOLOv8
- NumPy

## License

See TERMS_OF_SERVICE.md for usage restrictions and disclaimers.

**⚠️ IMPORTANT**: This is a simulation and visualization tool for educational and research purposes only. NOT for real-world safety-critical applications.
