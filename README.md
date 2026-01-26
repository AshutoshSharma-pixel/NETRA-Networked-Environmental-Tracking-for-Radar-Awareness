# Project NETRA
## Networked Environmental Tracking for Radar Awareness

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

**Project NETRA** is an advanced AI-powered radar simulation and visualization system that combines computer vision, real-time object detection, and professional-grade radar displays to create an immersive tracking experience.

---

## 🎯 Overview

NETRA transforms your camera feed into a sophisticated radar tracking system using:
- **YOLOv8 AI Detection**: State-of-the-art object detection
- **Multi-Object Tracking**: Persistent ID tracking with centroid matching
- **Professional Radar UI**: Patriot-style radar visualization
- **Real-time Analytics**: Threat scoring, velocity computation, and predictive tracking

---

## ✨ Key Features

### 🎨 Radar Visualization
- **Patriot-Style Display**: Professional military-grade radar interface
- **360° Sweep Animation**: Smooth rotating radar beam with motion blur
- **Concentric Range Rings**: Distance indicators with numeric labels
- **Color-Coded Threats**: Dynamic threat assessment (green → yellow → red)
- **Uncertainty Halos**: Visual confidence indicators
- **Fading Trails**: Motion history visualization

### 🎯 Object Tracking
- **Persistent IDs**: Tracks maintain identity across frames
- **Exponential Smoothing**: Reduces position jitter
- **Velocity Vectors**: Real-time speed and direction
- **Ghost Predictions**: Predicted future positions
- **Altitude Estimation**: 3-level altitude classification (LOW/MID/HIGH)
- **Stability Scoring**: Track quality assessment

### 🎛️ Advanced UI
- **Track Details Panel**: Comprehensive track information
- **Event Log System**: Real-time event notifications
- **Elevation Strip**: Vertical position visualization
- **Altitude Stack**: 3D altitude layer display
- **Mini-Zoom View**: Magnified sector view for selected tracks
- **Live Camera Preview**: Real-time video feed overlay

### 🔊 Audio Feedback
- **Track Detection Beeps**: New object alerts
- **Sweep Tick Sounds**: Radar beam crossing notifications
- **Warning Tones**: High-threat approaching object alerts
- **Spatial Audio**: Position-aware sound cues

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- macOS, Linux, or Windows

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AshutoshSharma-pixel/NETRA-Networked-Environmental-Tracking-for-Radar-Awareness-.git
cd NETRA-Networked-Environmental-Tracking-for-Radar-Awareness-
```

2. **Install dependencies**
```bash
pip install -r netra/requirements.txt
```

3. **Run the radar system**
```bash
python netra/vision/radar_cv.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit application |
| **T** | Toggle motion trails |
| **P** | Toggle prediction paths |
| **L** | Toggle object labels |
| **R** | Toggle replay mode |
| **3** | Toggle 3D perspective mode |
| **Left Click** | Select and focus on a track |

---

## 📁 Project Structure

```
Project NETRA/
├── netra/
│   ├── vision/          # Computer vision & radar modules
│   │   └── radar_cv.py  # Main radar visualization system
│   ├── engine/          # Physics simulation engine
│   ├── visuals/         # Rendering components
│   ├── controls/        # Input handling
│   ├── assets/          # Audio and visual assets
│   │   └── sounds/      # Audio feedback files
│   ├── config/          # Configuration files
│   ├── simulation/      # Simulation modules
│   ├── tools/           # Utility scripts
│   ├── tests/           # Test suite
│   └── main.py          # Main entry point
├── yolov8s.pt           # YOLOv8 model weights
├── track_logs.csv       # Exported tracking data
└── README.md            # This file
```

---

## 📊 Data Export

Track data is automatically logged to `track_logs.csv` with the following fields:
- **Track ID**: Unique identifier
- **Timestamp**: Unix timestamp
- **Position (x, y)**: Pixel coordinates
- **Speed**: Velocity magnitude (px/s)
- **Confidence**: Detection confidence (0-1)
- **Threat Score**: Computed threat level (0-1)

---

## 🔧 Configuration

Key parameters in `radar_cv.py`:

```python
SCREEN_SIZE = (900, 700)      # Display resolution
FRAME_WIDTH = 640             # Camera frame width
FRAME_HEIGHT = 480            # Camera frame height
CAMERA_INDEX = 0              # Camera device index
H_FOV_DEGREES = 60.0          # Horizontal field of view
MAX_TRACK_MISSED = 12         # Frames before track deletion
FPS = 30                      # Target frame rate
```

---

## 🎓 Technical Details

### Object Detection
- **Model**: YOLOv8s (small variant)
- **Preprocessing**: CLAHE lighting normalization
- **Confidence Threshold**: Dynamic (0.08 - 0.35)
- **Classes**: 80 COCO object categories

### Tracking Algorithm
- **Method**: Centroid-based matching
- **Smoothing**: Exponential moving average (α = 0.6)
- **Distance Threshold**: Adaptive (6% of diagonal)
- **Classification**: Weighted majority voting with temporal memory

### Threat Assessment
- **Factors**: Confidence (40%), Speed (45%), Stability (15%)
- **Range**: 0.0 (low) to 1.0 (high)
- **Color Mapping**: Green → Yellow → Red gradient

---

## ⚠️ Important Disclaimers

**CRITICAL WARNING: SIMULATION ONLY**

This software is designed for:
- ✅ Educational purposes
- ✅ Research and development
- ✅ Demonstrations and prototyping
- ✅ Computer vision experiments

**NOT suitable for:**
- ❌ Air traffic control
- ❌ Maritime navigation
- ❌ Safety-critical applications
- ❌ Real-world operational use

See [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md) for complete legal disclaimers.

---

## 🛠️ Troubleshooting

### Camera not detected
```bash
# List available cameras (macOS/Linux)
ls /dev/video*

# Try different camera index in radar_cv.py
CAMERA_INDEX = 1  # or 2, 3, etc.
```

### Low frame rate
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Increase `skip_factor` for detection throttling
- Use YOLOv8n (nano) instead of YOLOv8s

### Audio not working
- Ensure sound files exist in `netra/assets/sounds/`
- Check pygame mixer initialization
- Verify audio device permissions

---

## 📝 License

This project is open-source software. See the repository license for details.

**Liability Disclaimer**: This software is provided "AS IS" without warranty of any kind. The authors are not liable for any damages arising from its use.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 👨‍💻 Author

**Ashutosh Sharma**
- GitHub: [@AshutoshSharma-pixel](https://github.com/AshutoshSharma-pixel)

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: Object detection framework
- **OpenCV**: Computer vision library
- **PyGame**: Graphics and audio engine
- **COCO Dataset**: Training data for object detection

---

**Project NETRA** - *Networked Environmental Tracking for Radar Awareness*

*Transforming vision into awareness* 🎯
