# Geo Matcher - Visual Localization Project

This project performs visual localization by matching an input aerial photograph against an offline map chip using feature matching techniques. It's primarily designed for scenarios where GPS might be unreliable. This current version uses data centered around Ramat Hasharon, Tel Aviv District, Israel.

## Features

* Extracts map chips based on initial coordinate estimates.
* Uses SuperPoint and SuperGlue for deep-learning-based feature detection and matching.
* Calculates Homography using RANSAC to find the geometric transformation.
* Determines the aerial photo's center coordinates (Lat/Lon) and rotation relative to the map.
* Appends calculated coordinates (Timestamp, Lat, Lon, Angle) to `visual_location.txt`.
* Saves visualization images (map with bounding box, feature matches) to the `output/` directory.
* Includes optional resource monitoring (CPU/RAM/GPU).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd geo_matcher
    ```
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```
3.  **Activate the environment:**
    * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
    * Linux/macOS (Bash/Zsh): `source .venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: For GPU support, ensure you have compatible NVIDIA drivers and install the CUDA-enabled version of PyTorch - see PyTorch website for instructions).*
5.  **Place Data/Weights:**
    * Add your offline map file (e.g., `.tif`, `.png` with world file) to the `data/offline_map/` directory.
    * Download and place SuperPoint/SuperGlue weights (`.pth` files) into the `weights/` directory.
6.  **Configure:** Edit `config.py` to set the correct paths for `OFFLINE_MAP_FILE`, `SUPERPOINT_WEIGHTS_PATH`, `SUPERGLUE_WEIGHTS_PATH`, and adjust RANSAC parameters if needed. Set `BASE_DIR` to the project root.

## Usage

Run the main script from the project root directory:


Command to run in terminal:
```bash
python main.py [--lat <latitude>] [--lon <longitude>] [--no-save] <path_to_aerial_photo>