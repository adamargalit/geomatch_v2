import os
# We removed 'dotenv' import as we aren't loading API keys from .env for now

# --- File Paths ---
# Get the absolute path of the project's root directory (geo_matcher)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is geo_matcher/
DATA_DIR = os.path.join(BASE_DIR, "data")

# ==> Path to your manually downloaded map file <==
# This MUST point to the .png file. Make sure the filename here matches
# the actual name of your .png file in the data/offline_map folder.
OFFLINE_MAP_FILE = os.path.join(DATA_DIR, "offline_map", "ramat_hasharon_map.png") # <-- ADJUST FILENAME if needed

# This path is for later when you provide the drone/aerial photos
INPUT_AERIAL_DIR = os.path.join(DATA_DIR, "input_aerial_photos")


# --- Model Weights Paths ---
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
# Adjust filenames below if your downloaded files have different names
SUPERPOINT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "superpoint_v1.pth")
SUPERGLUE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "superglue_outdoor.pth")

# Check if weight files exist (optional but helpful)
if not os.path.exists(SUPERPOINT_WEIGHTS_PATH):
    print(f"WARNING: SuperPoint weights not found at: {SUPERPOINT_WEIGHTS_PATH}")
if not os.path.exists(SUPERGLUE_WEIGHTS_PATH):
    print(f"WARNING: SuperGlue weights not found at: {SUPERGLUE_WEIGHTS_PATH}")

# --- Offline Map Information ---
# These are placeholders for now. We will write Python code using the 'rasterio'
# library to read the actual geographic bounds and coordinate system
# directly from your map files (using the .png and .pgw).
OFFLINE_MAP_BOUNDS = {
    "west": None,   # To be determined from file (Longitude/Easting West edge)
    "south": None,  # To be determined from file (Latitude/Northing South edge)
    "east": None,   # To be determined from file (Longitude/Easting East edge)
    "north": None   # To be determined from file (Latitude/Northing North edge)
}
OFFLINE_MAP_CRS = None # Coordinate Reference System (To be determined from file, likely ITM)


# --- Feature Matching (Placeholders for later steps) ---
# These settings control how the image matching will work later.
MIN_MATCH_COUNT = 10          # Minimum number of feature matches needed to consider it a valid localisation
RANSAC_REPROJ_THRESHOLD = 5.0 # RANSAC reprojection error threshold (pixels) - typical default
RANSAC_CONFIDENCE = 0.999     # RANSAC confidence level - typical default
# --- Check if the map file exists (for basic verification) ---
# This code runs when the config is loaded, just to warn you if the file is missing.
if not os.path.exists(OFFLINE_MAP_FILE):
    print(f"WARNING: Config - Offline map file not found at: {OFFLINE_MAP_FILE}")
    print("Please ensure you have placed the .png (and .pgw) file correctly.")
else:
    # This print statement might run multiple times if config is imported elsewhere
    # print(f"Config - Offline map file found: {OFFLINE_MAP_FILE}")
    pass # Keep it quiet for now

# File to store the latest visually calculated coordinates
# Placed in BASE_DIR (project root) by default for easy access by other processes
VISUAL_OUTPUT_COORD_FILE = "visual_location.txt"