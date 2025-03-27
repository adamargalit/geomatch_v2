import rasterio
import rasterio.windows
import numpy as np
import config # Import config for default EPSG codes or paths if needed elsewhere
from src.utils import transform_coords, coords_to_pixels # Import needed utils

# Define EPSG codes (can also come from config)
WGS84_EPSG = 4326
ITM_EPSG = 2039

def get_map_info(map_path):
    """
    Opens the specified offline map file and reads its georeferencing information.

    Args:
        map_path (str): The full path to the map image file (e.g., PNG, GeoTIFF).

    Returns:
        dict: A dictionary containing map metadata ('crs', 'transform', 'width',
              'height', 'bounds', 'count') or None if loading fails.
    """
    print(f"Map Handler: Attempting to load map info from: {map_path}")
    try:
        with rasterio.open(map_path) as dataset:
            # Extract metadata
            bounds = dataset.bounds
            crs = dataset.crs
            transform = dataset.transform
            width = dataset.width
            height = dataset.height
            count = dataset.count # Number of bands

            # --- Assign CRS if missing ---
            if crs is None:
                try:
                    # Assume ITM/EPSG:2039 based on source (govmap)
                    crs = rasterio.crs.CRS.from_epsg(ITM_EPSG)
                    print(f"INFO: Map CRS not found in file; assigned {crs.to_string()}.")
                except Exception as crs_e:
                    print(f"WARNING: Could not create CRS object for EPSG:{ITM_EPSG}: {crs_e}")
                    # crs remains None
            # --- End CRS Assignment ---

            map_info = {
                "path": map_path, # Store path for convenience
                "bounds": bounds,
                "crs": crs,
                "transform": transform,
                "width": width,
                "height": height,
                "count": count # Store number of bands
            }

            # Optional: Print summary after potential CRS update
            print("-" * 30)
            print("Successfully loaded map information:")
            print(f"  - Bounds (W, S, E, N): {map_info['bounds']}")
            print(f"  - Coordinate System (CRS): {map_info['crs']}")
            print(f"  - Pixel Width: {map_info['width']}")
            print(f"  - Pixel Height: {map_info['height']}")
            print(f"  - Band Count: {map_info['count']}")
            print("-" * 30)

            return map_info

    except rasterio.RasterioIOError as e:
        print(f"!!! ERROR loading map file: {e}")
        print(f"!!! File path attempted: {map_path}")
        return None
    except Exception as e:
        print(f"!!! An unexpected error occurred while loading map info: {e}")
        import traceback
        traceback.print_exc()
        return None


def read_map_chip(map_path, map_info, center_lat, center_lon, chip_height_px, chip_width_px):
    """
    Reads a rectangular chip from the offline map centered at the given Lat/Lon.

    Args:
        map_path (str): Path to the map image file.
        map_info (dict): Dictionary containing map metadata (crs, transform, width, height).
        center_lat (float): Target center latitude (WGS84).
        center_lon (float): Target center longitude (WGS84).
        chip_height_px (int): The desired height of the chip in pixels.
        chip_width_px (int): The desired width of the chip in pixels.

    Returns:
        tuple: (chip_data, chip_transform, chip_bounds)
               - chip_data (numpy.ndarray): Extracted chip (H, W, C) or None.
               - chip_transform (affine.Affine): Geotransform for the extracted chip or None.
               - chip_bounds (rasterio.coords.BoundingBox): Bounds for the extracted chip or None.
    """
    if not map_info or 'crs' not in map_info or 'transform' not in map_info:
        print("ERROR: Invalid or incomplete map_info provided to read_map_chip.")
        return None, None, None

    map_crs = map_info['crs']
    map_transform = map_info['transform']
    map_total_height = map_info['height']
    map_total_width = map_info['width']

    # 1. Convert center Lat/Lon to Map CRS (ITM)
    target_x, target_y = transform_coords(center_lon, center_lat,
                                          source_epsg=WGS84_EPSG,
                                          target_epsg=map_crs.to_epsg()) # Use map's actual CRS EPSG
    if target_x is None:
        print("ERROR: Failed to transform center coordinates in read_map_chip.")
        return None, None, None

    # 2. Convert Map CRS coordinate to center pixel
    center_row, center_col = coords_to_pixels(target_x, target_y, map_transform)
    if center_row is None:
        print("ERROR: Failed to convert center coordinates to pixels in read_map_chip.")
        return None, None, None

    print(f"DEBUG read_map_chip: Target center pixel: Row={center_row}, Col={center_col}")

    # 3. Calculate window boundaries
    row_start = round(center_row - chip_height_px / 2)
    col_start = round(center_col - chip_width_px / 2)

    # Boundary Checks (clip window to map edges)
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    actual_width = min(chip_width_px, map_total_width - col_start)
    actual_height = min(chip_height_px, map_total_height - row_start)

    if actual_width <= 0 or actual_height <= 0:
         print("ERROR: Calculated chip size is zero or negative (center likely outside map).")
         return None, None, None

    # 4. Define the rasterio Window
    window = rasterio.windows.Window(col_start, row_start, actual_width, actual_height)
    print(f"DEBUG read_map_chip: Reading window: {window}")

    # 5. Read data using the window and get window transform/bounds
    try:
        with rasterio.open(map_path) as dataset:
            # Get the transform specific to this window
            window_transform = dataset.window_transform(window)
            # Read the pixel data
            chip_data = dataset.read(window=window)
            # Calculate the geographic bounds of the window
            window_bounds = rasterio.windows.bounds(window, dataset.transform)

            if chip_data is None or chip_data.size == 0:
                print("ERROR: No data read from the specified window.")
                return None, None, None

            # Transpose to (height, width, bands) if necessary
            if chip_data.ndim == 3 and chip_data.shape[0] in [3, 4]:
                 chip_data = np.transpose(chip_data, (1, 2, 0)) # HWC
                 print(f"DEBUG read_map_chip: Chip read (HWC), shape: {chip_data.shape}")
            elif chip_data.ndim == 2:
                 print(f"DEBUG read_map_chip: Chip read (Grayscale), shape: {chip_data.shape}")
            else:
                 print(f"WARNING read_map_chip: Chip read with unexpected shape: {chip_data.shape}")

            # Convert bounds tuple to BoundingBox object for consistency (optional)
            chip_bounds_obj = rasterio.coords.BoundingBox(*window_bounds)

            return chip_data, window_transform, chip_bounds_obj

    except Exception as e:
        print(f"ERROR reading map chip window: {e}")
        return None, None, None

# --- Remove the old Self-Test Block ---
# if __name__ == "__main__":
#     ... (old test code is invalid as get_map_info now needs path) ...