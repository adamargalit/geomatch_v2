# Required imports for new functions
import numpy as np
import torch
import cv2

# Existing imports
import pyproj
import rasterio.transform
from affine import Affine # We need this for the test block

# Default EPSG codes
WGS84_EPSG = 4326 # Standard Lat/Lon
ITM_EPSG = 2039   # Israeli Transverse Mercator

# --- NEW: Image Preprocessing for SuperGlue/SuperPoint ---

def process_resize_sg(w, h, resize_max_dim):
    """
    Helper function to calculate new dimensions for resizing,
    maintaining aspect ratio based on the maximum dimension.
    Mimics the resizing logic often used with SuperPoint/SuperGlue.

    Args:
        w (int): Original width.
        h (int): Original height.
        resize_max_dim (int): Target size for the largest dimension.
                              If <= 0, original dimensions are returned.

    Returns:
        tuple: (w_new, h_new) integers.
    """
    if resize_max_dim <= 0: # Treat <= 0 as no resize
        return w, h
    scale = resize_max_dim / max(h, w)
    w_new = int(round(w * scale))
    h_new = int(round(h * scale))

    # Optional: Add warnings if resolution is too small/large, like in original SuperGlue utils
    # if max(w_new, h_new) < 160:
    #     print('Warning: SuperGlue/Point - Input resolution is very small, results may vary')
    # elif max(w_new, h_new) > 2000: # Example threshold
    #     print('Warning: SuperGlue/Point - Input resolution is very large, results may vary')

    return w_new, h_new

def preprocess_image_sg(image_np, resize_max_dim=None, device='cpu'):
    """
    Preprocesses a NumPy image array (BGR, Grayscale, or RGBA) for SuperGlue.
    Converts to Grayscale, resizes, normalizes, and converts to a PyTorch tensor.

    Args:
        image_np (numpy.ndarray): Input image (H, W), (H, W, 3) or (H, W, 4).
        resize_max_dim (int, optional): If set, resizes the image so its
                                         longest dimension is equal to this value.
        device (str): PyTorch device ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Processed image tensor (1, 1, H, W) on the specified device.
    """
       # ===> ADD THIS DEBUG LINE <===
    print(f"DEBUG preprocess_image_sg: Input type={type(image_np)}, ndim={image_np.ndim if image_np is not None else 'None'}, shape={image_np.shape if image_np is not None else 'None'}")
    # =============================
    
    if image_np is None:
        raise ValueError("Input image is None")

    # --- Convert to Grayscale ---
    if image_np.ndim == 3:
        if image_np.shape[2] == 3: # BGR
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        elif image_np.shape[2] == 4: # RGBA
            print("DEBUG preprocess_image_sg: Converting RGBA to Grayscale.")
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Input image has unexpected 3rd dimension: {image_np.shape[2]}")
    elif image_np.ndim == 2: # Already Grayscale
        gray_image = image_np
    else:
        raise ValueError(f"Input image has unexpected number of dimensions: {image_np.ndim}")

    # --- Resize ---
    if resize_max_dim and resize_max_dim > 0:
        h, w = gray_image.shape
        scale = resize_max_dim / max(h, w)
        if scale < 1: # Only downscale, don't upscale
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            gray_image = cv2.resize(gray_image, (w_new, h_new), interpolation=cv2.INTER_AREA)
            print(f"DEBUG preprocess_image_sg: Resized grayscale image to: ({h_new}, {w_new})")


    # --- Normalize and Convert to Tensor ---
    # SuperPoint expects image values in [0, 1]
    norm_image = gray_image.astype(np.float32) / 255.0
    # Add channel and batch dimensions: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(norm_image).unsqueeze(0).unsqueeze(0).to(device)

    return tensor

# --- EXISTING: Coordinate Transformation ---

def transform_coords(lon, lat, source_epsg=WGS84_EPSG, target_epsg=ITM_EPSG):
    """
    Transforms longitude, latitude coordinates from a source CRS to a target CRS.
    (Implementation remains the same)
    """
    try:
        source_crs = pyproj.CRS.from_epsg(source_epsg)
        target_crs = pyproj.CRS.from_epsg(target_epsg)
        # always_xy=True ensures order is (lon, lat) or (x, y)
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        x_target, y_target = transformer.transform(lon, lat)
        print(f"DEBUG: Transformed ({lon:.6f}, {lat:.6f})[EPSG:{source_epsg}] -> ({x_target:.2f}, {y_target:.2f})[EPSG:{target_epsg}]")
        return x_target, y_target
    except Exception as e:
        print(f"ERROR in transform_coords: {e}")
        return None

def coords_to_pixels(x, y, map_transform):
    """
    Converts map coordinates (x, y in the map's CRS) into image pixel coordinates (row, col).
    (Implementation remains the same)
    """
    try:
        # Use rasterio's helper function. It takes lists of x and y coords.
        rows, cols = rasterio.transform.rowcol(map_transform, [x], [y])
        # The result is arrays; get the first element and convert to int.
        row = int(rows[0])
        col = int(cols[0])
        print(f"DEBUG: Converted ({x:.2f}, {y:.2f}) -> Pixel (Row {row}, Col {col})")
        # Add bounds check? Ensure row/col are within image dimensions? (Later)
        return row, col
    except Exception as e:
        print(f"ERROR in coords_to_pixels: {e}")
        return None

# --- Optional: Self-Test Block ---
if __name__ == "__main__":
    print("Running utils.py directly for testing...")

    # Test Coordinate Transformation (WGS84 Lat/Lon to ITM)
    test_lon = 34.84
    test_lat = 32.15
    print(f"\nTesting transform_coords for Lon={test_lon}, Lat={test_lat} (WGS84 -> ITM)")
    itm_coords = transform_coords(test_lon, test_lat) # Use defaults

    if itm_coords:
        itm_x, itm_y = itm_coords
        print(f" -> ITM Result: X={itm_x:.2f}, Y={itm_y:.2f}")

        # Test Pixel Conversion (Requires a map transform)
        test_transform = Affine(1.32, 0.0, 186161.43, 0.0, -1.32, 671009.81)

        print(f"\nTesting coords_to_pixels for ITM X={itm_x:.2f}, Y={itm_y:.2f}")
        print(f"Using map transform: {test_transform}")
        pixel_coords = coords_to_pixels(itm_x, itm_y, test_transform)

        if pixel_coords:
            print(f" -> Pixel Result: Row={pixel_coords[0]}, Col={pixel_coords[1]}")
        else:
            print("coords_to_pixels failed.")
    else:
        print("transform_coords failed.")

    # --- NEW: Test Image Preprocessing ---
    print("\nTesting preprocess_image_sg...")
    try:
        # Create a dummy color image (e.g., 640x480)
        dummy_img_color = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Create a dummy grayscale image
        dummy_img_gray = np.random.randint(0, 256, (400, 600), dtype=np.uint8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        resize_target = 1024 # Example resize dimension

        # Test with color image
        print(f"Processing dummy color image {dummy_img_color.shape} with resize_max_dim={resize_target}")
        tensor_color = preprocess_image_sg(dummy_img_color, resize_target, device)
        print(f" -> Output tensor shape: {tensor_color.shape}, dtype: {tensor_color.dtype}, device: {tensor_color.device}")
        # Check range (should be approx 0 to 1)
        print(f" -> Output tensor min: {tensor_color.min():.4f}, max: {tensor_color.max():.4f}")

        # Test with grayscale image
        print(f"Processing dummy grayscale image {dummy_img_gray.shape} with resize_max_dim={resize_target}")
        tensor_gray = preprocess_image_sg(dummy_img_gray, resize_target, device)
        print(f" -> Output tensor shape: {tensor_gray.shape}, dtype: {tensor_gray.dtype}, device: {tensor_gray.device}")

        # Test with no resize (resize_max_dim = -1)
        print(f"Processing dummy grayscale image {dummy_img_gray.shape} with resize_max_dim=-1 (no resize)")
        tensor_no_resize = preprocess_image_sg(dummy_img_gray, -1, device)
        print(f" -> Output tensor shape: {tensor_no_resize.shape}, dtype: {tensor_no_resize.dtype}, device: {tensor_no_resize.device}")

    except Exception as e:
        print(f"ERROR during image preprocessing test: {e}")