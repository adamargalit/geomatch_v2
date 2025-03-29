import cv2
import numpy as np
import math
import time
import os
import rasterio
import rasterio.transform
import warnings
import datetime
import tempfile
import sys
# Removed tempfile import as it wasn't used in the provided snippet

try:
    import config
except ModuleNotFoundError:
    print("ERROR in visual_localizer: Could not import config.")
    config = None

# Import other necessary modules from src...
try:
    from src.map_handler import get_map_info, read_map_chip
    from src.feature_matcher import find_match
    from src.utils import transform_coords, WGS84_EPSG
except ImportError as e:
    print(f"ERROR in visual_localizer: Failed to import helper modules: {e}")
    # Depending on requirements, you might exit or handle this differently
    sys.exit(1) # Example: exit if core modules are missing

# Constants for return values...
RESULT_SUCCESS = 'SUCCESS'
RESULT_MATCH_FAILED = 'MATCH_FAILED'
RESULT_ERROR_MAP_CHIP = 'ERROR_MAP_CHIP'
RESULT_ERROR_IMAGE_LOAD = 'ERROR_IMAGE_LOAD'
RESULT_ERROR_COORDS = 'ERROR_COORDS'
RESULT_ERROR_CONFIG = 'ERROR_CONFIG'
RESULT_ERROR_FILE_IO = 'ERROR_FILE_IO'
RESULT_ERROR_UNKNOWN = 'ERROR_UNKNOWN'

# --- Helper Functions (Copied from main.py) ---

def draw_matches(img1, kp1_np, img2, kp2_np, match_indices_kp1, match_indices_kp2, inliers_mask, max_matches_to_draw=50):
    """
    Draws keypoint matches between two images using SuperGlue results.
    (Code identical to previous version)
    """
    if img1 is None or img2 is None or kp1_np is None or kp2_np is None or match_indices_kp1 is None or match_indices_kp2 is None:
        print("ERROR: Invalid input to draw_matches.")
        return None
    if len(match_indices_kp1) != len(match_indices_kp2):
         print("ERROR: Mismatched length of match indices in draw_matches.")
         return None
    if inliers_mask is not None and len(inliers_mask) != len(match_indices_kp1):
         print(f"ERROR: Mismatched length of inliers_mask ({len(inliers_mask)}) and match indices ({len(match_indices_kp1)}) in draw_matches.")
         inliers_mask = None

    try:
        keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp1_np]
        keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp2_np]
    except (TypeError, IndexError) as e:
        print(f"ERROR converting keypoints to cv2.KeyPoint objects: {e}")
        return None

    inlier_matches = []
    num_matches_total = len(match_indices_kp1)

    if inliers_mask is not None:
        try:
            mask_flat = np.array(inliers_mask).flatten()
            if len(mask_flat) == num_matches_total:
                for i in range(num_matches_total):
                    if mask_flat[i]:
                        idx1 = int(match_indices_kp1[i])
                        idx2 = int(match_indices_kp2[i])
                        if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
                            dmatch = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _imgIdx=0, _distance=0)
                            inlier_matches.append(dmatch)
                        else:
                             print(f"WARNING draw_matches: Invalid match index skipped (queryIdx={idx1} vs {len(keypoints1)}, trainIdx={idx2} vs {len(keypoints2)})")
            else:
                print(f"ERROR: Flattened RANSAC mask length ({len(mask_flat)}) != Number of matches ({num_matches_total}). Drawing no inliers.")
        except Exception as e:
             print(f"ERROR processing inliers_mask: {e}. Drawing no inliers.")
    else:
        print("DEBUG draw_matches: No inlier mask provided. Creating DMatch objects for all potential matches.")
        for i in range(num_matches_total):
            idx1 = int(match_indices_kp1[i])
            idx2 = int(match_indices_kp2[i])
            if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
                dmatch = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _imgIdx=0, _distance=0)
                inlier_matches.append(dmatch)
            else:
                print(f"WARNING draw_matches: Invalid match index skipped (queryIdx={idx1} vs {len(keypoints1)}, trainIdx={idx2} vs {len(keypoints2)})")

    inlier_matches = sorted(inlier_matches, key=lambda x: x.queryIdx)
    matches_to_draw = inlier_matches[:max_matches_to_draw]
    print(f"DEBUG draw_matches: Found {len(inlier_matches)} inlier/potential matches. Drawing {len(matches_to_draw)}.")

    if not matches_to_draw:
         print("DEBUG draw_matches: No valid matches to draw.")
         h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
         vis_h = max(h1, h2); vis_w = w1 + w2
         vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
         vis[:h1, :w1, :] = img1; vis[:h2, w1:w1 + w2, :] = img2
         cv2.putText(vis, "No Matches Found", (vis_w // 2 - 100, vis_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
         return vis

    try:
        img_matches = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, matches_to_draw, None,
            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
            matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return img_matches
    except cv2.error as e:
        print(f"ERROR during cv2.drawMatches: {e}")
        return None
    except Exception as e:
        print(f"UNEXPECTED ERROR during cv2.drawMatches: {e}")
        return None


def get_center_and_rotation(matrix, img_width, img_height):
    """
    Calculates the center coordinates and rotation angle from homography.
    (Code identical to previous version)
    """
    if matrix is None: return None, None, None
    h, w = img_height, img_width
    corners_img1 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    try:
        corners_img2 = cv2.perspectiveTransform(corners_img1, matrix)
        if corners_img2 is None or corners_img2.shape != (4, 1, 2):
             print(f"ERROR: Unexpected shape from perspectiveTransform: {corners_img2.shape if corners_img2 is not None else 'None'}")
             return None, None, None
        center_x = np.mean(corners_img2[:, 0, 0])
        center_y = np.mean(corners_img2[:, 0, 1])
        dx = corners_img2[1, 0, 0] - corners_img2[0, 0, 0]
        dy = corners_img2[1, 0, 1] - corners_img2[0, 0, 1]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        return center_x, center_y, angle_degrees
    except cv2.error as e:
         print(f"OpenCV ERROR calculating center/rotation: {e}")
         return None, None, None
    except Exception as e:
        print(f"ERROR calculating center/rotation from homography: {e}")
        return None, None, None


def draw_bounding_box(image, matrix, source_width, source_height, color=(0, 255, 0), thickness=2):
    """
    Draws the transformed bounding box onto the target image.
    (Code identical to previous version)
    """
    if matrix is None: return image
    h, w = source_height, source_width
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    try:
        dst_pts = cv2.perspectiveTransform(pts, matrix)
        if dst_pts is None:
             print("ERROR: cv2.perspectiveTransform returned None in draw_bounding_box.")
             return image
        dst_pts_int = np.int32(dst_pts)
        if dst_pts_int.shape != (4, 1, 2):
             print(f"ERROR: Unexpected shape for bounding box points: {dst_pts_int.shape}")
             return image
        img_out = cv2.polylines(image.copy(), [dst_pts_int], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img_out
    except cv2.error as e:
         print(f"OpenCV ERROR drawing bounding box: {e}")
         return image
    except Exception as e:
        print(f"ERROR drawing bounding box: {e}")
        return image

# --- Main Localization Function ---

def run_localization(lat, lon, aerial_photo_path, save_output=True, output_dir_override=None):
    """
    Loads images, performs matching, calculates results, saves visualizations,
    writes coordinates to a file, and returns the localization status and result.
    """
    start_time_total = time.time()
    print("--- [Visual Localizer] Initializing ---")

    # --- Initialize result variables ---
    calculated_lon_lat = None
    calculated_angle = None
    center_x_map, center_y_map = None, None
    img_with_box = None
    img_matches_vis = None # <<< FIX: Initialize img_matches_vis to None >>>
    timestamp_now = None # Initialize timestamp

    # --- Validate Config Access ---
    if config is None:
        print(f"ERROR [Visual Localizer]: config module not loaded.")
        return RESULT_ERROR_CONFIG, None
    try:
        offline_map_file = config.OFFLINE_MAP_FILE
        min_match_count = config.MIN_MATCH_COUNT
        base_dir = config.BASE_DIR
        output_coord_filename = config.VISUAL_OUTPUT_COORD_FILE
    except AttributeError as e:
        print(f"ERROR [Visual Localizer]: Missing configuration value in config.py: {e}")
        return RESULT_ERROR_CONFIG, None

    output_coord_filepath = os.path.join(base_dir, output_coord_filename)

    # --- Load Map Info ---
    print(f"Using Map: {offline_map_file}")
    map_info = get_map_info(offline_map_file)
    if not map_info:
        print("ERROR [Visual Localizer]: Failed to get map info.")
        return RESULT_ERROR_MAP_CHIP, None
    print(f"Map CRS: {map_info['crs']}")

    # --- Load Aerial Image ---
    print(f"Loading aerial image from: {aerial_photo_path}")
    img_np = cv2.imread(aerial_photo_path)
    if img_np is None:
        print(f"ERROR [Visual Localizer]: Failed to load aerial image at '{aerial_photo_path}'.")
        return RESULT_ERROR_IMAGE_LOAD, None
    img_h, img_w = img_np.shape[:2]
    img_clean = img_np.copy() # Keep a clean copy
    print(f"Aerial image loaded. Shape: {img_np.shape}")

    # --- Get Map Chip ---
    print(f"\n--- [Visual Localizer] Preparing Map Chip ---")
    print(f"Target center: Lat={lat:.6f}, Lon={lon:.6f}")
    # Consider making chip size configurable via config.py
    chip_height_px = config.MAP_CHIP_SIZE_PX if hasattr(config, 'MAP_CHIP_SIZE_PX') else 500
    chip_width_px = config.MAP_CHIP_SIZE_PX if hasattr(config, 'MAP_CHIP_SIZE_PX') else 500
    print(f"Getting map chip of size {chip_width_px}x{chip_height_px} pixels.")
    map_chip_np, map_chip_transform, map_chip_bounds = read_map_chip(
        offline_map_file, map_info, lat, lon, chip_height_px, chip_width_px
    )
    if map_chip_np is None:
        print("ERROR [Visual Localizer]: Failed to get map chip.")
        return RESULT_ERROR_MAP_CHIP, None
    chip_h, chip_w = map_chip_np.shape[:2]
    map_chip_clean = map_chip_np.copy() # Keep a clean copy
    img_with_box = map_chip_clean.copy() # Initialize image for drawing box
    print(f"Map chip obtained. Shape: {map_chip_np.shape}")

    print("\n--- [DEBUG] Visualizing Grayscale Inputs ---")
    try:
        gray_aerial_display = None
        gray_map_chip_display = None

        # Convert Aerial Image
        if img_np is not None:
            if len(img_np.shape) == 3: # Color
                gray_aerial_display = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            elif len(img_np.shape) == 2: # Already Grayscale
                gray_aerial_display = img_np
            if gray_aerial_display is not None:
                # Optional: Resize for consistent display size
                # gray_aerial_display = cv2.resize(gray_aerial_display, (600, 600))
                cv2.imshow("DEBUG: Grayscale Aerial Input", gray_aerial_display)

        # Convert Map Chip
        if map_chip_np is not None:
            if len(map_chip_np.shape) == 3: # Color (3 or 4 channels)
                if map_chip_np.shape[2] == 4: # BGRA/RGBA
                    gray_map_chip_display = cv2.cvtColor(map_chip_np, cv2.COLOR_BGRA2GRAY)
                else: # BGR/RGB
                    gray_map_chip_display = cv2.cvtColor(map_chip_np, cv2.COLOR_BGR2GRAY)
            elif len(map_chip_np.shape) == 2: # Already Grayscale
                gray_map_chip_display = map_chip_np
            if gray_map_chip_display is not None:
                # Optional: Resize for consistent display size
                # gray_map_chip_display = cv2.resize(gray_map_chip_display, (600, 600))
                cv2.imshow("DEBUG: Grayscale Map Chip Input", gray_map_chip_display)

        # Wait for user key press if any image was shown
        if gray_aerial_display is not None or gray_map_chip_display is not None:
            print(">>> Displaying grayscale images. Press any key in an image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(">>> Continuing localization...")
        else:
            print("No grayscale images generated for display.")

    except Exception as e_gray_vis:
        print(f"ERROR during grayscale visualization: {e_gray_vis}")
        # Ensure windows are closed if error occurred after imshow but before waitKey
        cv2.destroyAllWindows()
    print("--- [DEBUG] End Grayscale Visualization ---\n")

    # --- End of added block ---

    # The call to find_match(...) should come after this block
    # M, inliers_mask_ransac, kp1_np, kp2_np, match_idx_kp1, match_idx_kp2 = find_match(...)

    # --- Perform Feature Matching ---
    print("\n--- [Visual Localizer] Starting Feature Matching ---")
    start_time_match = time.time()
    M, inliers_mask_ransac, kp1_np, kp2_np, match_idx_kp1, match_idx_kp2 = None, None, None, None, None, None
    try:
        M, inliers_mask_ransac, kp1_np, kp2_np, match_idx_kp1, match_idx_kp2 = find_match(
            img_np, map_chip_np, min_match_count=min_match_count
        )
    except Exception as e:
        print(f"ERROR [Visual Localizer]: Exception during find_match: {e}")
        # Return unknown error, but pass None for result_data for consistency
        return RESULT_ERROR_UNKNOWN, None
    match_duration = time.time() - start_time_match
    print(f"Feature matching finished in {match_duration:.2f} seconds.")

    # --- Process Results ---
    if M is not None:
        print("\n--- [Visual Localizer] Homography Found ---")
        center_x_chip, center_y_chip, angle_deg_chip = get_center_and_rotation(M, img_w, img_h)

        if center_x_chip is not None:
            calculated_angle = angle_deg_chip
            print(f"Calculated Center (in map chip pixels): X={center_x_chip:.2f}, Y={center_y_chip:.2f}")
            print(f"Calculated Rotation (degrees): {calculated_angle:.2f}")

            try:
                center_x_map, center_y_map = rasterio.transform.xy(map_chip_transform, center_y_chip, center_x_chip, offset='center')
                print(f"Calculated Center (map CRS - {map_info['crs']}): X={center_x_map:.2f}, Y={center_y_map:.2f}")

                source_epsg_val = None
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module='pyproj')
                        source_epsg_val = map_info['crs'].to_epsg()
                    if source_epsg_val is None: raise ValueError("CRS.to_epsg() returned None")
                except Exception as crs_e:
                    print(f"Warning [Visual Localizer]: Could not get EPSG code from map CRS: {crs_e}. CRS: {map_info['crs']}")

                if source_epsg_val is not None:
                    center_lon_calc, center_lat_calc = transform_coords(
                        center_x_map, center_y_map, source_epsg=source_epsg_val, target_epsg=WGS84_EPSG
                    )
                    if center_lon_calc is not None:
                        calculated_lon_lat = (center_lon_calc, center_lat_calc)
                        print(f"--> Calculated Center (Lon/Lat): Lon={center_lon_calc:.6f}, Lat={center_lat_calc:.6f}")
                    else:
                        print("ERROR [Visual Localizer]: Failed to convert map center to Lon/Lat.")
                        calculated_lon_lat = None # Ensure failure propagates
                else:
                    print("ERROR [Visual Localizer]: Cannot perform Lon/Lat conversion due to missing source EPSG.")
                    calculated_lon_lat = None # Ensure failure propagates

            except (rasterio.errors.RasterioIOError, ImportError, Exception) as e:
                 print(f"ERROR [Visual Localizer]: during coordinate conversion of center: {e}")
                 calculated_lon_lat = None

            # Check if conversion ultimately failed
            if calculated_lon_lat is None:
                print("[Visual Localizer] Coordinate conversion failed after homography.")
                # Return error, include base images
                failure_data = {'error': 'Coordinate conversion failed'}
                if 'img_clean' in locals() and img_clean is not None: failure_data['img_clean'] = img_clean
                if 'map_chip_clean' in locals() and map_chip_clean is not None: failure_data['map_chip_clean'] = map_chip_clean
                return RESULT_ERROR_COORDS, failure_data # <<< MODIFIED: Return dict
            else:
                # Draw bounding box and center point ONLY if coordinates are valid
                # (Keep drawing logic as is)
                img_with_box = draw_bounding_box(img_with_box, M, img_w, img_h, color=(0, 255, 255), thickness=2)
                try:
                    cv2.circle(img_with_box, (int(round(center_x_chip)), int(round(center_y_chip))), radius=5, color=(255, 0, 0), thickness=-1)
                except (ValueError, TypeError):
                    print("Warning: Could not draw center point on map chip.")

        else: # center_x_chip is None
            print("[Visual Localizer] Could not calculate center/rotation from homography matrix.")
            # Return match failed, include base images
            failure_data = {'error': 'Center/rotation calculation failed'}
            if 'img_clean' in locals() and img_clean is not None: failure_data['img_clean'] = img_clean
            if 'map_chip_clean' in locals() and map_chip_clean is not None: failure_data['map_chip_clean'] = map_chip_clean
            return RESULT_MATCH_FAILED, failure_data # <<< MODIFIED: Return dict
        
    else: # M is None
        print("\n--- [Visual Localizer] Homography Not Found ---")
        # Return match failed, include base images
        failure_data = {'error': 'Homography not found (low matches/RANSAC failure)'}
        if 'img_clean' in locals() and img_clean is not None: failure_data['img_clean'] = img_clean
        if 'map_chip_clean' in locals() and map_chip_clean is not None: failure_data['map_chip_clean'] = map_chip_clean
        return RESULT_MATCH_FAILED, failure_data # <<< MODIFIED: Return dict

    # --- Save Outputs (if requested and successful so far) ---
    if save_output and calculated_lon_lat is not None:
        print("\n--- [Visual Localizer] Saving Outputs ---")
        output_dir = None # Initialize
        try:
            # Determine output directory
            output_dir = output_dir_override if output_dir_override else os.path.join(base_dir, "output")
            os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

            # Construct filename prefix
            aerial_fname = os.path.basename(aerial_photo_path)
            prefix = os.path.splitext(aerial_fname)[0]
            status_str = "SUCCESS"
            lat_str = f"{lat:.4f}".replace('.', 'p').replace('-', 'n')
            lon_str = f"{lon:.4f}".replace('.', 'p').replace('-', 'n')
            calc_lat_str = f"{calculated_lon_lat[1]:.4f}".replace('.', 'p').replace('-', 'n')
            calc_lon_str = f"{calculated_lon_lat[0]:.4f}".replace('.', 'p').replace('-', 'n')
            output_prefix_base = f"{prefix}__{status_str}__target_{lat_str}_{lon_str}__calc_{calc_lat_str}_{calc_lon_str}"
            output_prefix = os.path.join(output_dir, output_prefix_base)

            # --- Save map chip with box image ---
            save_path_box = f"{output_prefix}__map_chip_with_box.png"
            if img_with_box is not None:
                try:
                    success_box = cv2.imwrite(save_path_box, img_with_box)
                    if success_box:
                        print(f"Saved bounding box image to: {save_path_box}")
                    else:
                        print(f"ERROR: cv2.imwrite FAILED silently for box image: {save_path_box}")
                except Exception as e_save_box:
                    print(f"ERROR: cv2.imwrite raised exception for box image: {e_save_box}")
            else:
                print("Warning: img_with_box was None, cannot save.")

            # --- Generate and Save matches visualization image ---
            # Regenerate visualization image data
            # (Ensure kp1_np, kp2_np etc. are available from the find_match call)
            if kp1_np is not None and kp2_np is not None:
                 img_matches_vis = draw_matches(
                     img_clean, kp1_np, map_chip_clean, kp2_np,
                     match_idx_kp1, match_idx_kp2, inliers_mask_ransac,
                     max_matches_to_draw=50
                 )
            else:
                 img_matches_vis = None # Ensure it's None if keypoints weren't generated

            save_path_matches = f"{output_prefix}__matches_vis.png"
            if img_matches_vis is not None:
                try:
                    success_matches = cv2.imwrite(save_path_matches, img_matches_vis)
                    if success_matches:
                        print(f"Saved matches visualization to: {save_path_matches}")
                    else:
                        print(f"ERROR: cv2.imwrite FAILED silently for matches image: {save_path_matches}")
                except Exception as e_save_matches:
                    print(f"ERROR: cv2.imwrite raised exception for matches image: {e_save_matches}")
            else:
                print("Skipping save for matches visualization (was None or generation failed).")

        except Exception as e_setup:
            # Catch errors during path/prefix setup or makedirs
            print(f"ERROR [Visual Localizer]: Failed during output saving setup: {e_setup}")
            # Continue to save coordinates if possible, but images failed

        # --- Append Coordinates to File (Still inside `if save_output...`) ---
        write_success = False
        if calculated_lon_lat is not None and calculated_angle is not None:
            try:
                timestamp_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                output_line = (
                    f"{timestamp_now},"
                    f"{calculated_lon_lat[1]:.8f},"
                    f"{calculated_lon_lat[0]:.8f},"
                    f"{calculated_angle:.3f}\n"
                )
                file_exists = os.path.exists(output_coord_filepath)
                needs_header = not file_exists or os.path.getsize(output_coord_filepath) == 0
                with open(output_coord_filepath, 'a', encoding='utf-8') as f:
                    if needs_header:
                        header = "timestamp_utc_iso,latitude,longitude,angle\n"
                        f.write(header)
                    f.write(output_line)
                print(f"Successfully appended coordinates to {output_coord_filepath}")
                write_success = True
            except (IOError, Exception) as e_coord_write:
                print(f"ERROR [Visual Localizer]: Failed to write/append coordinates to {output_coord_filepath}: {e_coord_write}")
                # Don't change overall status, but log the error
                # Optionally: could return RESULT_ERROR_FILE_IO here if critical
        # --- End Append Coordinates ---

    # --- Final Processing Time ---
    total_duration = time.time() - start_time_total
    print(f"\n--- [Visual Localizer] Processing Complete ({total_duration:.2f} seconds) ---")

    # --- Return Result ---
    # Check successful calculation again before returning success
    if calculated_lon_lat is not None and calculated_angle is not None:
        # --- SUCCESS PATH ---
        result_data = {
            'lat': calculated_lon_lat[1],
            'lon': calculated_lon_lat[0],
            'angle': calculated_angle,
            'timestamp': timestamp_now if 'timestamp_now' in locals() else None,
            'img_with_box': img_with_box,
            'img_matches_vis': img_matches_vis
        }
        return RESULT_SUCCESS, result_data
    else:
        # --- FAILURE PATH (Final Check) ---
        print("[Visual Localizer] Reached final return block without success criteria met.")
        # Prepare failure data dictionary
        failure_data = {}
        # Assume img_clean and map_chip_clean were defined earlier in the function scope
        # Add them if they exist and are not None
        if 'img_clean' in locals() and img_clean is not None:
             failure_data['img_clean'] = img_clean
        if 'map_chip_clean' in locals() and map_chip_clean is not None:
             failure_data['map_chip_clean'] = map_chip_clean

        # Determine likely reason based on M and calculated_lon_lat state
        if M is None:
             failure_data['error'] = 'Homography not found (final check)'
             return RESULT_MATCH_FAILED, failure_data # <<< MODIFIED: Return dict
        elif calculated_lon_lat is None: # Should have returned earlier, but as fallback
             failure_data['error'] = 'Coordinate calculation failed (final check)'
             return RESULT_ERROR_COORDS, failure_data # <<< MODIFIED: Return dict
        else: # Unlikely case, but handle it
             failure_data['error'] = 'Unknown processing error before final return'
             # Return dictionary even for unknown error
             return RESULT_ERROR_UNKNOWN, failure_data # <<< MODIFIED: Return dict