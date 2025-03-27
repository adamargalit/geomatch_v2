import cv2
import numpy as np
import math
import time # For timing analysis
import argparse # For command-line arguments
import os # For creating directories
import rasterio # <--- Added import
import rasterio.transform # <--- Added import

# Assume config.py exists and defines BASE_DIR, OFFLINE_MAP_FILE, MIN_MATCH_COUNT
import config # Project configuration (paths, settings)
# Correctly import only the functions needed from map_handler
# Assume src/map_handler.py exists with get_map_info, read_map_chip
from src.map_handler import get_map_info, read_map_chip
# Use the updated feature matcher
# Assume src/feature_matcher.py exists with find_match
from src.feature_matcher import find_match
# Keep utils for coordinate transformations
# Assume src/utils.py exists with transform_coords, coords_to_pixels, WGS84_EPSG, ITM_EPSG
from src.utils import transform_coords, coords_to_pixels, WGS84_EPSG, ITM_EPSG

def draw_matches(img1, kp1_np, img2, kp2_np, match_indices_kp1, match_indices_kp2, inliers_mask, max_matches_to_draw=50):
    """
    Draws keypoint matches between two images using SuperGlue results.
    (Code for draw_matches remains the same as your pasted version - no changes needed here)
    """
    if img1 is None or img2 is None or kp1_np is None or kp2_np is None or match_indices_kp1 is None or match_indices_kp2 is None:
        print("ERROR: Invalid input to draw_matches.")
        return None
    if len(match_indices_kp1) != len(match_indices_kp2):
         print("ERROR: Mismatched length of match indices in draw_matches.")
         return None
    # Use .any() for numpy array mask check if necessary, but RANSAC mask is usually list/tuple
    # Check if inliers_mask is not None before checking its length
    if inliers_mask is not None and len(inliers_mask) != len(match_indices_kp1):
         print(f"ERROR: Mismatched length of inliers_mask ({len(inliers_mask)}) and match indices ({len(match_indices_kp1)}) in draw_matches.")
         # Fallback: Treat mask as None if lengths don't match
         inliers_mask = None

    # Convert NumPy keypoints (x, y coordinates) to cv2.KeyPoint objects
    # Ensure kp1_np and kp2_np are iterable and contain tuples/lists of coordinates
    try:
        keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp1_np]
        keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp2_np]
    except (TypeError, IndexError) as e:
        print(f"ERROR converting keypoints to cv2.KeyPoint objects: {e}")
        print(f"kp1_np type: {type(kp1_np)}, kp2_np type: {type(kp2_np)}")
        # Optionally print first few elements if they exist
        if kp1_np is not None and len(kp1_np) > 0: print(f"First kp1_np element: {kp1_np[0]}")
        if kp2_np is not None and len(kp2_np) > 0: print(f"First kp2_np element: {kp2_np[0]}")
        return None


    # Create a list of cv2.DMatch objects *only for the inliers*
    inlier_matches = []
    num_matches_total = len(match_indices_kp1)

    if inliers_mask is not None:
        # Ensure inliers_mask is boolean or can be evaluated as such
        try:
            # Flatten if it's a nested list/array from findHomography
            mask_flat = np.array(inliers_mask).flatten()
            if len(mask_flat) == num_matches_total:
                for i in range(num_matches_total):
                    if mask_flat[i]: # Check if this match is an inlier
                         # Validate indices before creating DMatch
                        idx1 = int(match_indices_kp1[i])
                        idx2 = int(match_indices_kp2[i])
                        if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
                            dmatch = cv2.DMatch(_queryIdx=idx1,
                                                _trainIdx=idx2,
                                                _imgIdx=0, _distance=0)
                            inlier_matches.append(dmatch)
                        else:
                             print(f"WARNING draw_matches: Invalid match index skipped (queryIdx={idx1} vs {len(keypoints1)}, trainIdx={idx2} vs {len(keypoints2)})")

            else:
                print(f"ERROR: Flattened RANSAC mask length ({len(mask_flat)}) != Number of matches ({num_matches_total}). Drawing no inliers.")
                # Proceed with empty inlier_matches list
        except Exception as e:
             print(f"ERROR processing inliers_mask: {e}. Drawing no inliers.")
             # Proceed with empty inlier_matches list

    else:
        # If no mask (e.g., homography failed but we still have matches)
        print("DEBUG draw_matches: No inlier mask provided. Creating DMatch objects for all potential matches.")
        for i in range(num_matches_total):
             # Validate indices before creating DMatch
            idx1 = int(match_indices_kp1[i])
            idx2 = int(match_indices_kp2[i])
            if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
                dmatch = cv2.DMatch(_queryIdx=idx1,
                                    _trainIdx=idx2,
                                    _imgIdx=0, _distance=0)
                inlier_matches.append(dmatch)
            else:
                print(f"WARNING draw_matches: Invalid match index skipped (queryIdx={idx1} vs {len(keypoints1)}, trainIdx={idx2} vs {len(keypoints2)})")


    # Limit the number of matches drawn for clarity
    # Sort by queryIdx (arbitrary but consistent) if distance isn't meaningful
    inlier_matches = sorted(inlier_matches, key=lambda x: x.queryIdx)
    matches_to_draw = inlier_matches[:max_matches_to_draw]

    print(f"DEBUG draw_matches: Found {len(inlier_matches)} inlier/potential matches. Drawing {len(matches_to_draw)}.")

    # Draw the matches only if there are matches to draw
    if not matches_to_draw:
         print("DEBUG draw_matches: No valid matches to draw.")
         # Create a placeholder image showing both inputs side-by-side
         h1, w1 = img1.shape[:2]
         h2, w2 = img2.shape[:2]
         vis_h = max(h1, h2)
         vis_w = w1 + w2
         vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
         vis[:h1, :w1, :] = img1
         vis[:h2, w1:w1 + w2, :] = img2
         cv2.putText(vis, "No Matches Found", (vis_w // 2 - 100, vis_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
         return vis


    try:
        img_matches = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, matches_to_draw, None,
            matchColor=(0, 255, 0), # Green for inliers/matches
            singlePointColor=(0, 0, 255), # Red for single points (optional)
            matchesMask=None, # We've already filtered matches
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # Optionally draw single points: cv2.DrawMatchesFlags_DEFAULT
        )
        return img_matches
    except cv2.error as e:
        print(f"ERROR during cv2.drawMatches: {e}")
        print(f"  img1 shape: {img1.shape if img1 is not None else 'None'}, kp1 count: {len(keypoints1)}")
        print(f"  img2 shape: {img2.shape if img2 is not None else 'None'}, kp2 count: {len(keypoints2)}")
        print(f"  Matches to draw count: {len(matches_to_draw)}")
        # Detailed check of match indices
        for i, m in enumerate(matches_to_draw):
            if m.queryIdx >= len(keypoints1) or m.trainIdx >= len(keypoints2):
                print(f"  -> Invalid match index at matches_to_draw[{i}]: queryIdx={m.queryIdx}, trainIdx={m.trainIdx}")
                # This case should be prevented by the checks above, but good to keep
        return None # Return None on error
    except Exception as e:
        print(f"UNEXPECTED ERROR during cv2.drawMatches: {e}")
        return None


def get_center_and_rotation(matrix, img_width, img_height):
    """
    Calculates the center coordinates and rotation angle of the detected object (image1)
    within the coordinate system of image2, based on the homography matrix.
    (Code for get_center_and_rotation remains the same - no changes needed here)
    """
    if matrix is None:
        return None, None, None

    h, w = img_height, img_width
    # Ensure corners are float32 for perspectiveTransform
    corners_img1 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    try:
        corners_img2 = cv2.perspectiveTransform(corners_img1, matrix)
        if corners_img2 is None: # Check if perspectiveTransform failed
             print("ERROR: cv2.perspectiveTransform returned None in get_center_and_rotation.")
             return None, None, None

        # Ensure corners_img2 is not empty and has the expected shape
        if corners_img2.shape != (4, 1, 2):
             print(f"ERROR: Unexpected shape from perspectiveTransform: {corners_img2.shape}")
             return None, None, None

        center_x = np.mean(corners_img2[:, 0, 0])
        center_y = np.mean(corners_img2[:, 0, 1])

        # Calculate angle based on the top edge (corner 1 - corner 0)
        dx = corners_img2[1, 0, 0] - corners_img2[0, 0, 0]
        dy = corners_img2[1, 0, 1] - corners_img2[0, 0, 1]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)

        return center_x, center_y, angle_degrees

    except cv2.error as e:
         print(f"OpenCV ERROR calculating center/rotation from homography: {e}")
         return None, None, None
    except Exception as e:
        print(f"ERROR calculating center/rotation from homography: {e}")
        return None, None, None

def draw_bounding_box(image, matrix, source_width, source_height, color=(0, 255, 0), thickness=2):
    """Draws the transformed bounding box onto the target image."""
    # (Code for draw_bounding_box remains the same - no changes needed here)
    if matrix is None:
        return image # Return original image if no matrix

    h, w = source_height, source_width
    # Ensure points are float32 for perspectiveTransform
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    try:
        dst_pts = cv2.perspectiveTransform(pts, matrix)
        if dst_pts is None: # Check if perspectiveTransform failed
             print("ERROR: cv2.perspectiveTransform returned None in draw_bounding_box.")
             return image # Return original image

        # Ensure dst_pts are valid for polylines
        dst_pts_int = np.int32(dst_pts)
        if dst_pts_int.shape != (4, 1, 2):
             print(f"ERROR: Unexpected shape for bounding box points: {dst_pts_int.shape}")
             return image # Return original image

        # Draw the polygon
        img_out = cv2.polylines(image.copy(), [dst_pts_int], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img_out
    except cv2.error as e:
         print(f"OpenCV ERROR drawing bounding box: {e}")
         return image # Return original image on error
    except Exception as e:
        print(f"ERROR drawing bounding box: {e}")
        return image # Return original image on error


# --- Main Localization Function ---
# Encapsulated the core logic into a function
def run_localization(lat, lon, aerial_photo_path):
    """Loads images, performs matching, calculates results, and shows visualization."""
    start_time_total = time.time()
    print("--- Initializing ---")

    # --- Load Map Info ---
    # Use the imported function get_map_info
    map_info = get_map_info(config.OFFLINE_MAP_FILE)
    if not map_info:
        print("ERROR: Failed to get map info. Exiting.")
        return # Use return instead of exit() if called as function
    print(f"Map CRS: {map_info['crs']}, Transform: {map_info['transform']}")

    # --- Load Aerial Image ---
    print(f"Loading aerial image from: {aerial_photo_path}")
    # Load directly using cv2.imread, remove ImageHandler dependency
    img_np = cv2.imread(aerial_photo_path)
    if img_np is None:
        print(f"ERROR: Failed to load aerial image at '{aerial_photo_path}'. Check path and file integrity. Exiting.")
        return
    img_h, img_w = img_np.shape[:2]
    print(f"Aerial image loaded. Shape: {img_np.shape}")

    # --- Get Map Chip ---
    print(f"\n--- Preparing Map Chip ---")
    print(f"Target center: Lat={lat:.6f}, Lon={lon:.6f}") # More precision
    # Define chip size locally (as set previously)
    chip_height_px = 500
    chip_width_px = 500
    # Use local variables in print statement:
    print(f"Getting map chip of size {chip_width_px}x{chip_height_px} pixels.")
    # Use local variables in function call:
    map_chip_np, map_chip_transform, map_chip_bounds = read_map_chip(
        config.OFFLINE_MAP_FILE, map_info, lat, lon, chip_height_px, chip_width_px # Pass local vars here
    )

    if map_chip_np is None:
        print("ERROR: Failed to get map chip. Exiting.")
        return
    chip_h, chip_w = map_chip_np.shape[:2]
    print(f"Map chip obtained. Shape: {map_chip_np.shape}")
    print(f"Map chip transform: {map_chip_transform}")
    # Use map_info['crs'] which should be a CRS object
    map_crs_str = str(map_info['crs']) # Get string representation for printing
    print(f"Map chip bounds (CRS: {map_crs_str}): {map_chip_bounds}")

    # --- Perform Feature Matching ---
    print("\n--- Starting Feature Matching (SuperGlue) ---")
    start_time_match = time.time()
    # Call the updated find_match function from feature_matcher
    M, inliers_mask_ransac, kp1_np, kp2_np, match_idx_kp1, match_idx_kp2 = find_match(
        img_np, map_chip_np, min_match_count=config.MIN_MATCH_COUNT
    )
    match_duration = time.time() - start_time_match
    print(f"Feature matching finished in {match_duration:.2f} seconds.")

    # --- Process Results ---
    img_with_box = map_chip_np.copy()
    calculated_lon_lat = None # Initialize variable

    if M is not None:
        print("\n--- Homography Found ---")
        print("Homography Matrix M (Aerial -> Map Chip):")
        print(M)

        center_x_chip, center_y_chip, angle_deg_chip = get_center_and_rotation(M, img_w, img_h)

        if center_x_chip is not None:
            print(f"Calculated Center (in map chip pixels): X={center_x_chip:.2f}, Y={center_y_chip:.2f}")
            print(f"Calculated Rotation (degrees): {angle_deg_chip:.2f}")

            # Convert chip center coordinates back to map CRS (using map_chip_transform)
            try:
                # Use the specific transform of the map chip we read
                # Note: rasterio.transform.xy takes row, col -> check if center_y, center_x is correct order
                # Often pixels are (x, y) but rasterio uses (row, col) ~ (y, x)
                # Assuming function expects (y, x) based on typical rasterio usage
                center_x_map, center_y_map = rasterio.transform.xy(map_chip_transform, center_y_chip, center_x_chip)
                # It might be the other way around:
                # center_x_map, center_y_map = rasterio.transform.xy(map_chip_transform, center_x_chip, center_y_chip)
                # You might need to verify this based on your map_chip_transform and expected CRS coordinates
                print(f"Calculated Center (map CRS - {map_crs_str}): X={center_x_map:.2f}, Y={center_y_map:.2f}")

                # Convert map CRS coordinates to Lon/Lat
                # Ensure map_info['crs'].to_epsg() is valid
                source_epsg_val = None
                try:
                    source_epsg_val = map_info['crs'].to_epsg()
                    if source_epsg_val is None:
                         raise ValueError("CRS.to_epsg() returned None")
                except Exception as crs_e:
                     print(f"ERROR getting EPSG code from map CRS: {crs_e}")
                     print(f"Map CRS object: {map_info['crs']}")
                     # Fallback or default if possible, otherwise conversion will fail
                     # Example fallback (replace with appropriate EPSG if needed):
                     # source_epsg_val = ITM_EPSG # Or another known CRS like 32636 for UTM 36N

                if source_epsg_val is not None:
                    center_lon_calc, center_lat_calc = transform_coords(
                        center_x_map, center_y_map, source_epsg=source_epsg_val, target_epsg=WGS84_EPSG
                    )
                    if center_lon_calc is not None:
                        calculated_lon_lat = (center_lon_calc, center_lat_calc)
                        # Moved the primary print to the end of the function
                        print(f"Successfully converted center to Lon/Lat.")
                    else:
                        print("ERROR: Failed to convert map center to Lon/Lat.")
                else:
                    print("ERROR: Cannot perform Lon/Lat conversion due to missing source EPSG.")


            except rasterio.errors.RasterioIOError as rio_e:
                 print(f"ERROR during rasterio coordinate transformation: {rio_e}")
            except ImportError:
                 print("ERROR: pyproj is likely needed for transform_coords. Install it (`pip install pyproj`)")
            except Exception as e:
                print(f"ERROR during coordinate conversion of calculated center: {e}")

            # Draw bounding box and center point IF coordinates were calculated
            img_with_box = draw_bounding_box(img_with_box, M, img_w, img_h, color=(0, 255, 255), thickness=2)
            # Draw circle only if center_x_chip and center_y_chip are valid numbers
            try:
                cv2.circle(img_with_box, (int(round(center_x_chip)), int(round(center_y_chip))), radius=5, color=(255, 0, 0), thickness=-1)
            except (ValueError, TypeError):
                 print("Warning: Could not draw center point on map chip.")
        else:
            print("Could not calculate center/rotation from homography.")
    else:
        print("\n--- Homography Not Found ---")
        print("Could not find a reliable transformation between the images.")

    # --- Visualization ---
    print("\n--- Generating Visualization ---")
    img_matches_vis = None
    # Ensure keypoints and indices are not None and are iterable/numpy arrays
    if kp1_np is not None and kp2_np is not None and match_idx_kp1 is not None and match_idx_kp2 is not None:
        # Check if we have any matches to draw
        if len(match_idx_kp1) > 0 and len(match_idx_kp2) > 0:
             # Check if kp arrays are valid (e.g., not empty if matches exist)
             if len(kp1_np) > 0 and len(kp2_np) > 0:
                 img_matches_vis = draw_matches(
                     img_np, kp1_np, map_chip_np, kp2_np,
                     match_idx_kp1, match_idx_kp2, inliers_mask_ransac,
                     max_matches_to_draw=50
                 )
                 if img_matches_vis is None:
                     print("Warning: draw_matches returned None, visualization might be missing or invalid.")
             else:
                  print("Cannot draw matches - keypoint arrays are empty.")
        else:
            print("No matches found between images, cannot draw matches visualization.")
    else:
        print("Cannot draw matches - keypoints or match indices missing or invalid.")

    # Save results before displaying
    # Ensure config.BASE_DIR is defined and valid
    output_dir = os.path.join(config.BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists
    aerial_fname = os.path.basename(aerial_photo_path)
    prefix = os.path.splitext(aerial_fname)[0]
    status = "SUCCESS" if M is not None and calculated_lon_lat is not None else "FAIL"
    # Format coordinates safely for filenames
    lat_str = f"{lat:.4f}".replace('.', 'p').replace('-', 'n')
    lon_str = f"{lon:.4f}".replace('.', 'p').replace('-', 'n')
    output_prefix_base = f"{prefix}__{status}__target_{lat_str}_{lon_str}"
    output_prefix = os.path.join(output_dir, output_prefix_base)


    if calculated_lon_lat:
         # Format calculated coords safely
        calc_lat_str = f"{calculated_lon_lat[1]:.4f}".replace('.', 'p').replace('-', 'n')
        calc_lon_str = f"{calculated_lon_lat[0]:.4f}".replace('.', 'p').replace('-', 'n')
        # Append calculated coords to the filename prefix
        output_prefix += f"__calc_{calc_lat_str}_{calc_lon_str}"


    try:
        save_path_box = f"{output_prefix}__map_chip_with_box.png"
        cv2.imwrite(save_path_box, img_with_box)
        print(f"Saved bounding box image to: {save_path_box}")
    except cv2.error as e:
         print(f"ERROR saving bounding box image: {e}")
    except Exception as e:
         print(f"Unexpected error saving bounding box image: {e}")

    if img_matches_vis is not None:
        try:
            save_path_matches = f"{output_prefix}__matches_vis.png"
            cv2.imwrite(save_path_matches, img_matches_vis)
            print(f"Saved matches visualization to: {save_path_matches}")
        except cv2.error as e:
             print(f"ERROR saving matches visualization: {e}")
        except Exception as e:
             print(f"Unexpected error saving matches visualization: {e}")


    # Display results
    print("\n--- Displaying Results (if possible) ---")
    display_scale = 0.5
    try:
        # Resize for display
        # Check if images are valid before resizing
        img_display = None
        if img_np is not None:
            img_display = cv2.resize(img_np, None, fx=display_scale, fy=display_scale)

        map_chip_display = None
        if map_chip_np is not None:
            map_chip_display = cv2.resize(map_chip_np, None, fx=display_scale, fy=display_scale)

        img_with_box_display = None
        if img_with_box is not None:
            img_with_box_display = cv2.resize(img_with_box, None, fx=display_scale, fy=display_scale)

        # Show windows only if images are valid
        if img_display is not None: cv2.imshow("Aerial Image (Input)", img_display)
        if map_chip_display is not None: cv2.imshow("Map Chip (Reference)", map_chip_display)
        if img_with_box_display is not None: cv2.imshow("Map Chip with Box / Center", img_with_box_display)

        if img_matches_vis is not None:
             # Resize matches visualization if it's too wide
            h_match, w_match = img_matches_vis.shape[:2]
            max_display_w = 1800 # Adjust as needed for your screen
            img_matches_vis_display = img_matches_vis
            if w_match > max_display_w:
                scale_match = max_display_w / w_match
                img_matches_vis_display = cv2.resize(img_matches_vis, None, fx=scale_match, fy=scale_match, interpolation=cv2.INTER_AREA)
            cv2.imshow("Feature Matches (SuperGlue Inliers)", img_matches_vis_display)

        # --- Final Output Section ---
        total_duration = time.time() - start_time_total
        print(f"\n--- Processing Complete ---")

        # *** MODIFICATION START: Display Calculated Coordinates ***
        print(f"\n--- Localization Result ---")
        if calculated_lon_lat is not None:
            final_lon, final_lat = calculated_lon_lat
            print(f"Successfully localized aerial image.")
            print(f"Calculated Center GPS Location (WGS84): ")
            print(f"  Latitude:  {final_lat:.6f}") # Standard 6 decimal places for GPS
            print(f"  Longitude: {final_lon:.6f}")
        else:
            print("Localization failed or coordinates could not be determined.")
            print("No updated GPS location available.")
        # *** MODIFICATION END ***

        print(f"\nTotal execution time: {total_duration:.2f} seconds.")

        # Only wait for key if windows were potentially shown
        if img_display is not None or map_chip_display is not None or img_with_box_display is not None or img_matches_vis is not None:
            print("\nPress 'q' or Esc key in any image window to close...")
            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27: # Quit on 'q' or Esc
                    break
            cv2.destroyAllWindows()
            print("Windows closed.")
        else:
            print("No images to display.")


    except cv2.error as e:
        print(f"ERROR during image display or resizing: {e}")
        print("Attempting to close any open windows.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"UNEXPECTED ERROR during display setup: {e}")
        print("Attempting to close any open windows.")
        cv2.destroyAllWindows()


# --- Main execution block ---
# Uses argparse for flexibility
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localize an aerial photo using an offline map.")
    parser.add_argument("--lat", type=float, default=None,
                        help="Optional: Latitude of the initial target center point (WGS84). If omitted, you will be prompted.")
    parser.add_argument("--lon", type=float, default=None,
                        help="Optional: Longitude of the initial target center point (WGS84). If omitted, you will be prompted.")
    parser.add_argument("aerial_photo", type=str,
                        help="Path to the input aerial photo.")
    args = parser.parse_args()

    # *** MODIFICATION START: Prompt for coordinates if not provided ***
    latitude_to_use = args.lat
    longitude_to_use = args.lon

    if latitude_to_use is None:
        while True:
            try:
                lat_str = input("Please enter the target Latitude (WGS84 decimal degrees): ")
                latitude_to_use = float(lat_str)
                # Basic validation for typical latitude range
                if -90 <= latitude_to_use <= 90:
                     break # Exit loop if conversion and range check are successful
                else:
                     print("Latitude must be between -90 and 90.")
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g., 32.1307)")
            except Exception as e:
                 print(f"An unexpected error occurred: {e}")


    if longitude_to_use is None:
        while True:
            try:
                lon_str = input("Please enter the target Longitude (WGS84 decimal degrees): ")
                longitude_to_use = float(lon_str)
                 # Basic validation for typical longitude range
                if -180 <= longitude_to_use <= 180:
                     break # Exit loop if conversion and range check are successful
                else:
                     print("Longitude must be between -180 and 180.")
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g., 34.8578)")
            except Exception as e:
                 print(f"An unexpected error occurred: {e}")

    # *** MODIFICATION END ***

    print(f"\n--- Running Localization ---")
    print(f"Initial Target Coordinates: Lat={latitude_to_use:.6f}, Lon={longitude_to_use:.6f}")
    print(f"Aerial Photo: {args.aerial_photo}")
    # Ensure OFFLINE_MAP_FILE is accessible via config
    try:
        print(f"Using Map: {config.OFFLINE_MAP_FILE}")
        # Optional: Check if map file exists early
        if not os.path.exists(config.OFFLINE_MAP_FILE):
             print(f"WARNING: Map file not found at {config.OFFLINE_MAP_FILE}")
             # Depending on requirements, you might want to exit here
             # import sys
             # sys.exit("Exiting due to missing map file.")
    except AttributeError:
         import sys
         print("ERROR: OFFLINE_MAP_FILE not found in config module. Please define it in config.py")
         sys.exit(1)
    except Exception as e:
         import sys
         print(f"Error accessing configuration: {e}")
         sys.exit(1)


    # Ensure aerial photo path is valid before proceeding
    if not os.path.exists(args.aerial_photo):
        import sys
        print(f"ERROR: Aerial photo not found at '{args.aerial_photo}'. Please check the path.")
        sys.exit(1)


    # --- Run Main Function ---
    run_localization(latitude_to_use, longitude_to_use, args.aerial_photo)

    print("\n--- Script Finished ---")