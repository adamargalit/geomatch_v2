import torch
import cv2
import numpy as np
import config # To get weights paths
import time
# --- Import model definitions and preprocessing ---
# Assumes model definitions are in src/models/ and utils has the preprocess function
try:
    from .models.superpoint import SuperPoint
    from .models.superglue import SuperGlue
    from src.utils import preprocess_image_sg # Use the preprocess func from our utils
    print("DEBUG: Successfully imported SuperPoint, SuperGlue, and preprocess_image_sg.")
except ImportError as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Ensure SuperPoint/SuperGlue model files are in 'src/models/'")
    print("and 'preprocess_image_sg' is defined in 'src/utils.py'.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise

# --- Configuration ---
# Resize images to suitable dimension for SP/SG (e.g., 640 or 1024)
# -1 means no resize, but SP/SG often expect images around VGA resolution
RESIZE_MAX_DIM = 1024 # Or 640, needs testing

# SuperPoint config (matching SuperGlue repo defaults)
superpoint_config = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024 # Limit keypoints passed to SuperGlue
}
# SuperGlue config
superglue_config = {
    # 'descriptor_dim': 256, # Usually defined within model
    # 'weights': 'outdoor', # Not needed if loading externally
    # 'keypoint_encoder': [32, 64, 128, 256], # Usually defined within model
    # 'GNN_layers': ['self', 'cross'] * 9, # Usually defined within model
    'sinkhorn_iterations': 20, # From SuperGlue repo default
    'match_threshold': 0.2,
}

# --- Device Selection ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"INFO: Feature Matcher - Running SuperPoint/SuperGlue on device: {device}")

# --- Load Models ---
# Load model architecture
print("INFO: Feature Matcher - Instantiating SuperPoint...")
superpoint = SuperPoint(superpoint_config).eval().to(device)
print("INFO: Feature Matcher - Instantiating SuperGlue...")
# Pass only config needed by modified SuperGlue __init__
superglue = SuperGlue(superglue_config).eval().to(device)

# Load weights externally using paths from config.py
try:
    print(f"INFO: Feature Matcher - Loading SuperPoint weights from: {config.SUPERPOINT_WEIGHTS_PATH}")
    superpoint.load_state_dict(torch.load(config.SUPERPOINT_WEIGHTS_PATH, map_location=device))
    print(f"INFO: Feature Matcher - Loading SuperGlue weights from: {config.SUPERGLUE_WEIGHTS_PATH}")
    # SuperGlue weights might be nested (check .pth file structure if error)
    # Try loading directly first
    superglue.load_state_dict(torch.load(config.SUPERGLUE_WEIGHTS_PATH, map_location=device))
    print("INFO: Feature Matcher - SuperPoint and SuperGlue weights loaded successfully.")
except FileNotFoundError as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: Weight file not found: {e}")
    print("Ensure weight files exist at paths specified in config.py and")
    print("that the filenames in config.py match the actual files.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR loading model weights: {e}")
    print("Check if weight file is corrupt or if model definition mismatches.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise

@torch.no_grad() # Ensure no gradients are computed during inference
def run_superglue(img0_bgr, img1_bgr):
    """
    Runs SuperPoint and SuperGlue on two BGR images.

    Args:
        img0_bgr (np.ndarray): First image (aerial), BGR format.
        img1_bgr (np.ndarray): Second image (map chip), BGR or RGBA format.

    Returns:
        tuple: (matches, kp0, kp1, desc0, desc1, scores0, scores1, confidence) or None tuple if error.
    """
    # device variable is defined globally in the file
    try:
        # Preprocess images -> Grayscale tensors
        tensor0 = preprocess_image_sg(img0_bgr, resize_max_dim=RESIZE_MAX_DIM, device=device)
        tensor1 = preprocess_image_sg(img1_bgr, resize_max_dim=RESIZE_MAX_DIM, device=device)
        print(f"DEBUG: Preprocessed image tensors to shapes: {tensor0.shape}, {tensor1.shape}")

        # SuperPoint Inference - Use global superpoint model
        pred0 = superpoint({'image': tensor0})
        pred1 = superpoint({'image': tensor1})

        # *** Data Preparation for SuperGlue ***
        # Add batch dimension using unsqueeze(0)
        kp0_tensor = pred0['keypoints'][0].to(device).float().unsqueeze(0)    # Shape: [1, N, 2]
        scores0_tensor = pred0['scores'][0].to(device).float().unsqueeze(0)    # Shape: [1, N]
        desc0_tensor = pred0['descriptors'][0].to(device).float().unsqueeze(0) # Shape: [1, D, N]
        kp1_tensor = pred1['keypoints'][0].to(device).float().unsqueeze(0)    # Shape: [1, M, 2]
        scores1_tensor = pred1['scores'][0].to(device).float().unsqueeze(0)    # Shape: [1, M]
        desc1_tensor = pred1['descriptors'][0].to(device).float().unsqueeze(0) # Shape: [1, D, M]

        data = {
            'image0': tensor0, # Shape: [1, 1, H1, W1]
            'image1': tensor1, # Shape: [1, 1, H2, W2]
            'keypoints0': kp0_tensor,
            'scores0': scores0_tensor,
            'descriptors0': desc0_tensor,
            'keypoints1': kp1_tensor,
            'scores1': scores1_tensor,
            'descriptors1': desc1_tensor
        }
        # *** END Data Preparation ***

        # SuperGlue Inference - Use global superglue model
        pred_sg = superglue(data) # Pass data with batch dimension

        # Detach tensors and move to CPU for NumPy conversion
        # Remove batch dimension [0] after getting results
        kp0 = kp0_tensor[0].cpu().numpy()
        kp1 = kp1_tensor[0].cpu().numpy()
        desc0 = desc0_tensor[0].cpu().numpy().T # D x N -> N x D
        desc1 = desc1_tensor[0].cpu().numpy().T # D x M -> M x D
        scores0 = scores0_tensor[0].cpu().numpy()
        scores1 = scores1_tensor[0].cpu().numpy()
        matches = pred_sg['matches0'][0].cpu().numpy() # Indices of kp1 that match kp0 (-1 if no match)
        match_confidence = pred_sg['matching_scores0'][0].cpu().numpy() # Confidence scores for matches

        # kp0 and kp1 are (N, 2) and (M, 2) arrays of coordinates
        # matches is (N,) array where matches[i] is index in kp1 matching kp0[i], or -1

        return matches, kp0, kp1, desc0, desc1, scores0, scores1, match_confidence

    except Exception as e:
        print(f"ERROR during SuperPoint/SuperGlue inference: {e}")
        import traceback
        traceback.print_exc()
        # Return None for all expected outputs on error
        return None, None, None, None, None, None, None, None
# <<< End of updated run_superglue function >>>
# <<< PASTE THIS UPDATED find_match function into src/feature_matcher.py >>>
# Make sure import time, import config, import numpy as np, import cv2 are present above this function

def find_match(img1_bgr, img2_bgr, min_match_count=10): # Accepts min_match_count
    """
    Finds matches between two images using SuperGlue and RANSAC. STANDALONE version.
    Uses parameters defined in config.py (RANSAC_REPROJ_THRESHOLD, RANSAC_CONFIDENCE).

    Args:
        img1_bgr (np.ndarray): Aerial image (BGR).
        img2_bgr (np.ndarray): Map chip image (BGR or RGBA).
        min_match_count (int): Minimum number of *inliers* required for homography.

    Returns:
        tuple: (M, mask, kp1_np, kp2_np, match_idx_kp1, match_idx_kp2) if successful,
               (None, None, None, None, None, None) otherwise.
        - M: Homography matrix (3x3).
        - mask: Inlier mask from RANSAC (relative to matched_kp1/kp2).
        - kp1_np, kp2_np: Original keypoints detected by SuperPoint.
        - match_idx_kp1, match_idx_kp2: Indices of the *inlier* keypoints in kp1_np and kp2_np.
    """
    start_time = time.time()
    print(f"\n--- Starting Feature Matching (SuperGlue) ---")

    # Call standalone run_superglue
    matches_raw, kp1_np, kp2_np, desc1_np, desc2_np, scores1_np, scores2_np, confidence = run_superglue(img1_bgr, img2_bgr)

    # Check if SuperGlue failed
    if matches_raw is None and kp1_np is None:
        print("ERROR: SuperGlue matching step failed.")
        return None, None, None, None, None, None

    print(f"SuperPoint detected {len(kp1_np)} keypoints in aerial, {len(kp2_np)} in map chip.")

    # Filter matches based on valid indices from SuperGlue
    valid_match_indices_kp1 = np.where(matches_raw > -1)[0]
    valid_match_indices_kp2 = matches_raw[valid_match_indices_kp1]
    matched_kp1 = kp1_np[valid_match_indices_kp1]
    matched_kp2 = kp2_np[valid_match_indices_kp2]
    num_raw_matches = len(matched_kp1)
    print(f"SuperGlue found {num_raw_matches} raw matches.")

    # Check if enough points for RANSAC (needs >= 4)
    MIN_PTS_FOR_HOMOGRAPHY = 4
    if num_raw_matches < MIN_PTS_FOR_HOMOGRAPHY:
        print(f"WARN: Only {num_raw_matches} raw matches found, less than minimum required ({MIN_PTS_FOR_HOMOGRAPHY}) for RANSAC. Skipping.")
        return None, None, kp1_np, kp2_np, None, None

    # --- RANSAC for Homography ---
    print("Running RANSAC to find Homography...")
    M = None
    mask = None

    # Ensure config values are imported and accessible
    try:
         # *** Use config values for RANSAC parameters ***
         # Make sure RANSAC_REPROJ_THRESHOLD and RANSAC_CONFIDENCE are defined in config.py
         M, mask = cv2.findHomography(matched_kp1,
                                      matched_kp2,
                                      cv2.RANSAC,
                                      config.RANSAC_REPROJ_THRESHOLD, # Use config value
                                      confidence=config.RANSAC_CONFIDENCE) # Use config value
    except cv2.error as e:
         print(f"ERROR during cv2.findHomography: {e}")
    except AttributeError as e:
         print(f"ERROR: Missing RANSAC setting in config.py? Details: {e}")
         print("Using default RANSAC parameters (5.0, 0.999) as fallback.")
         # Fallback to defaults if config attributes are missing
         M, mask = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 5.0, confidence=0.999)
    except Exception as e:
         print(f"UNEXPECTED ERROR during cv2.findHomography: {e}")


    if M is None or mask is None:
        print("WARN: RANSAC failed to find a valid homography.")
        return None, None, kp1_np, kp2_np, None, None

    num_inliers = np.sum(mask)
    inlier_ratio = num_inliers / num_raw_matches if num_raw_matches > 0 else 0
    print(f"RANSAC found {num_inliers} inliers ({inlier_ratio:.2%}) using threshold {config.RANSAC_REPROJ_THRESHOLD if hasattr(config, 'RANSAC_REPROJ_THRESHOLD') else 'DEFAULT(5.0)'}")

    # *** Use passed min_match_count for final check ***
    if num_inliers < min_match_count: # Use argument passed from main.py
        print(f"WARN: Only {num_inliers} inliers found, less than minimum required ({min_match_count}).")
        return None, None, kp1_np, kp2_np, None, None

    # Get inlier indices relative to original keypoint lists
    # mask is an array of arrays (e.g., [[1], [0], [1]]), use ravel() or flatten()
    inlier_indices_in_matched_list = np.where(mask.ravel() == 1)[0]
    final_match_idx_kp1 = valid_match_indices_kp1[inlier_indices_in_matched_list]
    final_match_idx_kp2 = valid_match_indices_kp2[inlier_indices_in_matched_list]

    end_time = time.time()
    print(f"Feature matching and RANSAC completed in {end_time - start_time:.2f} seconds.")

    # Return results
    return M, mask.ravel(), kp1_np, kp2_np, final_match_idx_kp1, final_match_idx_kp2


# --- (Optional Self-Test Block - Needs updating for SuperGlue) ---
# if __name__ == "__main__":
#    # Load test images, call find_match, print results
#    pass