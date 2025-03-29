# main.py (With Configurable Monitoring)

import argparse
import os
import time
import sys
import threading # For monitoring thread (Class defined but potentially not used)
import cv2         # For displaying images

# --- Configuration Flag for Monitoring ---
ENABLE_MONITORING = False # Set to True to enable, False to disable
# -----------------------------------------

# --- Conditional Monitoring Imports and Initialization ---
GPU_MONITORING_AVAILABLE = False # Default status
PSUTIL_AVAILABLE = False       # Default status
pynvml = None                  # Define pynvml in this scope, default to None
psutil = None                  # Define psutil in this scope, default to None

if ENABLE_MONITORING:
    print("Attempting to enable resource monitoring...")
    # Try to import and initialize psutil for CPU/RAM
    try:
        import psutil
        PSUTIL_AVAILABLE = True
        print("psutil imported successfully (CPU/RAM monitoring possible).")
    except ImportError:
        print("Warning: psutil not found. CPU/RAM monitoring disabled.")
        PSUTIL_AVAILABLE = False # Explicitly set False on failure

    # Try to import and initialize pynvml for GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_MONITORING_AVAILABLE = True # Set flag only if init succeeded
        print("pynvml imported and initialized successfully (GPU monitoring possible).")
    except ImportError:
        print("pynvml library not found. GPU monitoring disabled.")
        GPU_MONITORING_AVAILABLE = False
    except pynvml.NVMLError as nvml_error:
        print(f"pynvml initialization failed (NVML Error: {nvml_error}). GPU monitoring disabled.")
        GPU_MONITORING_AVAILABLE = False # Ensure False on init failure
    except Exception as e:
        print(f"An unexpected error occurred during pynvml setup: {e}. GPU monitoring disabled.")
        GPU_MONITORING_AVAILABLE = False # Ensure False on other errors
else:
    print("Resource monitoring disabled by configuration flag (ENABLE_MONITORING=False).")
# --- End Conditional Monitoring Setup ---


# Assuming config.py is in the same directory or Python path
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it exists in the project root.")
    # Check if NVML was successfully initialized before trying to shut down
    if GPU_MONITORING_AVAILABLE and pynvml:
        try: pynvml.nvmlShutdown()
        except Exception: pass # Ignore shutdown errors here
    sys.exit(1)
except Exception as e:
    print(f"Error loading config: {e}")
    if GPU_MONITORING_AVAILABLE and pynvml:
        try: pynvml.nvmlShutdown()
        except Exception: pass
    sys.exit(1)


# Import the visual localizer function from src/visual_localizer.py
try:
    # Import status codes if needed for more detailed handling later
    from src.visual_localizer import run_localization, RESULT_SUCCESS
except ModuleNotFoundError:
     print("ERROR: Could not import from src.visual_localizer. Ensure src/__init__.py exists and src is in the Python path.")
     if GPU_MONITORING_AVAILABLE and pynvml:
         try: pynvml.nvmlShutdown()
         except Exception: pass
     sys.exit(1)
except ImportError as e:
     print(f"ERROR: Importing from src.visual_localizer failed: {e}")
     if GPU_MONITORING_AVAILABLE and pynvml:
         try: pynvml.nvmlShutdown()
         except Exception: pass
     sys.exit(1)

# ==================================================
# <<< Resource Monitoring Code Section (Uncommented) >>>
# ==================================================
# Class definition remains uncommented
class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_mem_usage = []
        self.gpu_handle = None

        # Try to get GPU handle only if GPU monitoring was successfully initialized
        if GPU_MONITORING_AVAILABLE and pynvml:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print(f"Monitoring GPU: {pynvml.nvmlDeviceGetName(self.gpu_handle)}")
            except Exception as e:
                print(f"Could not get GPU handle: {e}")
                self.gpu_handle = None # Disable GPU part if handle fails

    def _monitor(self):
        """Monitoring loop runs in a separate thread."""
        while not self._stop_event.is_set():
            # CPU Usage - Check if psutil is available
            if PSUTIL_AVAILABLE and psutil:
                try:
                    self.cpu_usage.append(psutil.cpu_percent(interval=None))
                except Exception as e:
                    # print(f"\nError getting CPU usage: {e}", end="") # Less verbose
                    self.cpu_usage.append(None)
            else:
                 self.cpu_usage.append(None) # Append None if psutil not available

            # RAM Usage - Check if psutil is available
            if PSUTIL_AVAILABLE and psutil:
                try:
                    self.ram_usage.append(psutil.virtual_memory().percent)
                except Exception as e:
                    # print(f"\nError getting RAM usage: {e}", end="") # Less verbose
                    self.ram_usage.append(None)
            else:
                self.ram_usage.append(None) # Append None if psutil not available

            # GPU Usage - Check if GPU handle is valid (implies GPU monitoring is available)
            if self.gpu_handle and pynvml:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_usage.append(util.gpu)
                    self.gpu_mem_usage.append(mem_info.used / mem_info.total * 100 if mem_info.total > 0 else 0)
                except pynvml.NVMLError: # Catch specific NVML error silently during loop
                    self.gpu_usage.append(None)
                    self.gpu_mem_usage.append(None)
                except Exception: # Catch any other unexpected errors silently
                    self.gpu_usage.append(None)
                    self.gpu_mem_usage.append(None)
            else:
                 # Append None if GPU monitoring not available/handle invalid
                 self.gpu_usage.append(None)
                 self.gpu_mem_usage.append(None)

            # Wait for the next interval
            wait_start = time.monotonic()
            while time.monotonic() < wait_start + self.interval:
                 if self._stop_event.is_set(): break
                 time.sleep(0.01)

    def start(self):
        """Starts the monitoring thread."""
        if self._thread is not None and self._thread.is_alive(): return # Already running
        # Reset stats lists
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_mem_usage = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        print("Resource monitor thread started.")

    def stop(self):
        """Stops the monitoring thread and returns results."""
        if self._thread is None or not self._thread.is_alive():
            # Return empty/zeroed stats if never started or already stopped
            return None # Return None to indicate no stats were collected

        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2)
        if self._thread.is_alive(): print("\nWarning: Resource monitor thread did not stop cleanly.")
        self._thread = None
        print("\nResource monitor thread stopped.")

        # Filter out None values before calculating stats
        valid_cpu = [x for x in self.cpu_usage if x is not None]
        valid_ram = [x for x in self.ram_usage if x is not None]
        valid_gpu = [x for x in self.gpu_usage if x is not None]
        valid_gpu_mem = [x for x in self.gpu_mem_usage if x is not None]

        # Return None if no valid data points were collected for primary metrics
        if not valid_cpu and not valid_ram:
            return None

        results = {
            "avg_cpu": sum(valid_cpu) / len(valid_cpu) if valid_cpu else 0,
            "max_cpu": max(valid_cpu) if valid_cpu else 0,
            "avg_ram": sum(valid_ram) / len(valid_ram) if valid_ram else 0,
            "max_ram": max(valid_ram) if valid_ram else 0,
            # Add GPU stats only if they were collected
            "avg_gpu_util": sum(valid_gpu) / len(valid_gpu) if valid_gpu else 0,
            "max_gpu_util": max(valid_gpu) if valid_gpu else 0,
            "avg_gpu_mem": sum(valid_gpu_mem) / len(valid_gpu_mem) if valid_gpu_mem else 0,
            "max_gpu_mem": max(valid_gpu_mem) if valid_gpu_mem else 0,
        }
        return results
# ==================================================
# <<< END: Resource Monitoring Code Section >>>
# ==================================================


# --- Main execution block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Localize an aerial photo using an offline map (Visual Localization Test).")
    parser.add_argument("--lat", type=float, default=None, help="Optional: Latitude...")
    parser.add_argument("--lon", type=float, default=None, help="Optional: Longitude...")
    parser.add_argument("--no-save", action="store_true", help="Disable saving output images and coordinate file.")
    parser.add_argument("aerial_photo", type=str, help="Path to the input aerial photo.")
    args = parser.parse_args()

    # --- Get Coordinates ---
    latitude_to_use = args.lat
    longitude_to_use = args.lon
    # (Coordinate prompting logic remains the same)
    if latitude_to_use is None:
        while True:
            try:
                lat_str = input("Please enter the target Latitude (WGS84 decimal degrees): ")
                latitude_to_use = float(lat_str)
                if -90 <= latitude_to_use <= 90: break
                else: print("Latitude must be between -90 and 90.")
            except ValueError: print("Invalid input...")
            except Exception as e: print(f"An unexpected error occurred: {e}")
    if longitude_to_use is None:
        while True:
            try:
                lon_str = input("Please enter the target Longitude (WGS84 decimal degrees): ")
                longitude_to_use = float(lon_str)
                if -180 <= longitude_to_use <= 180: break
                else: print("Longitude must be between -180 and 180.")
            except ValueError: print("Invalid input...")
            except Exception as e: print(f"An unexpected error occurred: {e}")

    # --- Initial Setup Prints ---
    print(f"\n--- Running Visual Localization Test ---")
    print(f"Initial Target Coordinates: Lat={latitude_to_use:.6f}, Lon={longitude_to_use:.6f}")
    print(f"Aerial Photo: {args. aerial_photo}")
    # (Check config and file existence)
    try:
        print(f"Using Map: {config.OFFLINE_MAP_FILE}")
        if not os.path.exists(config.OFFLINE_MAP_FILE): print(f"WARNING: Map file not found...")
    except AttributeError: print("ERROR: OFFLINE_MAP_FILE not found..."); sys.exit(1)
    if not os.path.exists(args.aerial_photo): print(f"ERROR: Aerial photo not found..."); sys.exit(1)


    # <<< --- Conditional START MONITORING --- >>>
    monitor = None  # Ensure 'monitor' exists, default to None
    if ENABLE_MONITORING:
        # Only create and start if monitoring is globally enabled AND
        # at least one monitoring library (psutil or pynvml) was available
        if PSUTIL_AVAILABLE or GPU_MONITORING_AVAILABLE:
            print("-" * 30)
            monitor = ResourceMonitor(interval=0.2)
            monitor.start()
            time.sleep(0.1) # Give monitor a tiny moment to start
            print("-" * 30)
        else:
            print("Monitoring enabled but required libraries (psutil/pynvml) not available/initialized.")

    # --- Run Visual Localization ---
    print("\nCalling run_localization...")
    start_run_time = time.time()
    status, result_data = run_localization(
        latitude_to_use,
        longitude_to_use,
        args.aerial_photo,
        save_output=not args.no_save
    )
    end_run_time = time.time()
    print(f"run_localization finished in {end_run_time - start_run_time:.2f} seconds.")

    # <<< --- Conditional STOP MONITORING --- >>>
    usage_stats = None # Ensure 'usage_stats' exists, default to None
    if monitor: # Check if monitor object was created and started
        print("-" * 30)
        usage_stats = monitor.stop() # Stop will return None if no data collected
        print("-" * 30)

    # --- Process Result (Including Image Display) ---
    # (This part remains largely the same)
     # --- Process Result (Including Image Display) ---
    print(f"\n--- Result ---")
    print(f"Status: {status}")

    display_windows_opened = False # Flag to track if we need waitKey

    # Print details if successful
    if status == RESULT_SUCCESS and result_data:
        print(f"Calculated Latitude:  {result_data.get('lat', 'N/A'):.6f}")
        print(f"Calculated Longitude: {result_data.get('lon', 'N/A'):.6f}")
        print(f"Calculated Angle:     {result_data.get('angle', 'N/A'):.2f}")
        if not args.no_save: print(f"\nOutput images/coordinates potentially saved.")
        else: print(f"\nOutput saving disabled via --no-save.")
    # Print error details if available on failure
    elif result_data and 'error' in result_data:
         print(f"Localization Error details: {result_data['error']}")
    elif status != RESULT_SUCCESS:
         print("Localization failed with an unknown error or no details provided.")

    # --- Attempt to Display Images (Success or Failure) ---
    print("\nAttempting to display images...")
    display_scale = 0.7 # Adjust scale as needed

    if result_data: # Check if result_data dictionary exists at all
        img_box = result_data.get('img_with_box')         # Available on success
        img_matches = result_data.get('img_matches_vis')  # Available on success
        # Get base images (assuming visual_localizer was modified to return them on failure)
        img_clean_aerial = result_data.get('img_clean')
        img_clean_map_chip = result_data.get('map_chip_clean')

        if status == RESULT_SUCCESS:
            # Display results on Success
            if img_box is not None:
                try:
                    img_box_display = cv2.resize(img_box, None, fx=display_scale, fy=display_scale)
                    cv2.imshow("Map Chip with Box / Center (SUCCESS)", img_box_display)
                    display_windows_opened = True
                except Exception as e: print(f"Error displaying success box image: {e}")

            if img_matches is not None:
                try:
                    # Add resize logic for matches image if needed (copied from previous version)
                    h_match, w_match = img_matches.shape[:2]
                    max_display_w = 1800
                    img_matches_display = img_matches
                    if w_match > max_display_w:
                        scale_match = max_display_w / w_match
                        img_matches_display = cv2.resize(img_matches, None, fx=scale_match, fy=scale_match, interpolation=cv2.INTER_AREA)
                    # --- end resize ---
                    cv2.imshow("Feature Matches (SUCCESS)", img_matches_display)
                    display_windows_opened = True
                except Exception as e: print(f"Error displaying success matches image: {e}")

        else:
            # Display inputs on Failure (if available)
            print(f"Localization Status: {status}. Displaying input images if available.")
            if img_clean_aerial is not None:
                try:
                    # Resize aerial input for display if needed
                    h_a, w_a = img_clean_aerial.shape[:2]
                    scale_a = display_scale
                    # Optional: Limit max size
                    # max_h, max_w = 800, 1000
                    # if h_a * scale_a > max_h or w_a * scale_a > max_w:
                    #     scale_a = min(max_h / h_a, max_w / w_a)

                    img_aerial_display = cv2.resize(img_clean_aerial, None, fx=scale_a, fy=scale_a)
                    cv2.imshow(f"Aerial Input ({status})", img_aerial_display)
                    display_windows_opened = True
                except Exception as e: print(f"Error displaying aerial input on failure: {e}")

            if img_clean_map_chip is not None:
                try:
                    # Handle potential 4 channels (RGBA/BGRA) in map chip before display/resize
                    img_map_chip_todisplay = img_clean_map_chip
                    if img_map_chip_todisplay.shape[2] == 4:
                         # Assuming BGRA, convert to BGR for display
                         img_map_chip_todisplay = cv2.cvtColor(img_map_chip_todisplay, cv2.COLOR_BGRA2BGR)

                    # Resize map chip input for display if needed
                    h_m, w_m = img_map_chip_todisplay.shape[:2]
                    scale_m = display_scale
                    # Optional: Limit max size
                    # max_h, max_w = 800, 1000
                    # if h_m * scale_m > max_h or w_m * scale_m > max_w:
                    #     scale_m = min(max_h / h_m, max_w / w_m)

                    img_map_chip_display = cv2.resize(img_map_chip_todisplay, None, fx=scale_m, fy=scale_m)
                    cv2.imshow(f"Map Chip ({status})", img_map_chip_display)
                    display_windows_opened = True
                except Exception as e: print(f"Error displaying map chip on failure: {e}")

    else: # result_data itself is None
         print("No result data available to display.")


    # --- Display Monitoring Results (if ENABLE_MONITORING was True and stats collected) ---
    # This block remains uncommented, relies on usage_stats being None if not run
    if 'usage_stats' in locals() and usage_stats: # Check if variable exists and is not None
        print("\n--- Resource Usage During run_localization ---")
        # ... (rest of monitoring print logic) ...
    elif 'ENABLE_MONITORING' in globals() and ENABLE_MONITORING: # Check if monitoring flag exists
         print("\nMonitoring enabled but failed to collect sufficient data or monitor didn't run.")


    # --- Wait for Key Press if any Windows Were Opened ---
    if display_windows_opened:
        print("\nPress any key in an image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Image windows closed.")

    # --- Conditional Shutdown GPU monitoring ---
    # Check uses the flag that indicates successful initialization
    if 'GPU_MONITORING_AVAILABLE' in globals() and GPU_MONITORING_AVAILABLE and 'pynvml' in globals() and pynvml:
        try:
            pynvml.nvmlShutdown()
            print("NVML Shutdown.")
        except Exception as e:
             print(f"Error during nvmlShutdown: {e}")

    print("\n--- Script Finished ---")