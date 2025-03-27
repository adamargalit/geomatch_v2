# main.py (With Resource Monitoring AND Image Display)

import argparse
import os
import time
import sys
import threading # For monitoring thread
import psutil    # For CPU/RAM monitoring
import cv2       # For displaying images

# --- Optional GPU Monitoring ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
    print("NVIDIA GPU monitoring enabled.")
except Exception as e:
    GPU_MONITORING_AVAILABLE = False
    print(f"NVIDIA GPU monitoring unavailable (pynvml error: {e}). Will only monitor CPU/RAM.")
# --- End Optional GPU Monitoring ---


# Assuming config.py is in the same directory or Python path
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it exists in the project root.")
    if GPU_MONITORING_AVAILABLE: pynvml.nvmlShutdown()
    sys.exit(1)
except Exception as e:
    print(f"Error loading config: {e}")
    if GPU_MONITORING_AVAILABLE: pynvml.nvmlShutdown()
    sys.exit(1)


# Import the visual localizer function from src/visual_localizer.py
try:
    # Import status codes if needed for more detailed handling later
    from src.visual_localizer import run_localization, RESULT_SUCCESS
except ModuleNotFoundError:
     print("ERROR: Could not import from src.visual_localizer. Ensure src/__init__.py exists and src is in the Python path.")
     if GPU_MONITORING_AVAILABLE: pynvml.nvmlShutdown()
     sys.exit(1)
except ImportError as e:
     print(f"ERROR: Importing from src.visual_localizer failed: {e}")
     if GPU_MONITORING_AVAILABLE: pynvml.nvmlShutdown()
     sys.exit(1)

# ==================================================
# <<< START: Resource Monitoring Code Section >>>
# ==================================================
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

        if GPU_MONITORING_AVAILABLE:
            try:
                # Assuming single GPU system or monitoring the first GPU
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print(f"Monitoring GPU: {pynvml.nvmlDeviceGetName(self.gpu_handle)}")
            except Exception as e:
                print(f"Could not get GPU handle: {e}")
                self.gpu_handle = None # Disable GPU part if handle fails

    def _monitor(self):
        """Monitoring loop runs in a separate thread."""
        while not self._stop_event.is_set():
            # CPU Usage (System Wide)
            try:
                self.cpu_usage.append(psutil.cpu_percent(interval=None))
            except Exception as e:
                print(f"\nError getting CPU usage: {e}", end="")
                self.cpu_usage.append(None)

            # RAM Usage (System Wide)
            try:
                self.ram_usage.append(psutil.virtual_memory().percent)
            except Exception as e:
                print(f"\nError getting RAM usage: {e}", end="")
                self.ram_usage.append(None)

            # GPU Usage (if available and handle is valid)
            if self.gpu_handle:
                try:
                    # Try querying GPU status
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_usage.append(util.gpu)
                    self.gpu_mem_usage.append(mem_info.used / mem_info.total * 100 if mem_info.total > 0 else 0)
                # --- Catch specific NVML error ---
                except pynvml.NVMLError as nvml_error:
                    print(f"\nNVML Error during GPU query: {nvml_error}", end="")
                    self.gpu_usage.append(None) # Append None to keep lists aligned
                    self.gpu_mem_usage.append(None)
                except Exception as e:
                    # Catch any other unexpected errors
                    print(f"\nUnexpected Error during GPU query: {e}", end="")
                    self.gpu_usage.append(None)
                    self.gpu_mem_usage.append(None)
                # --- End error catching ---

            # Wait for the next interval, checking stop event periodically
            # Use a shorter wait within a loop to check stop_event more often
            wait_start = time.monotonic()
            while time.monotonic() < wait_start + self.interval:
                 if self._stop_event.is_set():
                     break
                 time.sleep(0.01) # Sleep briefly to avoid busy-waiting

    def start(self):
        """Starts the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            print("Monitor already running.")
            return
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_mem_usage = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        print("Resource monitor started.")

    def stop(self):
        """Stops the monitoring thread and returns results."""
        if self._thread is None or not self._thread.is_alive():
            print("Monitor not running or already stopped.")
            # Return empty/zeroed stats if never started or already stopped
            return {
                "avg_cpu": 0, "max_cpu": 0, "avg_ram": 0, "max_ram": 0,
                "avg_gpu_util": 0, "max_gpu_util": 0, "avg_gpu_mem": 0, "max_gpu_mem": 0
            }


        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2) # Wait for thread to finish with timeout
        if self._thread.is_alive():
             print("\nWarning: Resource monitor thread did not stop cleanly.")
        self._thread = None
        print("\nResource monitor stopped.")

        # Filter out None values (e.g., if query failed) before calculating stats
        valid_cpu = [x for x in self.cpu_usage if x is not None]
        valid_ram = [x for x in self.ram_usage if x is not None]
        valid_gpu = [x for x in self.gpu_usage if x is not None]
        valid_gpu_mem = [x for x in self.gpu_mem_usage if x is not None]

        results = {
            "avg_cpu": sum(valid_cpu) / len(valid_cpu) if valid_cpu else 0,
            "max_cpu": max(valid_cpu) if valid_cpu else 0,
            "avg_ram": sum(valid_ram) / len(valid_ram) if valid_ram else 0,
            "max_ram": max(valid_ram) if valid_ram else 0,
        }
        # Only add GPU stats if monitoring was available AND successful queries were made
        if GPU_MONITORING_AVAILABLE and self.gpu_handle:
             results.update({
                "avg_gpu_util": sum(valid_gpu) / len(valid_gpu) if valid_gpu else 0,
                "max_gpu_util": max(valid_gpu) if valid_gpu else 0,
                "avg_gpu_mem": sum(valid_gpu_mem) / len(valid_gpu_mem) if valid_gpu_mem else 0,
                "max_gpu_mem": max(valid_gpu_mem) if valid_gpu_mem else 0,
             })
        # Add placeholders if GPU wasn't available/working, for consistent structure
        else:
             results.update({
                "avg_gpu_util": 0, "max_gpu_util": 0, "avg_gpu_mem": 0, "max_gpu_mem": 0
             })

        return results
# ==================================================
# <<< END: Resource Monitoring Code Section >>>
# ==================================================


# --- Main execution block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Localize an aerial photo using an offline map (Visual Localization Test).")
    parser.add_argument("--lat", type=float, default=None, help="Optional: Latitude...") # Shortened help text
    parser.add_argument("--lon", type=float, default=None, help="Optional: Longitude...")
    parser.add_argument("--no-save", action="store_true", help="Disable saving output images and coordinate file.")
    parser.add_argument("aerial_photo", type=str, help="Path to the input aerial photo.")
    args = parser.parse_args()

    # --- Get Coordinates ---
    latitude_to_use = args.lat
    longitude_to_use = args.lon
    # (Coordinate prompting logic)
    if latitude_to_use is None:
        while True:
            try:
                lat_str = input("Please enter the target Latitude (WGS84 decimal degrees): ")
                latitude_to_use = float(lat_str)
                if -90 <= latitude_to_use <= 90: break
                else: print("Latitude must be between -90 and 90.")
            except ValueError: print("Invalid input...") # Shortened
            except Exception as e: print(f"An unexpected error occurred: {e}")
    if longitude_to_use is None:
        while True:
            try:
                lon_str = input("Please enter the target Longitude (WGS84 decimal degrees): ")
                longitude_to_use = float(lon_str)
                if -180 <= longitude_to_use <= 180: break
                else: print("Longitude must be between -180 and 180.")
            except ValueError: print("Invalid input...") # Shortened
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

    # <<< --- START MONITORING --- >>>
    print("-" * 30) # Separator
    monitor = ResourceMonitor(interval=0.2) # Sample every 0.2 seconds
    monitor.start()
    time.sleep(0.1) # Give monitor a tiny moment to start
    print("-" * 30) # Separator
    # <<< --- END START MONITORING --- >>>

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

    # <<< --- STOP MONITORING --- >>>
    print("-" * 30) # Separator
    usage_stats = monitor.stop()
    print("-" * 30) # Separator
    # <<< --- END STOP MONITORING --- >>>

    # --- Process Result (Including Image Display) ---
    print(f"\n--- Result ---")
    print(f"Status: {status}")

    display_windows_opened = False # Flag to track if we need waitKey
    if status == RESULT_SUCCESS and result_data:
        print(f"Calculated Latitude:  {result_data['lat']:.6f}")
        print(f"Calculated Longitude: {result_data['lon']:.6f}")
        print(f"Calculated Angle:     {result_data['angle']:.2f}")
        if not args.no_save: print(f"\nOutput images/coordinates potentially saved.")
        else: print(f"\nOutput saving disabled via --no-save.")

        # --- Display Logic ---
        print("\nAttempting to display result images...")
        display_scale = 0.7
        img_box = result_data.get('img_with_box')
        img_matches = result_data.get('img_matches_vis')

        if img_box is not None:
             try:
                 img_box_display = cv2.resize(img_box, None, fx=display_scale, fy=display_scale)
                 cv2.imshow("Map Chip with Box / Center", img_box_display)
                 display_windows_opened = True
             except Exception as e: print(f"Error displaying map chip with box: {e}")
        # else: print("Map chip with box image not available in results.") # Less verbose

        if img_matches is not None:
            try:
                h_match, w_match = img_matches.shape[:2]
                max_display_w = 1800
                img_matches_display = img_matches
                if w_match > max_display_w:
                    scale_match = max_display_w / w_match
                    img_matches_display = cv2.resize(img_matches, None, fx=scale_match, fy=scale_match, interpolation=cv2.INTER_AREA)
                cv2.imshow("Feature Matches (SuperGlue Inliers)", img_matches_display)
                display_windows_opened = True
            except Exception as e: print(f"Error displaying feature matches: {e}")
        # else: print("Feature matches visualization image not available in results.") # Less verbose

    # --- Error Handling for Localization Failure ---
    elif result_data and 'error' in result_data:
         print(f"Localization Error details: {result_data['error']}")
    else:
        print("Localization failed or encountered an unknown error.")

    # --- Display Monitoring Results ---
    if usage_stats:
        print("\n--- Resource Usage During run_localization ---")
        print(f"CPU Usage (%): Avg={usage_stats['avg_cpu']:.1f}, Max={usage_stats['max_cpu']:.1f}")
        print(f"RAM Usage (%): Avg={usage_stats['avg_ram']:.1f}, Max={usage_stats['max_ram']:.1f}")
        # Check if GPU stats are meaningful (available and some data collected)
        if "avg_gpu_util" in usage_stats and (usage_stats['avg_gpu_util'] > 0 or usage_stats['max_gpu_util'] > 0 or usage_stats['avg_gpu_mem'] > 0):
            print(f"GPU Utilization (%): Avg={usage_stats['avg_gpu_util']:.1f}, Max={usage_stats['max_gpu_util']:.1f}")
            print(f"GPU Memory Usage (%): Avg={usage_stats['avg_gpu_mem']:.1f}, Max={usage_stats['max_gpu_mem']:.1f}")
        elif GPU_MONITORING_AVAILABLE:
            print("GPU Monitoring: Enabled but failed to collect stats or usage was zero.")
        else:
            print("GPU Monitoring: Not available on this system.")
    else:
        print("\nMonitoring data unavailable (Monitor might have been stopped early or failed).")


    # --- Wait for Key Press if Windows Were Opened ---
    if display_windows_opened:
        print("\nPress any key in an image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Image windows closed.")

    # --- Shutdown GPU monitoring ---
    if GPU_MONITORING_AVAILABLE:
        try:
             pynvml.nvmlShutdown()
             print("NVML Shutdown.")
        except Exception as e: # Catch potential errors during shutdown too
             print(f"Error during nvmlShutdown: {e}")

    print("\n--- Script Finished ---")