#!/usr/bin/env python
# run_training.py - Standalone robust training script

import os
import sys
import numpy as np
import time
import argparse
import glob
import nibabel as nib
from pathlib import Path
import signal
import traceback

def load_time_series_data(base_dir, patient_id, time_points=None):
    """
    Load time series MRI data from the specific directory structure:
    data_dir/
      └── Patient-XXX/
          └── week-YYY/
              ├── seg_mask.nii.gz (tumor)
              ├── T1_pve_1.nii.gz (GM)
              └── T1_pve_2.nii.gz (WM)
    """
    patient_dir = os.path.join(base_dir, patient_id)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    
    # Find all week directories
    week_dirs = sorted(glob.glob(os.path.join(patient_dir, 'week-*')))
    if not week_dirs:
        raise FileNotFoundError(f"No week directories found in {patient_dir}")
    
    # Extract week numbers and filter by requested time points
    week_info = []
    for week_dir in week_dirs:
        week_num = int(os.path.basename(week_dir).split('-')[1])
        if time_points is None or week_num in time_points:
            # Check required files exist
            wm_path = os.path.join(week_dir, 'T1_pve_2.nii.gz')
            gm_path = os.path.join(week_dir, 'T1_pve_1.nii.gz')
            tumor_path = os.path.join(week_dir, 'seg_mask.nii.gz')
            
            if os.path.exists(wm_path) and os.path.exists(gm_path) and os.path.exists(tumor_path):
                week_info.append((week_num, week_dir, wm_path, gm_path, tumor_path))
    
    if not week_info:
        raise FileNotFoundError(f"No valid week data found for specified time points")
    
    # Sort by week number
    week_info.sort(key=lambda x: x[0])
    
    # Initialize output arrays
    tumor_series = []
    actual_time_points = []
    
    # Load data for each week
    print(f"Loading data for {len(week_info)} time points...")
    
    for week_num, _, wm_path, gm_path, tumor_path in week_info:
        wm_nifti = nib.load(wm_path)
        gm_nifti = nib.load(gm_path)
        tumor_nifti = nib.load(tumor_path)
        
        wm_data = wm_nifti.get_fdata()
        gm_data = gm_nifti.get_fdata()
        tumor_data = tumor_nifti.get_fdata()
        
        # For the first week, save WM and GM
        if len(tumor_series) == 0:
            first_wm = wm_data
            first_gm = gm_data
            affine = wm_nifti.affine
        
        tumor_series.append(tumor_data)
        actual_time_points.append(week_num)
    
    print(f"Loaded {len(tumor_series)} time points: {actual_time_points}")
    
    return first_wm, first_gm, tumor_series, actual_time_points, affine

def inverse_time_distance_weights(time_points, target_time=None, power=1):
    """Weight based on inverse distance to the most recent or target time point"""
    if target_time is None:
        target_time = max(time_points)
    
    # Calculate inverse distances (avoid division by zero)
    epsilon = 1e-6
    weights = [1 / ((abs(target_time - t) + epsilon) ** power) for t in time_points]
    
    # Normalize weights
    total = sum(weights)
    return [w / total for w in weights]

def initialize_settings(tumor_data, gm_shape, time_points, weighting_strategy='inverse', target_time=None):
    """Initialize the settings for the time-series solver."""
    # Calculate center of mass of the initial tumor
    import scipy.ndimage as ndimage
    
    initial_tumor = tumor_data[0]
    tumor_mask = initial_tumor > 0.5
    
    if np.sum(tumor_mask) > 0:
        com = ndimage.center_of_mass(tumor_mask)
    else:
        # Default to center if tumor mask is empty
        com = (gm_shape[0] / 2, gm_shape[1] / 2, gm_shape[2] / 2)
    
    settings = {}
    
    # Initial parameters for the solver
    settings["rho0"] = 0.06          # Initial proliferation rate
    settings["dw0"] = 1.0            # Initial diffusion coefficient
    settings["NxT1_pct0"] = float(com[0] / gm_shape[0])
    settings["NyT1_pct0"] = float(com[1] / gm_shape[1])
    settings["NzT1_pct0"] = float(com[2] / gm_shape[2])
    
    # Parameter ranges and other settings
    settings["parameterRanges"] = [
        [0, 1],          # NxT1_pct
        [0, 1],          # NyT1_pct
        [0, 1],          # NzT1_pct
        [0.001, 3],      # Dw
        [0.0001, 0.225]  # rho
    ]
    
    # Loss function settings
    settings["loss_type"] = "combined"  # Options: dice, soft_dice, boundary, hausdorff, combined
    settings["soft_dice_margin"] = 2     # Margin for soft dice loss
    settings["dice_weight"] = 0.4
    settings["soft_weight"] = 0.3
    settings["boundary_weight"] = 0.2
    settings["hausdorff_weight"] = 0.1
    
    # Time-series specific settings
    settings["real_time_unit"] = "week"  # Time unit (week, day, month)
    
    # Set time weights based on strategy
    if weighting_strategy == 'equal':
        settings["time_weights"] = [1.0] * len(time_points)
    elif weighting_strategy == 'inverse':
        settings["time_weights"] = inverse_time_distance_weights(time_points, target_time)
    else:
        # Default to equal weighting
        settings["time_weights"] = [1.0] * len(time_points)
    
    # Print the weights for information
    print(f"Time weights: {[round(w, 3) for w in settings['time_weights']]}")
    
    # Early stopping settings
    settings["early_stopping"] = True
    settings["patience"] = 5
    
    # Optimization settings
    settings["workers"] = 4  # Reduced from default 9 for stability
    settings["sigma0"] = 0.02
    settings["resolution_factor"] = {0: 0.3, 0.4: 0.5, 0.7: 0.7}
    settings["generations"] = 25
    
    return settings

def save_results(output_dir, settings, result_dict, time_series_prediction, time_points, affine):
    """Save the solver results and the final tumor prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save settings and results
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_settings.npy"), settings)
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_results.npy"), result_dict)
    
    # Save optimized parameters in a text file for easy reference
    opt_params = result_dict["opt_params"]
    if opt_params is not None:
        with open(os.path.join(output_dir, "optimized_parameters.txt"), 'w') as f:
            f.write(f"NxT1_pct: {opt_params[0]:.6f}\n")
            f.write(f"NyT1_pct: {opt_params[1]:.6f}\n")
            f.write(f"NzT1_pct: {opt_params[2]:.6f}\n")
            f.write(f"Dw: {opt_params[3]:.6f}\n")
            f.write(f"rho: {opt_params[4]:.6f}\n")
    
    # Save tumor predictions for each time point
    if time_series_prediction is not None:
        for i, t in enumerate(time_points):
            if i < len(time_series_prediction):
                prediction = time_series_prediction[i].astype(np.float32)
                np.save(os.path.join(output_dir, f"prediction_t{t}.npy"), prediction)
                
                # Save as NIfTI
                nifti_img = nib.Nifti1Image(prediction, affine)
                nib.save(nifti_img, os.path.join(output_dir, f"prediction_t{t}.nii.gz"))

def cleanup_workers():
    """Force clean up any orphaned worker processes."""
    try:
        import psutil
        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)
        
        for child in current_process.children(recursive=True):
            try:
                print(f"Terminating child process: {child.pid}")
                child.terminate()
            except:
                pass
    except:
        print("Could not clean up worker processes")

def enable_timeout_protection(timeout_seconds=3600):
    """Enable timeout protection for the whole script."""
    def timeout_handler(signum, frame):
        print(f"Script timed out after {timeout_seconds} seconds!")
        cleanup_workers()
        sys.exit(1)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

def main():
    """Main function to run the robust training process."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run robust time-series tumor growth training.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing data')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient ID')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--time_points', type=int, nargs='+', default=None,
                        help='Time points to use (e.g., 0 2 5 7)')
    parser.add_argument('--loss_type', type=str, default='soft_dice',
                        choices=['dice', 'soft_dice', 'boundary', 'hausdorff', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--generations', type=int, default=13,
                        help='Number of generations for CMA-ES')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes for parallel computation')
    parser.add_argument('--weighting_strategy', type=str, default='inverse',
                    choices=['equal', 'inverse'],
                    help='Strategy for weighting time points')
    parser.add_argument('--target_time', type=int, default=None,
                    help='Target prediction time for weight calculation')
    parser.add_argument('--timeout', type=int, default=7200,
                    help='Timeout in seconds for the whole script')
    
    args = parser.parse_args()
    
    # Set timeout protection
    enable_timeout_protection(args.timeout)
    
    try:
        # Create output directory
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading time series data...")
        wm_data, gm_data, tumor_data, time_points, affine = load_time_series_data(
            args.data_dir, args.patient_id, args.time_points
        )
        
        print("Initializing solver settings...")
        settings = initialize_settings(tumor_data, gm_data.shape, time_points, 
                                      weighting_strategy=args.weighting_strategy,
                                      target_time=args.target_time)
        
        # Override settings with command line arguments
        settings["loss_type"] = args.loss_type
        settings["generations"] = args.generations
        settings["workers"] = args.workers
        settings["output_dir"] = output_dir  # Store output dir in settings too
        
        # Import solver here to avoid any circular dependencies
        print("Importing solver modules...")
        from TimeSeriesFittingSolver import TimeSeriesFittingSolver
        
        print("Initializing solver...")
        solver = TimeSeriesFittingSolver(settings, wm_data, gm_data, tumor_data, time_points)
        
        # Run the optimization with timeouts for different phases
        print("Starting optimization...")
        start_time = time.time()
        
        # Try with sequential-only mode first to see if it works better
        try:
            print("Running optimization with sequential processing...")
            original_workers = settings["workers"]
            settings["workers"] = 0  # Sequential-only
            predicted_time_series, result_dict = solver.run()
        except Exception as e:
            print(f"Sequential mode failed: {e}")
            print("Falling back to parallel processing with fewer workers...")
            settings["workers"] = min(2, original_workers)  # Use at most 2 workers
            predicted_time_series, result_dict = solver.run()
        
        end_time = time.time()
        
        # Save results
        print("Saving results...")
        save_results(output_dir, settings, result_dict, predicted_time_series, time_points, affine)
        
        # Print execution summary
        print("\n--- Execution Summary ---")
        print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Best loss: {result_dict.get('minLoss', 'unknown')}")
        if result_dict.get("opt_params") is not None:
            params = result_dict["opt_params"]
            print(f"Best parameters:")
            print(f"  NxT1_pct: {params[0]:.6f}")
            print(f"  NyT1_pct: {params[1]:.6f}")
            print(f"  NzT1_pct: {params[2]:.6f}")
            print(f"  Dw: {params[3]:.6f}")
            print(f"  rho: {params[4]:.6f}")
        else:
            print("No optimal parameters found!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always try to clean up processes
        cleanup_workers()
        
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
