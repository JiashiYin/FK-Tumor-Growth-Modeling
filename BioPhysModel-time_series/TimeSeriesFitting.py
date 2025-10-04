# TimeSeriesFitting.py

import os
import sys
import numpy as np
import math
import nibabel as nib
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse
import glob
from pathlib import Path

# Import custom classes
from TimeSeriesFittingSolver import TimeSeriesFittingSolver
from TimeSeriesLossVisualizer import TimeSeriesLossVisualizer
from TumorGrowthToolkit.FK import Solver as FKSolver


sys.stdout.reconfigure(line_buffering=True)  # Only works in Python 3.7+


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

def exponential_recency_weights(time_points, decay_factor=0.1):
    """Assign exponentially higher weights to more recent time points"""
    max_time = max(time_points)
    weights = [math.exp(decay_factor * (t - min(time_points))) for t in time_points]
    
    # Normalize weights
    total = sum(weights)
    return [w / total for w in weights]

def log_time_distance_weights(time_points, target_time=None):
    """Weight using logarithmic time distances for balanced importance"""
    if target_time is None:
        target_time = max(time_points)
    
    # Adjust times to avoid log(0)
    adjusted_times = [t + 1 for t in time_points]
    adjusted_target = target_time + 1
    
    # Calculate log-based weights
    weights = [1 / math.log(abs(adjusted_target - t) + math.e) for t in adjusted_times]
    
    # Normalize weights
    total = sum(weights)
    return [w / total for w in weights]

def load_time_series_data(base_dir, patient_id, time_points=None):
    """
    Load time series MRI data from the specific directory structure:
    data_dir/
      └── Patient-XXX/
          └── week-YYY/
              ├── seg_mask.nii.gz (tumor)
              ├── T1_pve_1.nii.gz (GM)
              └── T1_pve_2.nii.gz (WM)
    
    Args:
        base_dir: Base directory containing the data
        patient_id: Patient identifier (e.g. 'Patient-091')
        time_points: Optional list of time points to load (if None, load all available)
        
    Returns:
        Tuple of (wm_data, gm_data, tumor_series, actual_time_points, affine)
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
    wm_series = []
    gm_series = []
    
    # Load data for each week
    for week_num, _, wm_path, gm_path, tumor_path in week_info:
        wm_nifti = nib.load(wm_path)
        gm_nifti = nib.load(gm_path)
        tumor_nifti = nib.load(tumor_path)
        
        wm_data = wm_nifti.get_fdata()
        gm_data = gm_nifti.get_fdata()
        tumor_data = tumor_nifti.get_fdata()
        
        wm_series.append(wm_data)
        gm_series.append(gm_data)
        tumor_series.append(tumor_data)
        actual_time_points.append(week_num)
    
    # For the affine, use the first week's data
    affine = wm_nifti.affine
    
    print(f"Loaded {len(tumor_series)} time points: {actual_time_points}")
    
    # For the initial implementation, use the first week's WM/GM data
    # Later we can enhance to use interpolated WM/GM if needed
    return wm_series[0], gm_series[0], tumor_series, actual_time_points, affine

def initialize_settings(tumor_data, gm_shape, time_points, weighting_strategy='inverse', target_time=None):
    """
    Initialize the settings for the time-series solver.
    """
    # Calculate center of mass of the initial tumor
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
    settings["dw0"] = 0.001            # Initial diffusion coefficient
    settings["NxT1_pct0"] = float(com[0] / gm_shape[0])
    settings["NyT1_pct0"] = float(com[1] / gm_shape[1])
    settings["NzT1_pct0"] = float(com[2] / gm_shape[2])
    settings["t_start0"] = -5.0      # Initial estimate: tumor started 5 weeks before first observation
    settings["time_scale0"] = 1.0    # Initial estimate: model time = real time
    
    # Parameter ranges and other settings
    settings["parameterRanges"] = [
        [0, 1],          # NxT1_pct
        [0, 1],          # NyT1_pct
        [0, 1],          # NzT1_pct
        [0.0001, 3],      # Dw
        [0.0001, 0.225], # rho
        [-50, 0],        # t_start (tumor can start up to 50 weeks before first observation)
        [0.1, 5]         # time_scale (scaling between model time and real time)
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
    elif weighting_strategy == 'exponential':
        settings["time_weights"] = exponential_recency_weights(time_points)
    elif weighting_strategy == 'inverse':
        settings["time_weights"] = inverse_time_distance_weights(time_points, target_time)
    elif weighting_strategy == 'log':
        settings["time_weights"] = log_time_distance_weights(time_points, target_time)
    else:
        # Default to equal weighting
        settings["time_weights"] = [1.0] * len(time_points)
    
    # Print the weights for information
    print(f"Time weights: {[round(w, 3) for w in settings['time_weights']]}")
    
    # Early stopping settings
    settings["early_stopping"] = True
    settings["patience"] = 5
    
    # Optimization settings
    settings["workers"] = 9
    settings["sigma0"] = 0.3
    settings["resolution_factor"] = {0: 0.3, 0.4: 0.5, 0.7: 0.7}  # More progressive scaling
    settings["generations"] = 25
    
    return settings
def save_results(output_dir, settings, result_dict, time_series_prediction, time_points, affine):
    """
    Save the solver results and the final tumor prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save settings and results
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_settings.npy"), settings)
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_results.npy"), result_dict)
    
    # Save optimized parameters in a text file for easy reference
    opt_params = result_dict["opt_params"]
    with open(os.path.join(output_dir, "optimized_parameters.txt"), 'w') as f:
        f.write(f"NxT1_pct: {opt_params[0]:.6f}\n")
        f.write(f"NyT1_pct: {opt_params[1]:.6f}\n")
        f.write(f"NzT1_pct: {opt_params[2]:.6f}\n")
        f.write(f"Dw: {opt_params[3]:.6f}\n")
        f.write(f"rho: {opt_params[4]:.6f}\n")
        if len(opt_params) > 5:  # Check if new parameters exist
            f.write(f"t_start: {opt_params[5]:.6f}\n")
            f.write(f"time_scale: {opt_params[6]:.6f}\n")
    
    # Save tumor predictions for each time point
    for i, t in enumerate(time_points):
        if i < len(time_series_prediction):
            prediction = time_series_prediction[i].astype(np.float32)
            np.save(os.path.join(output_dir, f"prediction_t{t}.npy"), prediction)
            
            # Save as NIfTI
            nifti_img = nib.Nifti1Image(prediction, affine)
            nib.save(nifti_img, os.path.join(output_dir, f"prediction_t{t}.nii.gz"))

def main():
    """Main function to run the time-series fitting workflow."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run time-series tumor growth fitting.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing data')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Patient ID (optional, if using nested directory structure)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--time_points', type=int, nargs='+', default=None,
                        help='Time points to use (e.g., 0 2 5 7)')
    parser.add_argument('--loss_type', type=str, default='soft_dice',
                        choices=['dice', 'soft_dice', 'boundary', 'hausdorff', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--generations', type=int, default=12,
                        help='Number of generations for CMA-ES')
    parser.add_argument('--workers', type=int, default=9,
                        help='Number of worker processes for parallel computation')
    parser.add_argument('--predict_week', type=int, default=None,
                        help='Future week to predict (e.g., 10)')
    parser.add_argument('--weighting_strategy', type=str, default='inverse',
                    choices=['equal', 'exponential', 'inverse', 'log'],
                    help='Strategy for weighting time points')
    parser.add_argument('--target_time', type=int, default=None,
                    help='Target prediction time for weight calculation (default: max time point)')
    
    args = parser.parse_args()
    
    try:
        # Load time series data
        wm_data, gm_data, tumor_data, time_points, affine = load_time_series_data(
            args.data_dir, args.patient_id, args.time_points
        )
        
        # Initialize solver settings
        settings = initialize_settings(tumor_data, gm_data.shape, time_points, 
                              weighting_strategy=args.weighting_strategy,
                              target_time=args.target_time)
        
        # Override settings with command line arguments
        settings["loss_type"] = args.loss_type
        settings["generations"] = args.generations
        settings["workers"] = args.workers
        
        # Create output directory
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the time series fitting solver
        solver = TimeSeriesFittingSolver(settings, wm_data, gm_data, tumor_data, time_points)
        
        # Run the optimization
        print("Starting time-series fitting optimization...")
        start_time = time.time()
        predicted_time_series, result_dict = solver.run()
        end_time = time.time()
        
        # Save the results
        save_results(output_dir, settings, result_dict, predicted_time_series, time_points, affine)
        
        # Initialize visualizer and generate visualizations
        visualizer = TimeSeriesLossVisualizer(save_dir=os.path.join(output_dir, "visualizations"))
        
        # Compute a slice index for visualization
        tumor_sum = np.sum(tumor_data[0], axis=(0, 1))
        slice_index = np.argmax(tumor_sum)
        print(f"Using slice index {slice_index} for visualizations")
        
        # Generate visualizations
        visualizer.visualize_all(
            result_dict=result_dict,
            predicted_series=predicted_time_series,
            actual_series=tumor_data,
            time_points=time_points,
            wm_data=wm_data,
            slice_index=slice_index,
            voxel_volume_mm3=1.0  # Adjust based on your data's voxel size
        )
        
        # Print runtime information
        print(f"Total runtime: {round((end_time - start_time) / 60, 1)} minutes")
        print(f"Best parameters: {result_dict['opt_params']}")
        print(f"Minimum loss: {result_dict['minLoss']}")
        
        # Predict future time point if requested
        if args.predict_week is not None and args.predict_week > max(time_points):
            print(f"\nPredicting tumor growth at future time point: {args.predict_week} weeks")
            
            # Extract time parameters if available
            t_start = result_dict['opt_params'][5] if len(result_dict['opt_params']) > 5 else 0
            time_scale = result_dict['opt_params'][6] if len(result_dict['opt_params']) > 6 else 1.0
            
            # Extend time points and run prediction
            prediction_params = {
                'Dw': result_dict['opt_params'][3],
                'rho': result_dict['opt_params'][4],
                'NxT1_pct': result_dict['opt_params'][0],
                'NyT1_pct': result_dict['opt_params'][1],
                'NzT1_pct': result_dict['opt_params'][2],
                'gm': gm_data,
                'wm': wm_data,
                'segm': tumor_data[0],
                'tumor_data': tumor_data,
                'time_points': time_points + [args.predict_week],
                't_start': t_start,
                'time_scale': time_scale,
                'real_time_unit': 'week',
                'resolution_factor': 0.7
            }
            
            future_solver = FKSolver(prediction_params)
            future_result = future_solver.solve()
            
            if future_result['success']:
                # Save the future prediction
                future_prediction = future_result['time_series'][-1].astype(np.float32)
                
                # Save as numpy and NIfTI
                np.save(os.path.join(output_dir, f"prediction_t{args.predict_week}.npy"), future_prediction)
                future_nifti = nib.Nifti1Image(future_prediction, affine)
                nib.save(future_nifti, os.path.join(output_dir, f"prediction_t{args.predict_week}.nii.gz"))
                
                # Visualize the future prediction
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(wm_data[:, :, slice_index], cmap="gray", alpha=0.5)
                ax.imshow(future_prediction[:, :, slice_index], cmap="Reds", alpha=0.9)
                ax.set_title(f"Predicted Tumor at {args.predict_week} weeks")
                ax.axis('off')
                plt.savefig(os.path.join(output_dir, f"prediction_t{args.predict_week}.png"))
                plt.close()
                
                print(f"Future prediction at {args.predict_week} weeks saved successfully")
            else:
                print(f"Error in future prediction: {future_result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()