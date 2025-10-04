import os
import numpy as np
import nibabel as nib
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import matplotlib.colors
from TumorGrowthToolkit.FK import Solver as FKSolver
import glob
import argparse
from pathlib import Path

def load_nifti_data(file_path):
    """Load a NIfTI file and return the image data as a NumPy array."""
    nifti_data = nib.load(file_path)
    return nifti_data.get_fdata()

def save_nifti(data, affine, output_path):
    """Save data as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

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

def setup_parameters(gm_data, wm_data, tumor_data, time_points, params=None, settings=None):
    """
    Set up the parameters for the FK Solver based on input data.
    """
    parameters = {
        'Dw': 0.05,                   # Diffusion coefficient for white matter
        'rho': 0.05,                  # Proliferation rate
        'NxT1_pct': 0.3,              # Tumor position [%]
        'NyT1_pct': 0.3,
        'NzT1_pct': 0.3,
        't_start': 0,                 # When tumor started relative to first observation
        'time_scale': 1.0,            # Scaling between model time and real time
        'gm': gm_data,                # Grey matter data
        'wm': wm_data,                # White matter data
        'segm': tumor_data[0],        # Initial segmentation data
        'tumor_data': tumor_data,     # All tumor data
        'time_points': time_points,   # Time points of the tumor data
        'real_time_unit': 'week',     # Time unit (e.g., week, day, month)
        'RatioDw_Dg': 100,            # Ratio of diffusion coefficients in white and grey matter
        'init_scale': 1.0,            # Scale of the initial Gaussian
        'resolution_factor': 0.6,     # Resolution scaling for calculations
        'th_matter': 0.1,             # Threshold to stop diffusing (gm + wm > th_matter)
        'verbose': True,              # Print timesteps
        'time_series_solution_Nt': None,  # Number of timesteps in the output (auto from time_points)
    }
    
    if params:
        for key, value in params.items():
            if key in parameters:
                parameters[key] = value
    
    if settings:
        for key, value in settings.items():
            if key in parameters:
                parameters[key] = value
    
    return parameters

def predict_future_timepoint(parameters, future_timepoint):
    """
    Predict tumor state at a future time point.
    """
    # Create a copy of parameters to avoid modifying the original
    prediction_params = parameters.copy()
    
    # Extend time points to include the future time point
    original_time_points = prediction_params['time_points']
    prediction_params['time_points'] = original_time_points + [future_timepoint]
    
    # Run simulation
    solver = FKSolver(prediction_params)
    result = solver.solve()
    
    if not result['success']:
        print(f"Error occurred during prediction: {result.get('error', 'Unknown error')}")
        return None
    
    # Get the predicted tumor state at the future time point
    future_index = len(original_time_points)
    if future_index < len(result['time_series']):
        return result['time_series'][future_index]
    else:
        print("Prediction failed: Future time point not captured in the simulation")
        return None

def plot_time_series_results(wm_data, actual_series, predicted_series, time_points, slice_index, save_dir):
    """
    Plot and save the actual and predicted time series results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot comparison for each time point
    n_timepoints = len(time_points)
    fig, axes = plt.subplots(2, n_timepoints, figsize=(5*n_timepoints, 10))
    
    # Create custom color maps
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)
    
    # Handle single timepoint case
    if n_timepoints == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    for i, t in enumerate(time_points):
        # Plot actual tumor
        ax1 = axes[0, i]
        ax1.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        if i < len(actual_series):
            ax1.imshow(actual_series[i][:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        ax1.set_title(f"Actual - Time {t}")
        ax1.axis('off')
        
        # Plot predicted tumor
        ax2 = axes[1, i]
        ax2.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        if i < len(predicted_series):
            ax2.imshow(predicted_series[i][:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        ax2.set_title(f"Predicted - Time {t}")
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_img_path = os.path.join(save_dir, 'time_series_comparison.png')
    plt.savefig(comparison_img_path)
    plt.close()
    
    # Save individual masks as numpy arrays
    for i, t in enumerate(time_points):
        if i < len(actual_series):
            np.save(os.path.join(save_dir, f'actual_tumor_t{t}.npy'), actual_series[i])
        if i < len(predicted_series):
            np.save(os.path.join(save_dir, f'predicted_tumor_t{t}.npy'), predicted_series[i])

def load_fitted_params(param_file):
    """Load fitted parameters from a numpy file."""
    if not os.path.exists(param_file):
        print(f"Warning: Parameter file {param_file} not found. Using default parameters.")
        return None
    
    data = np.load(param_file, allow_pickle=True)
    
    if isinstance(data, np.ndarray) and data.dtype == np.dtype('O'):
        # This is likely a dictionary stored as an object array
        data = data.item()
    
    if isinstance(data, dict) and 'opt_params' in data:
        opt_params = data['opt_params']
        
        # Basic parameters (always present)
        params = {
            'Dw': opt_params[3],
            'rho': opt_params[4],
            'NxT1_pct': opt_params[0],
            'NyT1_pct': opt_params[1],
            'NzT1_pct': opt_params[2],
        }
        
        # Add time parameters if available (new)
        if len(opt_params) > 5:
            params['t_start'] = opt_params[5]
            params['time_scale'] = opt_params[6]
    else:
        print("Warning: Could not extract parameters from file. Using default parameters.")
        params = None
    
    return params

def main():
    """Main function to run the time-series tumor growth prediction."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run time-series tumor growth prediction.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing data')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Patient ID (optional, if using nested directory structure)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--time_points', type=int, nargs='+', default=None,
                        help='Time points to use (e.g., 0 2 5 7)')
    parser.add_argument('--predict_week', type=int, default=None,
                        help='Future week to predict (e.g., 10)')
    parser.add_argument('--param_file', type=str, default=None,
                        help='File containing fitted parameters')
    parser.add_argument('--slice_index', type=int, default=None,
                        help='Slice index for visualization (default: auto-select)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load time series data
        wm_data, gm_data, tumor_data, time_points, affine = load_time_series_data(
            args.data_dir, args.patient_id, args.time_points
        )
        
        # If time_points was not provided as an argument, use the ones from the data
        if args.time_points is None:
            args.time_points = time_points
        
        # Load fitted parameters if provided
        if args.param_file:
            params = load_fitted_params(args.param_file)
        else:
            # Use default parameters
            params = {
                'Dw': 0.05,
                'rho': 0.05,
                'NxT1_pct': 0.3,
                'NyT1_pct': 0.3,
                'NzT1_pct': 0.3,
                't_start': 0,
                'time_scale': 1.0,
            }
        
        # Additional settings
        settings = {
            'RatioDw_Dg': 120,
            'th_matter': 0.15,
            'init_scale': 1.2,
            'resolution_factor': 0.7,
            'real_time_unit': 'week'
        }
        
        # Setup parameters for the time-series simulation
        parameters = setup_parameters(gm_data, wm_data, tumor_data, time_points, params, settings)
        
        # Run the time-series simulation
        start_time = time.time()
        fk_solver = FKSolver(parameters)
        result = fk_solver.solve()
        end_time = time.time()
        
        # Calculate execution time
        execution_time = int(end_time - start_time)
        print(f"Execution Time: {execution_time} seconds")
        
        # Check if simulation was successful
        if result['success']:
            print("Simulation successful!")
            
            # Get the predicted time series
            predicted_series = result['time_series']
            
            # Compute a slice index for visualization if not provided
            if args.slice_index is None:
                # Find a slice with tumor
                tumor_sum = np.sum(tumor_data[0], axis=(0, 1))
                slice_index = np.argmax(tumor_sum)
                print(f"Auto-selected slice index: {slice_index}")
            else:
                slice_index = args.slice_index
            
            # Plot and save the comparison between actual and predicted tumors
            plot_time_series_results(
                wm_data, tumor_data, predicted_series, time_points, 
                slice_index, args.output_dir
            )
            
            # Save predicted tumor masks as NIfTI files
            for i, t in enumerate(time_points):
                if i < len(predicted_series):
                    pred_file = os.path.join(args.output_dir, f'predicted_tumor_t{t}.nii.gz')
                    save_nifti(predicted_series[i].astype(np.float32), affine, pred_file)
            
            # Predict future time point if requested
            if args.predict_week is not None and args.predict_week > max(time_points):
                print(f"\nPredicting tumor state at future time point: {args.predict_week} weeks")
                
                future_tumor = predict_future_timepoint(parameters, args.predict_week)
                
                if future_tumor is not None:
                    # Save the future prediction
                    future_np_file = os.path.join(args.output_dir, f'predicted_tumor_t{args.predict_week}.npy')
                    future_nifti_file = os.path.join(args.output_dir, f'predicted_tumor_t{args.predict_week}.nii.gz')
                    
                    np.save(future_np_file, future_tumor)
                    save_nifti(future_tumor.astype(np.float32), affine, future_nifti_file)
                    
                    # Visualize the future prediction
                    fig, ax = plt.subplots(figsize=(10, 5))
                    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
                    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)
                    
                    ax.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
                    ax.imshow(future_tumor[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
                    ax.set_title(f"Predicted Tumor at {args.predict_week} weeks")
                    ax.axis('off')
                    
                    future_img_file = os.path.join(args.output_dir, f'predicted_tumor_t{args.predict_week}.png')
                    plt.savefig(future_img_file)
                    plt.close()
                    
                    print(f"Future prediction saved to {future_nifti_file}")
            
            print(f"All results saved to {args.output_dir}")
        else:
            print(f"Error occurred: {result['error']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()