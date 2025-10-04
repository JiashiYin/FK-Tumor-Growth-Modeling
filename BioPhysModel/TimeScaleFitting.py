import numpy as np 
import os
import nibabel as nib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import json
import argparse
from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import multiprocessing

# Global variables for module-level functions
_params = None
_wm = None
_gm = None
_initial_tumor = None
_target_tumor = None

def load_data(wm_path, gm_path, tumor_path):
    """
    Load white matter (WM), grey matter (GM), and tumor segmentation data.

    Args:
        wm_path (str): Path to the white matter data.
        gm_path (str): Path to the grey matter data.
        tumor_path (str): Path to the tumor segmentation data.

    Returns:
        tuple: White matter (WM), grey matter (GM), and tumor segmentation data.
    """
    WM = nib.load(wm_path).get_fdata()
    GM = nib.load(gm_path).get_fdata()
    segmentation = nib.load(tumor_path).get_fdata()

    return WM, GM, segmentation

def load_params(params_file):
    """
    Load previously fitted parameters from JSON file.

    Args:
        params_file (str): Path to the parameters JSON file.

    Returns:
        dict: Loaded parameters.
    """
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def dice_coefficient(a, b):
    """
    Calculate the Dice coefficient between two binary masks.

    Args:
        a (numpy.ndarray): First binary mask.
        b (numpy.ndarray): Second binary mask.

    Returns:
        float: Dice coefficient value.
    """
    boolA, boolB = a > 0.5, b > 0.5 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0

    return 2 * np.sum(np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

def forward_model(params, time_scale, wm, gm, initial_tumor, resolution_factor=0.7):
    """
    Run the forward tumor growth model with a time scale parameter.

    Args:
        params (dict): FK equation parameters.
        time_scale (float): Time scale factor.
        wm (numpy.ndarray): White matter data.
        gm (numpy.ndarray): Grey matter data.
        initial_tumor (numpy.ndarray): Initial tumor segmentation.
        resolution_factor (float, optional): Resolution factor for simulation.

    Returns:
        numpy.ndarray: Simulated tumor growth.
    """
    # Scale the growth time by the time scale factor
    scaled_time = params['growth_time'] * time_scale
    
    # Ensure NxT1_pct, NyT1_pct, NzT1_pct are between 0 and 1
    # Normalize by max dimension
    max_dim = max(wm.shape)
    
    # Setup forward model parameters
    forward_params = {
        'Dw': params['Dw'],
        'rho': params['rho'],
        'stopping_time': scaled_time,
        'RatioDw_Dg': 10,
        'gm': gm,
        'wm': wm,
        'NxT1_pct': wm.shape[0]/max_dim,  # Normalized to [0,1]
        'NyT1_pct': wm.shape[1]/max_dim,
        'NzT1_pct': wm.shape[2]/max_dim,
        'resolution_factor': resolution_factor,
        'segm': initial_tumor
    }
    
    # Run the forward model
    try:
        solver = fwdSolver(forward_params)
        result = solver.solve()
    except Exception as e:
        print(f"  Simulation failed with error: {str(e)}")
        return None
    
    if not result['success']:
        return None
        
    return result["final_state"]

def time_scale_loss_function(time_scale, params, wm, gm, initial_tumor, target_tumor, generation):
    """
    Calculate loss between simulated tumor and target tumor.

    Args:
        time_scale (float): Time scale parameter.
        params (dict): FK equation parameters.
        wm (numpy.ndarray): White matter data.
        gm (numpy.ndarray): Grey matter data.
        initial_tumor (numpy.ndarray): Initial tumor segmentation.
        target_tumor (numpy.ndarray): Target tumor segmentation to match.
        generation (int): Current generation number.

    Returns:
        float: Loss value (1 - Dice coefficient).
    """
    # Log time scale being evaluated
    print(f"Gen {generation}: Evaluating time scale: {time_scale:.4f}")
    
    # Run forward model with time scale
    predicted_tumor = forward_model(params, time_scale, wm, gm, initial_tumor)
    
    if predicted_tumor is None:
        print(f"  Simulation failed")
        return 1.0, None
    
    # Calculate Dice coefficient loss
    dice = dice_coefficient(predicted_tumor, target_tumor)
    loss = 1.0 - dice
    
    # Log result
    print(f"  Loss: {loss:.6f}, Dice: {dice:.6f}")
    
    # Create solution record using the same keys as in cmaesFK.py
    solution = {
        'time_scale': time_scale,
        'dice': dice,
        'lossTotal': loss,  # Key used in original code
        'allParams': [time_scale]  # Key used in original code
    }
    
    return loss, solution

# Module-level function for multiprocessing compatibility
def objective_function(x, gen):
    """Module-level objective function that can be pickled."""
    time_scale = x[0]
    return time_scale_loss_function(time_scale, _params, _wm, _gm, _initial_tumor, _target_tumor, gen)

def optimize_time_scale(params, wm, gm, initial_tumor, target_tumor, generations=10, workers=4):
    """
    Optimize the time scale parameter using CMA-ES.

    Args:
        params (dict): FK equation parameters.
        wm (numpy.ndarray): White matter data.
        gm (numpy.ndarray): Grey matter data.
        initial_tumor (numpy.ndarray): Initial tumor segmentation.
        target_tumor (numpy.ndarray): Target tumor segmentation to match.
        generations (int, optional): Number of generations for CMA-ES.
        workers (int, optional): Number of parallel workers.

    Returns:
        tuple: (optimal_time_scale, loss_value)
    """
    # Set global variables for the objective function
    global _params, _wm, _gm, _initial_tumor, _target_tumor
    _params = params
    _wm = wm
    _gm = gm
    _initial_tumor = initial_tumor
    _target_tumor = target_tumor
    
    # Initial value for time scale (start with 1.0)
    initial_time_scale = 1.0
    initial_sigma = 0.3  # Increased sigma for better exploration
    param_range = [[0.1, 10.0]]  # Range for time scale parameter
    
    # Run CMA-ES for time scale optimization
    print(f"Starting time scale optimization with {generations} generations and {workers} workers")
    
    # Run CMA-ES with trace=True to get the history
    trace = cmaes.cmaes(
        objective_function,
        [initial_time_scale], 
        initial_sigma, 
        generations, 
        workers=workers, 
        trace=True, 
        parameterRange=param_range
    )
    
    # Extract trace elements
    _, _, _, _, _, _, _, _, _, _, lossDir = zip(*trace)
    
    # Find best solution using the same approach as in cmaesFK.py
    min_loss = 1.0
    optimal_time_scale = initial_time_scale  # Default in case optimization fails
    
    for i in range(len(lossDir)):
        for j in range(len(lossDir[i])):
            if lossDir[i][j] is not None and "lossTotal" in lossDir[i][j]:
                if lossDir[i][j]["lossTotal"] <= min_loss:
                    min_loss = lossDir[i][j]["lossTotal"]
                    if "allParams" in lossDir[i][j]:
                        optimal_time_scale = lossDir[i][j]["allParams"][0]
    
    print(f"Optimization completed. Optimal time scale: {optimal_time_scale:.4f} with loss: {min_loss:.6f}")
    return optimal_time_scale, min_loss

def save_results(params, time_scale, loss, output_dir):
    """
    Save the time scale optimization results.

    Args:
        params (dict): FK equation parameters.
        time_scale (float): Optimized time scale.
        loss (float): Loss value.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a new parameters dictionary with time scale
    full_params = params.copy()
    full_params['time_scale'] = time_scale
    full_params['time_scale_loss'] = loss
    full_params['time_scale_dice'] = 1.0 - loss
    
    # Save as JSON
    with open(os.path.join(output_dir, "time_scaled_params.json"), 'w') as f:
        json.dump(full_params, f, indent=4)
    
    # Also save as numpy array for compatibility
    np.save(os.path.join(output_dir, "time_scaled_params.npy"), full_params)
    
    print(f"Saved time-scaled parameters to {output_dir}")

def visualize_results(wm, gm, initial_tumor, target_tumor, predicted_tumor, output_dir):
    """
    Visualize the time scale optimization results.

    Args:
        wm (numpy.ndarray): White matter data.
        gm (numpy.ndarray): Grey matter data.
        initial_tumor (numpy.ndarray): Initial tumor segmentation.
        target_tumor (numpy.ndarray): Target tumor segmentation.
        predicted_tumor (numpy.ndarray): Predicted tumor with optimized time scale.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find a good slice to visualize (center of tumor)
    tumor_sum = initial_tumor + target_tumor
    if np.sum(tumor_sum) > 0:
        z_slice = int(ndimage.center_of_mass(tumor_sum)[2])
    else:
        z_slice = gm.shape[2] // 2  # Default to middle slice
    
    # Create visualization of initial, target, and predicted tumors
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial tumor
    axes[0].imshow(gm[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[0].imshow(wm[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[0].imshow(initial_tumor[:, :, z_slice], cmap="Blues", alpha=0.8)
    axes[0].set_title("Initial Tumor")
    axes[0].axis('off')
    
    # Target tumor
    axes[1].imshow(gm[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[1].imshow(wm[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[1].imshow(target_tumor[:, :, z_slice], cmap="Greens", alpha=0.8)
    axes[1].set_title("Target Tumor")
    axes[1].axis('off')
    
    # Predicted tumor
    axes[2].imshow(gm[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[2].imshow(wm[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[2].imshow(predicted_tumor[:, :, z_slice], cmap="Reds", alpha=0.8)
    axes[2].set_title("Predicted Tumor (Time-Scaled)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_scale_comparison.png"))
    plt.close()

def get_data_path(base_dir, patient_id, time_point):
    """
    Convert time point to path in data directory with the correct structure:
    data_dir/Patient-XXX/week-YYY/
    
    Args:
        base_dir (str): Base data directory
        patient_id (str): Patient ID (e.g., 'Patient-042')
        time_point (int or str): Time point
        
    Returns:
        str: Full path to the specific time point data directory
    """
    # Convert time_point to integer if it's a string
    if isinstance(time_point, str) and time_point.isdigit():
        time_point = int(time_point)
    
    # Format the week directory according to the correct pattern
    if isinstance(time_point, int):
        week_dir = f"week-{time_point:03d}"  # Format as 'week-000', 'week-004', etc.
    else:
        # If it's already a formatted string, use it directly
        week_dir = time_point if time_point.startswith("week-") else f"week-{time_point}"
    
    # Build and return the full path
    return os.path.join(base_dir, patient_id, week_dir)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Optimize time scale parameter for tumor growth model.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing input data')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient ID (e.g., Patient-042)')
    parser.add_argument('--initial_time', type=str, required=True,
                        help='Initial time point (e.g., 0 for week-000)')
    parser.add_argument('--target_time', type=str, required=True,
                        help='Target time point (e.g., 4 for week-004)')
    parser.add_argument('--params_file', type=str, required=True,
                        help='Path to JSON file with fitted FK parameters')
    parser.add_argument('--output_dir', type=str, default='./time_scale_results',
                        help='Directory to save results')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations for CMA-ES algorithm')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for parallel processing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load FK parameters from previous fitting
    print(f"Loading FK parameters from: {args.params_file}")
    params = load_params(args.params_file)
    
    # Get directory paths for initial and target times
    initial_time_dir = get_data_path(args.data_dir, args.patient_id, args.initial_time)
    target_time_dir = get_data_path(args.data_dir, args.patient_id, args.target_time)
    
    # Load data for initial and target time points
    initial_wm_path = os.path.join(initial_time_dir, 'T1_pve_2.nii.gz')
    initial_gm_path = os.path.join(initial_time_dir, 'T1_pve_1.nii.gz')
    initial_tumor_path = os.path.join(initial_time_dir, 'seg_mask.nii.gz')
    
    target_wm_path = os.path.join(target_time_dir, 'T1_pve_2.nii.gz')
    target_gm_path = os.path.join(target_time_dir, 'T1_pve_1.nii.gz')
    target_tumor_path = os.path.join(target_time_dir, 'seg_mask.nii.gz')
    
    print(f"Loading initial time point data from: {initial_time_dir}")
    initial_wm, initial_gm, initial_tumor = load_data(initial_wm_path, initial_gm_path, initial_tumor_path)
    
    print(f"Loading target time point data from: {target_time_dir}")
    target_wm, target_gm, target_tumor = load_data(target_wm_path, target_gm_path, target_tumor_path)
    
    # Optimize time scale parameter
    print("Starting time scale optimization...")
    start_time = time.time()
    
    optimal_time_scale, min_loss = optimize_time_scale(
        params, 
        initial_wm, 
        initial_gm, 
        initial_tumor, 
        target_tumor,
        args.generations,
        args.workers
    )
    
    end_time = time.time()
    print(f"Time scale optimization completed in {(end_time - start_time) / 60:.2f} minutes")
    
    # Generate final prediction with optimal time scale
    print("Generating final prediction with optimal time scale...")
    predicted_tumor = forward_model(
        params, 
        optimal_time_scale, 
        initial_wm, 
        initial_gm, 
        initial_tumor, 
        resolution_factor=0.8  # Higher resolution for final prediction
    )
    
    if predicted_tumor is None:
        raise RuntimeError("Failed to generate final prediction with optimal time scale.")
    
    # Save optimized time scale and parameters
    save_results(params, optimal_time_scale, min_loss, args.output_dir)
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_results(
        initial_wm, 
        initial_gm, 
        initial_tumor, 
        target_tumor, 
        predicted_tumor, 
        os.path.join(args.output_dir, 'visualizations')
    )
    
    # Save the predicted tumor
    nib_img = nib.Nifti1Image(predicted_tumor.astype(np.float32), np.eye(4))
    nib.save(nib_img, os.path.join(args.output_dir, 'predicted_tumor.nii.gz'))
    
    print(f"Time scale optimization completed successfully. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()