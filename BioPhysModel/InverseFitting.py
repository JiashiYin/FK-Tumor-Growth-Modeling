#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import json
import time
from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import multiprocessing

# Global variables for sharing across processes
_WM = None
_GM = None
_segmentation = None

def load_data(data_dir, wm_file, gm_file, tumor_file):
    """
    Load input data for tumor growth modeling.
    
    Args:
        data_dir (str): Directory containing input data
        wm_file (str): Filename for white matter segmentation
        gm_file (str): Filename for gray matter segmentation
        tumor_file (str): Filename for tumor segmentation
        
    Returns:
        tuple: WM, GM, tumor segmentations, and initialization position
    """
    print(f"Loading data from: {data_dir}")
    
    # Load MRI data
    wm_path = os.path.join(data_dir, wm_file)
    gm_path = os.path.join(data_dir, gm_file)
    tumor_path = os.path.join(data_dir, tumor_file)
    
    if not all(os.path.exists(p) for p in [wm_path, gm_path, tumor_path]):
        raise FileNotFoundError(f"Input files not found in {data_dir}")
    
    WM = nib.load(wm_path).get_fdata()
    GM = nib.load(gm_path).get_fdata()
    segmentation = nib.load(tumor_path).get_fdata()
    
    # Find tumor center of mass for initialization
    if np.sum(segmentation) > 0:
        com = ndimage.center_of_mass(segmentation)
        pos = (
            com[0] / segmentation.shape[0],
            com[1] / segmentation.shape[1],
            com[2] / segmentation.shape[2]
        )
    else:
        # If no tumor, initialize at center
        pos = (0.5, 0.5, 0.5)
    
    return WM, GM, segmentation, pos

def loss_function(params, WM, GM, segmentation, generation, resolution_factor=0.5):
    """
    Calculate loss between predicted and actual tumor.
    
    Args:
        params (list): Parameters [Dw, rho, time, pos_x, pos_y, pos_z]
        WM (ndarray): White matter segmentation
        GM (ndarray): Gray matter segmentation
        segmentation (ndarray): Actual tumor segmentation
        generation (int): Current generation number
        resolution_factor (float): Resolution factor for simulation
        
    Returns:
        tuple: (loss, metadata_dict)
    """
    # Extract parameters
    Dw = params[0]
    rho = params[1]
    stop_time = params[2]
    pos_x = params[3]
    pos_y = params[4]
    pos_z = params[5]
    
    # Log parameter combination
    print(f"Gen {generation}: Params [Dw={Dw:.4f}, rho={rho:.4f}, t={stop_time:.1f}, pos=({pos_x:.2f},{pos_y:.2f},{pos_z:.2f})]")
    
    # Initialize tumor at specified position
    init_tumor = np.zeros_like(segmentation)
    i, j, k = int(pos_x * segmentation.shape[0]), int(pos_y * segmentation.shape[1]), int(pos_z * segmentation.shape[2])
    init_tumor[i, j, k] = 1.0
    
    # Create initial segmentation with a single cell at the specified position
    init_seg = ndimage.binary_dilation(init_tumor, iterations=1).astype(np.float32)
    
    # Set up simulation parameters
    # Normalize by max dimension to ensure values < 1
    max_dim = max(WM.shape)
    
    sim_params = {
        'Dw': Dw,
        'rho': rho,
        'stopping_time': stop_time,
        'RatioDw_Dg': 10,  # Default ratio for Dw/Dg
        'gm': GM,
        'wm': WM,
        'segm': init_seg,
        'NxT1_pct': WM.shape[0]/max_dim,  # Normalized to [0,1]
        'NyT1_pct': WM.shape[1]/max_dim,
        'NzT1_pct': WM.shape[2]/max_dim,
        'resolution_factor': resolution_factor
    }
    
    # Run forward simulation
    try:
        solver = fwdSolver(sim_params)
        result = solver.solve()
    except Exception as e:
        print(f"  Simulation failed with error: {str(e)}")
        return 1.0, None
    
    # Check for simulation success
    if not result.get('success', False):
        print(f"  Simulation failed")
        return 1.0, None
    
    # Get simulated tumor
    simulated_tumor = result["final_state"]
    
    # Calculate Dice coefficient
    def dice_coeff(y_true, y_pred, smooth=1.0):
        y_true_f = y_true.flatten() > 0.5
        y_pred_f = y_pred.flatten() > 0.5
        
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    dice = dice_coeff(segmentation, simulated_tumor)
    loss = 1.0 - dice
    
    # Log loss result
    print(f"  Loss: {loss:.6f}, Dice: {dice:.6f}")
    
    # Create solution record in the format expected by the optimizer
    solution = {
        "params": params,
        "Dw": Dw,
        "rho": rho, 
        "time": stop_time,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "dice": dice,
        "lossTotal": loss,  # Key used in original code
        "allParams": params # Key used in original code
    }
    
    return loss, solution

# Module-level function for multiprocessing compatibility
def objective_function(x, generation):
    """Global objective function that uses module-level variables."""
    return loss_function(x, _WM, _GM, _segmentation, generation)

def fk_parameter_optimization(WM, GM, segmentation, init_pos, generations=10, workers=8):
    """
    Optimize FK equation parameters using CMA-ES.
    
    Args:
        WM (ndarray): White matter segmentation
        GM (ndarray): Gray matter segmentation
        segmentation (ndarray): Actual tumor segmentation
        init_pos (tuple): Initial position (x,y,z) normalized to [0,1]
        generations (int): Number of generations for CMA-ES
        workers (int): Number of parallel workers
        
    Returns:
        dict: Best parameters and convergence info
    """
    print("Initializing solver settings...")
    
    # Set the global variables
    global _WM, _GM, _segmentation
    _WM = WM
    _GM = GM
    _segmentation = segmentation
    
    # Initial parameter values and ranges
    initial_params = [
        1.0,     # Dw
        0.05,    # rho
        50.0,    # stop_time
        init_pos[0],  # pos_x
        init_pos[1],  # pos_y
        init_pos[2]   # pos_z
    ]
    
    initial_sigma = 0.3  # Increased for better exploration
    
    # Parameter ranges for CMA-ES
    param_range = [
        [0.5, 2.0],      # Dw range
        [0.001, 0.1],    # rho range
        [30.0, 100.0],   # stop_time range
        [max(0.2, init_pos[0]-0.1), min(0.8, init_pos[0]+0.1)],  # pos_x range
        [max(0.2, init_pos[1]-0.1), min(0.8, init_pos[1]+0.1)],  # pos_y range
        [max(0.2, init_pos[2]-0.1), min(0.8, init_pos[2]+0.1)]   # pos_z range
    ]
    
    # Run CMA-ES optimization
    print(f"Starting CMA-ES with {generations} generations and {workers} workers")
    
    # Run CMA-ES with trace=True to get the full history
    trace = cmaes.cmaes(
        objective_function,
        initial_params,
        initial_sigma, 
        generations, 
        workers=workers, 
        trace=True,
        parameterRange=param_range
    )
    
    # Process trace to find best solution
    min_loss = 1.0
    best_params = None
    
    # Extract lossDir from trace
    _, _, _, _, _, _, _, _, _, _, lossDir = zip(*trace)
    
    # Find the best solution using the same approach as in cmaesFK.py
    for i in range(len(lossDir)):
        for j in range(len(lossDir[i])):
            if lossDir[i][j] is not None and "lossTotal" in lossDir[i][j]:
                if lossDir[i][j]["lossTotal"] <= min_loss:
                    min_loss = lossDir[i][j]["lossTotal"]
                    if "allParams" in lossDir[i][j]:
                        best_params = lossDir[i][j]["allParams"]
    
    if best_params is None:
        # No valid solutions found - abort
        raise RuntimeError("No valid solutions found during optimization. Check model parameters or data.")
    
    # Calculate normalized parameters for saving
    max_dim = max(WM.shape)
    
    # Extract each parameter from the best solution
    Dw = float(best_params[0])
    rho = float(best_params[1])
    growth_time = float(best_params[2])
    pos_x = float(best_params[3])
    pos_y = float(best_params[4])
    pos_z = float(best_params[5])
    
    # Assemble result
    result = {
        "Dw": Dw,
        "rho": rho,
        "growth_time": growth_time,
        "NxT1_pct": float(WM.shape[0]/max_dim),
        "NyT1_pct": float(WM.shape[1]/max_dim), 
        "NzT1_pct": float(WM.shape[2]/max_dim),
        "init_pos_x": pos_x,
        "init_pos_y": pos_y,
        "init_pos_z": pos_z,
        "dice": float(1.0 - min_loss),
        "loss": float(min_loss)
    }
    
    print(f"Optimization completed. Best parameters:")
    print(f"  Dw: {result['Dw']:.6f}")
    print(f"  rho: {result['rho']:.6f}")
    print(f"  growth_time: {result['growth_time']:.2f}")
    print(f"  position: ({result['init_pos_x']:.2f}, {result['init_pos_y']:.2f}, {result['init_pos_z']:.2f})")
    print(f"  dice: {result['dice']:.6f}")
    
    return result

def visualize_results(WM, GM, tumor, predicted, output_dir):
    """
    Create visualizations of the actual and predicted tumors.
    
    Args:
        WM (ndarray): White matter segmentation
        GM (ndarray): Gray matter segmentation
        tumor (ndarray): Actual tumor segmentation
        predicted (ndarray): Predicted tumor from model
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find a good slice to visualize (center of tumor)
    if np.sum(tumor) > 0:
        z_slice = int(ndimage.center_of_mass(tumor)[2])
    else:
        z_slice = tumor.shape[2] // 2
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot MRI with white and gray matter
    axes[0].imshow(GM[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[0].imshow(WM[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[0].set_title("MRI (White/Gray Matter)")
    axes[0].axis('off')
    
    # Plot actual tumor
    axes[1].imshow(GM[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[1].imshow(WM[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[1].imshow(tumor[:, :, z_slice], cmap="Reds", alpha=0.8)
    axes[1].set_title("Actual Tumor")
    axes[1].axis('off')
    
    # Plot predicted tumor
    axes[2].imshow(GM[:, :, z_slice], cmap="Greys", alpha=0.5)
    axes[2].imshow(WM[:, :, z_slice], cmap="Greys", alpha=0.25)
    axes[2].imshow(predicted[:, :, z_slice], cmap="Blues", alpha=0.8)
    axes[2].set_title("Predicted Tumor")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tumor_comparison.png"))
    plt.close()
    
    # Plot overlay showing difference
    plt.figure(figsize=(8, 8))
    plt.imshow(GM[:, :, z_slice], cmap="Greys", alpha=0.5)
    plt.imshow(WM[:, :, z_slice], cmap="Greys", alpha=0.25)
    plt.imshow(tumor[:, :, z_slice], cmap="Reds", alpha=0.5)
    plt.imshow(predicted[:, :, z_slice], cmap="Blues", alpha=0.5)
    plt.title("Overlay: Actual (Red) vs Predicted (Blue)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tumor_overlay.png"))
    plt.close()
    
    # Plot 3D axes
    fig = plt.figure(figsize=(15, 5))
    
    # X-axis mid-slice
    ax1 = fig.add_subplot(131)
    x_slice = tumor.shape[0] // 2
    ax1.imshow(np.rot90(tumor[x_slice, :, :]), cmap="Reds", alpha=0.7)
    ax1.imshow(np.rot90(predicted[x_slice, :, :]), cmap="Blues", alpha=0.7)
    ax1.set_title("X-axis Mid-slice")
    ax1.axis('off')
    
    # Y-axis mid-slice
    ax2 = fig.add_subplot(132)
    y_slice = tumor.shape[1] // 2
    ax2.imshow(np.rot90(tumor[:, y_slice, :]), cmap="Reds", alpha=0.7)
    ax2.imshow(np.rot90(predicted[:, y_slice, :]), cmap="Blues", alpha=0.7)
    ax2.set_title("Y-axis Mid-slice")
    ax2.axis('off')
    
    # Z-axis mid-slice
    ax3 = fig.add_subplot(133)
    ax3.imshow(tumor[:, :, z_slice], cmap="Reds", alpha=0.7)
    ax3.imshow(predicted[:, :, z_slice], cmap="Blues", alpha=0.7)
    ax3.set_title("Z-axis Mid-slice")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tumor_3d_views.png"))
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fit FK equation parameters to tumor data")
    parser.add_argument("--data_dir", required=True, help="Directory with input data")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--wm_file", default="T1_pve_2.nii.gz", help="White matter segmentation filename")
    parser.add_argument("--gm_file", default="T1_pve_1.nii.gz", help="Gray matter segmentation filename")
    parser.add_argument("--tumor_file", default="seg_mask.nii.gz", help="Tumor segmentation filename")
    parser.add_argument("--generations", type=int, default=10, help="Number of CMA-ES generations")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    WM, GM, tumor, init_pos = load_data(args.data_dir, args.wm_file, args.gm_file, args.tumor_file)

    print("Tumor data summary:")
    print(f"  Tumor segmentation has {np.sum(tumor > 0.5)} voxels with values > 0.5")
    print(f"  White matter has {np.sum(WM > 0.5)} voxels with values > 0.5") 
    print(f"  Gray matter has {np.sum(GM > 0.5)} voxels with values > 0.5")


    
    # Run parameter optimization
    best_params = fk_parameter_optimization(
        WM, GM, tumor, init_pos, 
        generations=args.generations,
        workers=args.workers
    )
    
    # Save best parameters to JSON
    with open(os.path.join(args.output_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Also save as numpy array for compatibility
    np.save(os.path.join(args.output_dir, "best_params.npy"), best_params)
    
    # Run a final forward simulation with best parameters
    if args.visualize:
        # Initialize tumor at best position
        init_tumor = np.zeros_like(tumor)
        i = int(best_params["init_pos_x"] * tumor.shape[0])
        j = int(best_params["init_pos_y"] * tumor.shape[1])
        k = int(best_params["init_pos_z"] * tumor.shape[2])
        init_tumor[i, j, k] = 1.0
        
        init_seg = ndimage.binary_dilation(init_tumor, iterations=1).astype(np.float32)
        
        # Set up simulation parameters for final visualization
        # Calculate max dimension for normalization
        max_dim = max(tumor.shape)
        
        params = {
            'Dw': best_params["Dw"],
            'rho': best_params["rho"],
            'stopping_time': best_params["growth_time"],
            'RatioDw_Dg': 10,
            'gm': GM,
            'wm': WM,
            'segm': init_seg,
            'NxT1_pct': WM.shape[0]/max_dim,
            'NyT1_pct': WM.shape[1]/max_dim, 
            'NzT1_pct': WM.shape[2]/max_dim,
            'resolution_factor': 0.7  # Higher quality for final visualization
        }
        
        # Run forward simulation
        solver = fwdSolver(params)
        result = solver.solve()
        
        if result.get('success', False):
            predicted_tumor = result["final_state"]
            
            # Save predicted tumor as NIfTI
            nib_img = nib.Nifti1Image(predicted_tumor.astype(np.float32), np.eye(4))
            nib.save(nib_img, os.path.join(args.output_dir, "predicted_tumor.nii.gz"))
            
            # Create visualizations
            visualize_results(WM, GM, tumor, predicted_tumor, os.path.join(args.output_dir, "visualizations"))
        else:
            print(f"Final visualization failed: {result.get('error', 'Unknown error')}")
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()