#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import glob
import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from TumorGrowthToolkit.FK import Solver as FKSolver
import cmaes
import multiprocessing

def dice(a, b):
    """Calculate Dice coefficient between two binary masks."""
    boolA, boolB = a > 0, b > 0 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0
    return 2 * np.sum(np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

def load_data(data_dir, patient_id, time_point):
    """Load MRI data for a specific time point."""
    patient_dir = os.path.join(data_dir, patient_id)
    
    # Format the time point with leading zeros (e.g., 49 -> 049)
    time_point_str = f"{time_point:03d}"
    
    # Look for directories with the exact week number
    week_dirs = glob.glob(os.path.join(patient_dir, f'week-{time_point_str}'))
    
    # If not found, try with additional suffixes
    if not week_dirs:
        week_dirs = glob.glob(os.path.join(patient_dir, f'week-{time_point_str}-*'))
    
    if not week_dirs:
        # As a fallback, try without padding
        week_dirs = glob.glob(os.path.join(patient_dir, f'week-{time_point}*'))
        
    if not week_dirs:
        raise FileNotFoundError(f"No data found for week {time_point} (looked for week-{time_point_str})")
    
    # Use the first matching directory
    week_dir = week_dirs[0]
    print(f"Found data directory: {week_dir}")
    
    # Load white matter, gray matter, and tumor segmentation
    wm_path = os.path.join(week_dir, 'T1_pve_2.nii.gz')
    gm_path = os.path.join(week_dir, 'T1_pve_1.nii.gz')
    tumor_path = os.path.join(week_dir, 'seg_mask.nii.gz')
    
    # Check if files exist
    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"White matter file not found: {wm_path}")
    if not os.path.exists(gm_path):
        raise FileNotFoundError(f"Gray matter file not found: {gm_path}")
    if not os.path.exists(tumor_path):
        raise FileNotFoundError(f"Tumor segmentation file not found: {tumor_path}")
    
    # Load data
    wm_nifti = nib.load(wm_path)
    gm_nifti = nib.load(gm_path)
    tumor_nifti = nib.load(tumor_path)
    
    wm_data = wm_nifti.get_fdata()
    gm_data = gm_nifti.get_fdata()
    tumor_data = tumor_nifti.get_fdata()
    
    # Get affine transformation matrix
    affine = wm_nifti.affine
    
    return wm_data, gm_data, tumor_data, affine

def prepare_tumor_regions(tumor_data):
    """Split tumor into regions (edema, enhancing, necrotic)."""
    # Try standard segmentation labels (1=necrotic, 2=edema, 4=enhancing)
    edema = np.logical_or(tumor_data == 2, tumor_data == 3)
    necrotic = tumor_data == 1
    enhancing = tumor_data == 4
    
    # If standard labels aren't found, use simple binary approach
    if not (np.any(edema) or np.any(necrotic) or np.any(enhancing)):
        print("Warning: No tumor regions found with standard labels. Using binary approach.")
        edema = tumor_data > 0
        necrotic = np.zeros_like(tumor_data, dtype=bool)
        enhancing = np.zeros_like(tumor_data, dtype=bool)
    
    return edema, necrotic, enhancing

def initialize_settings(edema, gm_shape):
    """Initialize settings for the FK equation fitting."""
    # Calculate tumor center of mass
    com = ndimage.center_of_mass(edema)
    
    # Initial parameters
    settings = {
        "rho0": 0.06,           # Initial proliferation rate
        "dw0": 0.001,             # Initial diffusion coefficient
        "model_time": 20.0,     # Fixed model time parameter (not optimized)
        "thresholdT1c": 0.675,  # T1c threshold
        "thresholdFlair": 0.25, # FLAIR threshold
        "NxT1_pct0": float(com[0] / gm_shape[0]),
        "NyT1_pct0": float(com[1] / gm_shape[1]),
        "NzT1_pct0": float(com[2] / gm_shape[2])
    }
    
    # Parameter ranges for optimization - excluding model_time
    settings["parameterRanges"] = [
        [0, 1],              # NxT1_pct
        [0, 1],              # NyT1_pct
        [0, 1],              # NzT1_pct
        [0.001, 3],          # Dw
        [0.001, 0.225],     # rho
        [0.5, 0.85],          # thresholdT1c
        [0.001, 0.5]         # thresholdFlair
    ]
    
    # Other settings
    settings["sigma0"] = 0.1   # Increased sigma for better exploration
    settings["resolution_factor"] = {0: 0.3, 0.7: 0.5}
    
    return settings

def load_fk_parameters(output_dir, fit_time):
    """
    Load previously fitted FK parameters from output directory.
    
    Args:
        output_dir: Directory containing saved results
        fit_time: Time point used for fitting
        
    Returns:
        Dictionary of FK parameters
    """
    print(f"Loading FK parameters for time point {fit_time}...")
    
    # Check for saved parameters in the fk directory
    fk_dir = os.path.join(output_dir, 'fk')
    if not os.path.exists(fk_dir):
        raise FileNotFoundError(f"FK directory not found: {fk_dir}")
    
    # Look for parameters file
    params_file = os.path.join(fk_dir, f'fk_params_t{fit_time}.npy')
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"FK parameters file not found: {params_file}")
    
    # Load parameters
    try:
        params = np.load(params_file, allow_pickle=True).item()
        
        # Verify essential parameters exist
        required_params = ['Dw', 'rho', 'model_time', 'NxT1_pct', 'NyT1_pct', 'NzT1_pct', 
                          'thresholdT1c', 'thresholdFlair']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters in loaded file: {', '.join(missing_params)}")
        
        print(f"Successfully loaded FK parameters: Dw={params['Dw']:.6f}, rho={params['rho']:.6f}")
        return params
        
    except Exception as e:
        raise ValueError(f"Error loading FK parameters: {str(e)}")

def load_time_scale(output_dir, scale_time):
    """
    Load previously fitted time scale parameter from output directory.
    
    Args:
        output_dir: Directory containing saved results
        scale_time: Time point used for fitting time scale
        
    Returns:
        Time scale value (float)
    """
    print(f"Loading time scale parameter for time point {scale_time}...")
    
    # Check for saved parameters in the scale directory
    scale_dir = os.path.join(output_dir, 'scale')
    if not os.path.exists(scale_dir):
        raise FileNotFoundError(f"Scale directory not found: {scale_dir}")
    
    # Look for parameters file
    params_file = os.path.join(scale_dir, f'scale_params_t{scale_time}.npy')
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Scale parameters file not found: {params_file}")
    
    # Load parameters
    try:
        params = np.load(params_file, allow_pickle=True).item()
        
        # Extract time scale parameter
        if 'time_scale' not in params:
            raise ValueError("Time scale parameter not found in loaded file")
        
        time_scale = params['time_scale']
        print(f"Successfully loaded time scale: {time_scale:.6f}")
        return time_scale
        
    except Exception as e:
        raise ValueError(f"Error loading time scale: {str(e)}")

def save_results(output_dir, phase, time_point, results, params=None):
    """Save optimization results and parameters."""
    # Create the output directory if it doesn't exist
    phase_dir = os.path.join(output_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)
    
    # Save the full results
    result_path = os.path.join(phase_dir, f'{phase}_results_t{time_point}.npy')
    np.save(result_path, results)
    
    # If parameters are provided, save them separately for easier access
    if params is not None:
        params_path = os.path.join(phase_dir, f'{phase}_params_t{time_point}.npy')
        np.save(params_path, params)
        
        # Also save as text for human readability
        txt_path = os.path.join(phase_dir, f'{phase}_params_t{time_point}.txt')
        with open(txt_path, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    return phase_dir

def visualize_results(output_dir, phase, time_point, wm_data, tumor_data, predicted_tumor, 
                      params=None, slice_index=None):
    """Create visualizations of the results."""
    # Create the output directory if it doesn't exist
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # If slice_index is not specified, find the slice with the most tumor
    if slice_index is None:
        tumor_sum = np.sum(tumor_data, axis=(0, 1))
        slice_index = np.argmax(tumor_sum)
    
    # Create figure comparing actual and predicted tumor
    plt.figure(figsize=(15, 5))
    
    # Plot actual tumor
    plt.subplot(1, 3, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap='gray', alpha=0.8)
    plt.imshow(tumor_data[:, :, slice_index], cmap='hot', alpha=0.7)
    plt.title(f'Actual Tumor (t={time_point})')
    plt.axis('off')
    
    # Plot predicted tumor
    plt.subplot(1, 3, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap='gray', alpha=0.8)
    plt.imshow(predicted_tumor[:, :, slice_index], cmap='hot', alpha=0.7)
    plt.title(f'Predicted Tumor (t={time_point})')
    plt.axis('off')
    
    # Plot difference
    plt.subplot(1, 3, 3)
    tumor_binary = tumor_data[:, :, slice_index] > 0
    pred_binary = predicted_tumor[:, :, slice_index] > 0.5
    
    # Create a difference map: green = true positive, red = false positive, blue = false negative
    diff_map = np.zeros((*tumor_binary.shape, 3))
    diff_map[np.logical_and(tumor_binary, pred_binary), 1] = 1.0  # True positive (green)
    diff_map[np.logical_and(~tumor_binary, pred_binary), 0] = 1.0  # False positive (red)
    diff_map[np.logical_and(tumor_binary, ~pred_binary), 2] = 1.0  # False negative (blue)
    
    plt.imshow(wm_data[:, :, slice_index], cmap='gray', alpha=0.5)
    plt.imshow(diff_map, alpha=0.7)
    plt.title('Difference Map')
    plt.axis('off')
    
    # Calculate Dice coefficient
    dice_coef = dice(tumor_binary, pred_binary)
    plt.suptitle(f'Phase: {phase}, Time: {time_point}, Dice: {dice_coef:.4f}', fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    fig_path = os.path.join(viz_dir, f'{phase}_tumor_comparison_t{time_point}.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return viz_dir

class FK_CMAESSolver:
    """FK equation parameter solver using CMA-ES optimization."""
    
    def __init__(self, settings, wm, gm, edema, enhancing, necrotic, segm, workers=8):
        self.settings = settings
        self.settings["workers"] = workers
        self.wm = wm
        self.gm = gm
        self.edema = edema
        self.enhancing = enhancing
        self.necrotic = necrotic
        self.segm = segm
    
    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):
        """Calculate loss between predicted and actual tumor."""
        lambdaFlair = 0.333
        lambdaT1c = 0.333
        
        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c)
        lossFlair = 1 - dice(proposedEdema, self.edema)
        lossT1c = 1 - dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c
        
        # Handle invalid loss values
        if not (0 <= loss <= 1):
            loss = 1
            
        return loss, {"lossFlair": lossFlair, "lossT1c": lossT1c, "lossTotal": loss}
    
    def forward(self, x, resolution_factor=1.0):
        """Run forward model with given parameters."""
        parameters = {
            'Dw': x[3],         # Diffusion coefficient for white matter
            'rho': x[4],        # Proliferation rate
            'stopping_time': self.settings["model_time"],  # Fixed model time parameter
            'RatioDw_Dg': 10,   # Ratio of diffusion coefficients
            'gm': self.gm,      # Grey matter data
            'wm': self.wm,      # White matter data
            'NxT1_pct': x[0],   # Initial focal position (in percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor': resolution_factor,
            'segm': self.segm
        }
        
        print(f"Running forward model with parameters: Dw={x[3]:.4f}, rho={x[4]:.6f}, time={self.settings['model_time']:.1f}")
        
        solver = FKSolver(parameters)
        result = solver.solve()
        
        if not result.get('success', False):
            print(f"Forward model failed: {result.get('error', 'Unknown error')}")
            return None
            
        return result["final_state"]
    
    def getLoss(self, x, gen):
        """Calculate loss for a parameter set."""
        start_time = time.time()
        
        # Handle adaptive resolution
        if isinstance(self.settings["resolution_factor"], dict):
            resolution_factor = None
            for relativeGen, resFactor in self.settings["resolution_factor"].items():
                if gen / self.settings["generations"] >= relativeGen:
                    resolution_factor = resFactor
        else:
            resolution_factor = self.settings["resolution_factor"]
        
        # Run forward model
        tumor = self.forward(x, resolution_factor)
        
        if tumor is None:
            # Handle failed simulation
            loss = 1.0
            lossDir = {
                "lossFlair": 1.0,
                "lossT1c": 1.0, 
                "lossTotal": 1.0,
                "simulation_failed": True,
                "allParams": x.copy()
            }
        else:
            # Calculate loss
            thresholdT1c = x[-2]    
            thresholdFlair = x[-1]
            loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
            lossDir["allParams"] = x.copy()
        
        # Record execution details
        end_time = time.time()
        lossDir["time"] = end_time - start_time
        lossDir["resolution_factor"] = resolution_factor
        lossDir["generation"] = gen
        
        print(f"Generation {gen}, Loss: {loss:.6f}, Time: {end_time - start_time:.2f}s")
        
        return loss, lossDir
    
    def run(self, generations):
        """Run the optimization process."""
        self.settings["generations"] = generations
        start_time = time.time()
        
        # Initial values WITHOUT model_time parameter (fixed at 30.0)
        initValues = (
            self.settings["NxT1_pct0"], 
            self.settings["NyT1_pct0"], 
            self.settings["NzT1_pct0"], 
            self.settings["dw0"], 
            self.settings["rho0"], 
            self.settings["thresholdT1c"], 
            self.settings["thresholdFlair"]
        )
        
        # Run CMA-ES optimization with trace=True to record all evaluations
        print(f"Starting CMA-ES optimization with {generations} generations...")
        try:
            trace = cmaes.cmaes(
                self.getLoss, 
                initValues, 
                self.settings["sigma0"], 
                generations, 
                workers=self.settings["workers"], 
                trace=True, 
                parameterRange=self.settings["parameterRanges"]
            )
            
            # Extract evaluation information from trace
            _, _, _, _, _, _, _, _, _, _, lossDir = zip(*trace)
            
            # Find the best parameters from all evaluations
            min_loss = float('inf')
            opt_params = None
            
            # Process all generations and find the best parameters
            for gen_lossDir in lossDir:
                for eval_loss in gen_lossDir:
                    # Make sure lossTotal is a valid number
                    if "lossTotal" in eval_loss and isinstance(eval_loss["lossTotal"], (int, float)) and 0 <= eval_loss["lossTotal"] <= 1:
                        if eval_loss["lossTotal"] < min_loss:
                            min_loss = eval_loss["lossTotal"]
                            if "allParams" in eval_loss:
                                opt_params = eval_loss["allParams"]
            
            print(f"Optimization completed. Best loss: {min_loss:.6f}")
            print(f"Best parameters: {[round(p, 6) for p in opt_params]}")
            
            # Run forward model with best parameters to get the final tumor prediction
            tumor_prediction = self.forward(opt_params)
            
            end_time = time.time()
            
            # Prepare result dictionary
            result_dict = {
                "minLoss": min_loss,
                "opt_params": opt_params,
                "time_min": (end_time - start_time) / 60,
                "trace": trace,
                "lossDir": lossDir,
                "model_time": self.settings["model_time"]  # Include fixed model time
            }
            
            return tumor_prediction, result_dict
            
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

class AdaptiveTimeScaleSolver:
    """Improved solver for time scale parameter with adaptive search strategy."""
    
    def __init__(self, fk_params, reference_wm, reference_gm, reference_tumor, 
                 target_tumor, physical_time_diff, workers=8):
        self.fk_params = fk_params
        self.reference_wm = reference_wm
        self.reference_gm = reference_gm
        self.reference_tumor = reference_tumor
        self.target_tumor = target_tumor
        self.physical_time_diff = physical_time_diff
        self.workers = workers
    
    def forward(self, time_scale):
        """Run forward model with fixed FK parameters and time scale."""
        # Calculate model time using time scale
        model_time = self.fk_params['model_time'] + (self.physical_time_diff * time_scale)
        
        parameters = {
            'Dw': self.fk_params['Dw'],
            'rho': self.fk_params['rho'],
            'stopping_time': model_time,
            'RatioDw_Dg': 10,
            'gm': self.reference_gm,
            'wm': self.reference_wm,
            'NxT1_pct': self.fk_params['NxT1_pct'],
            'NyT1_pct': self.fk_params['NyT1_pct'],
            'NzT1_pct': self.fk_params['NzT1_pct'],
            'resolution_factor': 0.5,
            'segm': self.reference_tumor
        }
        
        print(f"Running with time scale: {time_scale:.4f}, model time: {model_time:.1f}")
        
        solver = FKSolver(parameters)
        result = solver.solve()
        
        if not result.get('success', False):
            print(f"Forward model failed: {result.get('error', 'Unknown error')}")
            return None
            
        return result["final_state"]
    
    def calculate_loss(self, predicted, actual):
        """Calculate loss between predicted and actual tumor."""
        if predicted is None:
            return 1.0
        
        # Use Dice coefficient as loss (1 - dice)
        dice_coef = dice(predicted > 0.5, actual > 0)
        return 1 - dice_coef
    
    def getLoss(self, x, gen):
        """Calculate loss for a time scale parameter."""
        start_time = time.time()
        time_scale = float(x[0])  # Ensure time_scale is a float
        
        # Run forward model
        predicted_tumor = self.forward(time_scale)
        
        # Calculate loss
        loss = self.calculate_loss(predicted_tumor, self.target_tumor)
        
        # Ensure loss is valid
        if not (0 <= loss <= 1):
            loss = 1.0
        
        # Record execution details
        end_time = time.time()
        lossDir = {
            "time": end_time - start_time,
            "time_scale": time_scale,
            "generation": gen,
            "model_time": self.fk_params['model_time'] + (self.physical_time_diff * time_scale),
            "physical_time_diff": self.physical_time_diff,
            "loss": loss,
            "allParams": [time_scale]
        }
        
        print(f"Generation {gen}, Scale: {time_scale:.4f}, Loss: {loss:.6f}, "
              f"Time: {end_time - start_time:.2f}s")
        
        return loss, lossDir
    
    def run(self, generations, param_range=[1, 10.0]):
        """Run the time scale optimization with decreased learning rate over time."""
        start_time = time.time()
        init_scale = (param_range[0] + param_range[1]) / 2
        all_traces = []
        search_history = []
        
        # Two-stage optimization with decreasing learning rate
        # Stage 1: Broad search with high sigma
        print(f"Stage 1: Broad search with range {param_range}")
        stage1_gens = max(1, generations // 2)  # Ensure at least 1 generation
        sigma1 = (param_range[1] - param_range[0]) / 4  # Higher sigma for exploration
        
        trace1 = cmaes.cmaes(
            self.getLoss, 
            [init_scale], 
            sigma1,
            stage1_gens, 
            workers=self.workers, 
            trace=True, 
            parameterRange=[param_range]
        )
        
        all_traces.extend(trace1)
        
        # Extract the best time scale from first stage
        min_loss = float('inf')
        best_scale = init_scale
        
        for trace_entry in trace1:
            best_loss_gen = trace_entry[1]  # Best loss for this generation
            best_params_gen = trace_entry[2]  # Best parameters for this generation
            
            # Try to convert to float if not already
            try:
                best_loss_gen = float(best_loss_gen)
                
                # Check if it's a valid loss
                if 0 <= best_loss_gen <= 1 and best_loss_gen < min_loss:
                    min_loss = best_loss_gen
                    best_scale = float(best_params_gen[0])  # Ensure it's a float
            except (TypeError, ValueError):
                continue
            
            # Also collect search history from lossDir
            lossDir_list = trace_entry[10]  # lossDir is at index 10
            for lossDir in lossDir_list:
                if "time_scale" in lossDir and "loss" in lossDir:
                    try:
                        time_scale_val = float(lossDir["time_scale"])
                        loss_val = float(lossDir["loss"])
                        if 0 <= loss_val <= 1:
                            search_history.append((time_scale_val, loss_val))
                    except (TypeError, ValueError):
                        continue
        
        print(f"Stage 1 completed. Best loss: {min_loss:.6f}, Best scale: {best_scale:.6f}")
        
        # Stage 2: Fine-tuning around the best scale from stage 1
        # Create a narrower range around the best scale
        range_width = param_range[1] - param_range[0]
        stage2_range = [
            max(param_range[0], best_scale - range_width/4),
            min(param_range[1], best_scale + range_width/4)
        ]
        
        print(f"Stage 2: Fine-tuning with range {stage2_range}")
        stage2_gens = max(1, generations - stage1_gens)  # Ensure at least 1 generation
        sigma2 = (stage2_range[1] - stage2_range[0]) / 6  # Lower sigma for fine-tuning
        
        trace2 = cmaes.cmaes(
            self.getLoss, 
            [best_scale],  # Start from previous best 
            sigma2,
            stage2_gens, 
            workers=self.workers, 
            trace=True, 
            parameterRange=[stage2_range]
        )
        
        all_traces.extend(trace2)
        
        # Update best parameters considering stage 2
        for trace_entry in trace2:
            best_loss_gen = trace_entry[1]  # Best loss for this generation
            best_params_gen = trace_entry[2]  # Best parameters for this generation
            
            # Try to convert to float if not already
            try:
                best_loss_gen = float(best_loss_gen)
                
                # Check if it's a valid loss
                if 0 <= best_loss_gen <= 1 and best_loss_gen < min_loss:
                    min_loss = best_loss_gen
                    best_scale = float(best_params_gen[0])  # Ensure it's a float
            except (TypeError, ValueError):
                continue
            
            # Also collect search history from lossDir
            lossDir_list = trace_entry[10]  # lossDir is at index 10
            for lossDir in lossDir_list:
                if "time_scale" in lossDir and "loss" in lossDir:
                    try:
                        time_scale_val = float(lossDir["time_scale"])
                        loss_val = float(lossDir["loss"])
                        if 0 <= loss_val <= 1:
                            search_history.append((time_scale_val, loss_val))
                    except (TypeError, ValueError):
                        continue
        
        print(f"Stage 2 completed. Best loss: {min_loss:.6f}, Best scale: {best_scale:.6f}")
        
        # Get the final prediction
        final_prediction = self.forward(best_scale)
        
        end_time = time.time()
        
        # Prepare result dictionary
        result_dict = {
            "min_loss": min_loss,
            "best_scale": best_scale,
            "physical_time_diff": self.physical_time_diff,
            "time_min": (end_time - start_time) / 60,
            "trace": all_traces,
            "search_history": search_history
        }
        
        return final_prediction, best_scale, result_dict

def predict_tumor(fk_params, time_scale, reference_wm, reference_gm, reference_tumor, 
                 reference_time, target_time):
    """Predict tumor at target time using fitted parameters."""
    # Calculate time difference
    time_diff = target_time - reference_time
    
    # Calculate model time using time scale
    model_time = fk_params['model_time'] + (time_diff * time_scale)
    
    print(f"Predicting tumor at time {target_time} (reference: {reference_time})")
    print(f"Physical time difference: {time_diff}")
    print(f"Time scale: {time_scale:.4f}")
    print(f"Model time: {model_time:.1f}")
    
    # Setup parameters for forward model
    parameters = {
        'Dw': fk_params['Dw'],
        'rho': fk_params['rho'],
        'stopping_time': model_time,
        'RatioDw_Dg': 10,
        'gm': reference_gm,
        'wm': reference_wm,
        'NxT1_pct': fk_params['NxT1_pct'],
        'NyT1_pct': fk_params['NyT1_pct'],
        'NzT1_pct': fk_params['NzT1_pct'],
        'resolution_factor': 0.7,
        'segm': reference_tumor,
        'verbose': True
    }
    
    # Run forward model
    solver = FKSolver(parameters)
    result = solver.solve()
    
    if not result.get('success', False):
        print(f"Prediction failed: {result.get('error', 'Unknown error')}")
        return None
    
    return result['final_state']

def generate_time_series_predictions(fk_params, time_scale, reference_wm, reference_gm, reference_tumor, 
                                   reference_time, start_time, end_time, time_step=1):
    """
    Generate a series of tumor predictions over a range of time points.
    
    Args:
        fk_params: Dictionary of FK parameters
        time_scale: Time scale parameter
        reference_wm: Reference white matter data
        reference_gm: Reference gray matter data
        reference_tumor: Reference tumor data
        reference_time: Reference time point
        start_time: Starting time point for series
        end_time: Ending time point for series
        time_step: Step size between time points
        
    Returns:
        Dictionary mapping time points to predicted tumors
    """
    time_points = range(start_time, end_time + 1, time_step)
    predictions = {}
    
    print(f"Generating tumor predictions for time points {start_time} to {end_time} (step={time_step})...")
    
    for time_point in time_points:
        print(f"Predicting tumor at time {time_point}...")
        
        # Skip prediction at reference time - use actual tumor
        if time_point == reference_time:
            predictions[time_point] = reference_tumor
            continue
            
        # Predict tumor at this time point
        pred_tumor = predict_tumor(
            fk_params=fk_params,
            time_scale=time_scale,
            reference_wm=reference_wm,
            reference_gm=reference_gm,
            reference_tumor=reference_tumor,
            reference_time=reference_time,
            target_time=time_point
        )
        
        if pred_tumor is not None:
            predictions[time_point] = pred_tumor
        else:
            print(f"Warning: Failed to predict tumor at time {time_point}, skipping.")
    
    return predictions

def save_time_series(output_dir, patient_id, time_series_predictions, affine):
    """
    Save time series predictions as NIfTI files.
    
    Args:
        output_dir: Base output directory
        patient_id: Patient ID
        time_series_predictions: Dictionary mapping time points to predicted tumors
        affine: Affine transformation matrix from original data
        
    Returns:
        Directory where time series files were saved
    """
    time_series_dir = os.path.join(output_dir, 'time_series')
    os.makedirs(time_series_dir, exist_ok=True)
    
    print(f"Saving time series predictions to {time_series_dir}...")
    
    saved_files = []
    for time_point, tumor in time_series_predictions.items():
        # Format time point with leading zeros
        time_point_str = f"{time_point:03d}"
        
        # Create output filename
        output_file = os.path.join(time_series_dir, f"tumor_week_{time_point_str}.nii.gz")
        
        # Create NIfTI image
        tumor_nii = nib.Nifti1Image(tumor, affine)
        
        # Save file
        nib.save(tumor_nii, output_file)
        saved_files.append(output_file)
        
        print(f"Saved prediction for week {time_point} to {output_file}")
    
    return time_series_dir, saved_files

def visualize_training_progression(output_dir, phase, losses, generations):
    """
    Visualize the training loss progression over generations.
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Ensure losses are numeric values
    numeric_losses = []
    for loss in losses:
        # Handle if loss is a complex object or list
        if isinstance(loss, (list, tuple, np.ndarray)):
            # Try to get the first element if it's a collection
            try:
                numeric_losses.append(float(loss[0]))
            except (IndexError, TypeError, ValueError):
                continue
        else:
            try:
                numeric_losses.append(float(loss))
            except (TypeError, ValueError):
                continue
    
    # Skip visualization if no valid losses
    if not numeric_losses:
        print(f"Warning: No valid loss values found for {phase} visualization")
        return None
    
    # Create figure for loss progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(numeric_losses) + 1), numeric_losses, 'b-', linewidth=2)
    plt.plot(range(1, len(numeric_losses) + 1), numeric_losses, 'ro', markersize=4)
    
    # Add best loss marker
    best_gen = np.argmin(numeric_losses) + 1
    best_loss = min(numeric_losses)
    plt.plot(best_gen, best_loss, 'go', markersize=10, label=f'Best: Gen {best_gen}, Loss {best_loss:.4f}')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{phase.upper()} Optimization Progress', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Improve visual appearance
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(viz_dir, f'{phase}_loss_progression.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return fig_path

def visualize_comparison_across_timepoints(output_dir, patient_id, timepoints, dice_scores, tumor_volumes=None):
    """
    Create visualization comparing performance across different time points.
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot Dice scores
    plt.subplot(2, 1, 1)
    plt.bar(range(len(timepoints)), dice_scores, color='skyblue')
    plt.xticks(range(len(timepoints)), [f'Week {t}' for t in timepoints])
    plt.ylabel('Dice Coefficient', fontsize=12)
    plt.title(f'Prediction Accuracy Across Time Points - {patient_id}', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(dice_scores):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    
    # Plot tumor volumes if provided
    if tumor_volumes is not None:
        plt.subplot(2, 1, 2)
        plt.plot(timepoints, tumor_volumes, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Tumor Volume (mm³)', fontsize=12)
        plt.title('Tumor Volume Progression', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(viz_dir, f'{patient_id}_timepoint_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return fig_path

def visualize_parameter_evolution(output_dir, phase, param_history, param_names):
    """
    Visualize how parameters evolved during optimization.
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a plot for each parameter
    plt.figure(figsize=(15, 10))
    
    # Determine number of parameters and setup subplot grid
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    for i, (param_name, history) in enumerate(zip(param_names, param_history)):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot parameter value over generations
        plt.plot(range(1, len(history) + 1), history, 'b-', linewidth=1.5)
        plt.plot(range(1, len(history) + 1), history, 'ro', markersize=3)
        
        # Add final value marker
        final_value = history[-1]
        plt.axhline(y=final_value, color='g', linestyle='--', alpha=0.7)
        plt.text(len(history) * 0.5, final_value * 1.05, f'Final: {final_value:.4f}', 
                 ha='center', color='green')
        
        plt.xlabel('Generation', fontsize=10)
        plt.ylabel(param_name, fontsize=10)
        plt.title(f'Evolution of {param_name}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(viz_dir, f'{phase}_parameter_evolution.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return fig_path

def find_tumor_slices(tumor_data, n_slices=3):
    """
    Find the most representative tumor slices.
    """
    # Calculate tumor presence in each slice
    z_sums = np.sum(tumor_data > 0, axis=(0, 1))
    
    # If no tumor found, return empty list
    if np.max(z_sums) == 0:
        return []
    
    # Find slices with tumor
    tumor_slices = np.where(z_sums > 0)[0]
    
    if len(tumor_slices) <= n_slices:
        return tumor_slices.tolist()
    
    # If more than n_slices have tumor, select most representative ones
    selected_indices = np.linspace(0, len(tumor_slices)-1, n_slices).astype(int)
    return tumor_slices[selected_indices].tolist()

def visualize_3d_tumor(output_dir, phase, time_point, wm_data, actual_tumor, predicted_tumor=None):
    """
    Create 3D visualization of tumor data.
    """
    viz_dir = os.path.join(output_dir, 'visualizations', '3d')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a multi-slice view of the tumor
    z_indices = find_tumor_slices(actual_tumor)
    
    # If no indices found, use middle slices
    if not z_indices:
        z_indices = [actual_tumor.shape[2] // 4, actual_tumor.shape[2] // 2, 3 * actual_tumor.shape[2] // 4]
    
    plt.figure(figsize=(15, 5 * len(z_indices)))
    
    for i, z in enumerate(z_indices):
        # Original MRI with actual tumor
        plt.subplot(len(z_indices), 3 if predicted_tumor is not None else 1, i*3 + 1 if predicted_tumor is not None else i+1)
        plt.imshow(wm_data[:, :, z], cmap='gray', alpha=0.8)
        plt.imshow(actual_tumor[:, :, z], cmap='hot', alpha=0.7)
        if i == 0:
            plt.title(f'Actual Tumor (Week {time_point})', fontsize=12)
        plt.axis('off')
        plt.text(10, 20, f'Slice {z}', color='white', fontsize=10, backgroundcolor='black')
        
        # If predicted tumor is provided
        if predicted_tumor is not None:
            # MRI with predicted tumor
            plt.subplot(len(z_indices), 3, i*3 + 2)
            plt.imshow(wm_data[:, :, z], cmap='gray', alpha=0.8)
            plt.imshow(predicted_tumor[:, :, z], cmap='hot', alpha=0.7)
            if i == 0:
                plt.title(f'Predicted Tumor', fontsize=12)
            plt.axis('off')
            
            # Difference map
            plt.subplot(len(z_indices), 3, i*3 + 3)
            tumor_binary = actual_tumor[:, :, z] > 0
            pred_binary = predicted_tumor[:, :, z] > 0.5
            
            # Create difference map (green=TP, red=FP, blue=FN)
            diff_map = np.zeros((*tumor_binary.shape, 3))
            diff_map[np.logical_and(tumor_binary, pred_binary), 1] = 1.0  # TP (green)
            diff_map[np.logical_and(~tumor_binary, pred_binary), 0] = 1.0  # FP (red)
            diff_map[np.logical_and(tumor_binary, ~pred_binary), 2] = 1.0  # FN (blue)
            
            plt.imshow(wm_data[:, :, z], cmap='gray', alpha=0.5)
            plt.imshow(diff_map, alpha=0.7)
            if i == 0:
                plt.title('Difference Map', fontsize=12)
            plt.axis('off')
            
            # Calculate slice-specific Dice
            slice_dice = dice(tumor_binary, pred_binary)
            plt.text(10, 20, f'Dice: {slice_dice:.4f}', color='white', fontsize=10, backgroundcolor='black')
    
    plt.tight_layout()
    
    # Save the figure
    if predicted_tumor is not None:
        fig_path = os.path.join(viz_dir, f'{phase}_3d_comparison_t{time_point}.png')
    else:
        fig_path = os.path.join(viz_dir, f'{phase}_3d_tumor_t{time_point}.png')
    
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return fig_path

def calculate_tumor_volume(tumor_data, voxel_dims=(1, 1, 1)):
    """
    Calculate tumor volume in mm³.
    """
    # Count voxels where tumor is present
    tumor_voxels = np.sum(tumor_data > 0)
    
    # Calculate volume based on voxel dimensions
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    tumor_volume = tumor_voxels * voxel_volume
    
    return tumor_volume

def dice_coefficient(a, b):
    """
    Calculate the Dice coefficient between two binary arrays.
    """
    if not np.any(a) and not np.any(b):
        return 1.0
    
    intersection = np.sum(np.logical_and(a, b))
    dice = 2.0 * intersection / (np.sum(a) + np.sum(b))
    return dice

def visualize_time_series(output_dir, patient_id, wm_data, time_series_predictions, 
                         fit_time=None, scale_time=None, max_frames=8):
    """
    Create a visualization of tumor progression over time.
    
    Args:
        output_dir: Directory to save visualizations
        patient_id: Patient ID
        wm_data: White matter data for background
        time_series_predictions: Dictionary mapping time points to predicted tumors
        fit_time: Time point used for fitting FK parameters (optional)
        scale_time: Time point used for fitting time scale (optional)
        max_frames: Maximum number of frames to show in the visualization
        
    Returns:
        Path to saved visualization
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get sorted time points
    time_points = sorted(time_series_predictions.keys())
    tumors = [time_series_predictions[t] for t in time_points]
    
    # Find a representative slice across all timepoints
    combined_tumor = np.zeros_like(tumors[0])
    for tumor in tumors:
        combined_tumor = np.logical_or(combined_tumor, tumor > 0.5)
        
    # Find the slice with maximum tumor coverage
    slice_sums = np.sum(combined_tumor, axis=(0, 1))
    if np.max(slice_sums) == 0:
        # If no tumor found, use middle slice
        slice_index = combined_tumor.shape[2] // 2
    else:
        slice_index = np.argmax(slice_sums)
    
    # Sample time points if there are too many
    if len(time_points) > max_frames:
        # Sample approximately max_frames evenly spaced timepoints
        indices = np.linspace(0, len(time_points)-1, max_frames, dtype=int)
        selected_time_points = [time_points[i] for i in indices]
        selected_tumors = [tumors[i] for i in indices]
    else:
        selected_time_points = time_points
        selected_tumors = tumors
    
    # Create a row of images showing tumor evolution
    fig_width = min(20, 2.5 * len(selected_time_points))  # Limit width to 20 inches
    plt.figure(figsize=(fig_width, 5))
    
    for i, (time_point, tumor) in enumerate(zip(selected_time_points, selected_tumors)):
        plt.subplot(1, len(selected_time_points), i+1)
        
        # Plot white matter background with tumor overlay
        plt.imshow(wm_data[:, :, slice_index], cmap='gray', alpha=0.8)
        
        # Create a color mask for different tumor regions
        # Red for enhancing (T1c)
        # Green for edema (FLAIR)
        tumor_slice = tumor[:, :, slice_index]
        
        # In all visualization functions, use:
        edema_mask = np.logical_and(tumor_slice > fk_params['thresholdFlair'], 
                                    tumor_slice < fk_params['thresholdT1c'])
        enhancing_mask = tumor_slice >= fk_params['thresholdT1c']

        # Create RGB overlay for different tumor regions
        overlay = np.zeros((*wm_data[:, :, slice_index].shape, 3))
        
        # Green for edema
        overlay[edema_mask, 1] = 0.8
        
        # Red for enhancing
        overlay[enhancing_mask, 0] = 0.8
        
        plt.imshow(overlay, alpha=0.7)
        
        # Mark if this is a fitting timepoint
        title = f'Week {time_point}'
        if fit_time is not None and time_point == fit_time:
            title += '\n(FK Fit)'
        elif scale_time is not None and time_point == scale_time:
            title += '\n(Scale Fit)'
            
        plt.title(title)
        plt.axis('off')
    
    plt.suptitle(f'{patient_id} Tumor Progression (Slice {slice_index})', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(viz_dir, f'{patient_id}_time_series.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series visualization saved to {fig_path}")
    return fig_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tumor Growth Prediction Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing patient data')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient ID (e.g., Patient-042)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--fit_time', type=int, required=True,
                        help='Time point (week) to fit FK equation parameters')
    parser.add_argument('--scale_time', type=int, required=True,
                        help='Time point (week) to fit time scale parameter')
    parser.add_argument('--test_time', type=int, nargs='+', required=True,
                        help='Time point(s) (week) to test prediction (can specify multiple)')
    parser.add_argument('--fk_generations', type=int, default=15,
                        help='Number of generations for FK parameter optimization')
    parser.add_argument('--scale_generations', type=int, default=10,
                        help='Number of generations for time scale optimization')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker processes for parallel computation')
    parser.add_argument('--skip_fk', action='store_true',
                        help='Skip FK parameter optimization and load parameters from output directory')
    parser.add_argument('--skip_scale', action='store_true',
                        help='Skip time scale optimization and load parameters from output directory')
    parser.add_argument('--generate_time_series', action='store_true',
                        help='Generate and visualize a time series of tumor progression')
    
    # New arguments
    parser.add_argument('--manual_scale', type=float, 
                        help='Manually specify time scale parameter (overrides --skip_scale)')
    parser.add_argument('--viz_only', action='store_true',
                        help='Visualization only mode: skips optimization and generates visualizations')
    parser.add_argument('--time_series_only', action='store_true',
                        help='Time series generation only mode: skips optimization and testing phases')
    parser.add_argument('--start_time', type=int, default=None,
                        help='Starting time point for time series generation')
    parser.add_argument('--end_time', type=int, default=None,
                        help='Ending time point for time series generation')
    parser.add_argument('--time_step', type=int, default=1,
                        help='Time step between points in the time series')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine operating mode
    viz_only = args.viz_only
    time_series_only = args.time_series_only
    full_pipeline = not (viz_only or time_series_only)
    
    # Handle manual scale - overrides skip_scale
    use_manual_scale = args.manual_scale is not None
    
    # Force skip optimization for visualization and time series only modes
    if viz_only or time_series_only:
        args.skip_fk = True
        if not use_manual_scale:
            args.skip_scale = True
    
    # Track all results for final report
    all_results = {
        'summary': {},
        'fk_params': {},
        'time_scale': {},
        'training_plots': [],
        'tumor_plots': [],
        'timepoints': [args.fit_time, args.scale_time] + args.test_time,
        'dice_scores': [],
        'tumor_volumes': []
    }
    
    print(f"Starting tumor growth prediction pipeline for {args.patient_id}")
    if viz_only:
        print("Mode: Visualization only")
    elif time_series_only:
        print("Mode: Time series generation only")
    else:
        print("Mode: Full pipeline")
    
    # Phase 1: Get FK parameters (either fit or load)
    print(f"\n--- Phase 1: FK Parameters for week {args.fit_time} ---")
    
    # Load data for the first time point
    print(f"Loading data for week {args.fit_time}...")
    fit_wm, fit_gm, fit_tumor, fit_affine = load_data(args.data_dir, args.patient_id, args.fit_time)
    
    # Either load or fit FK parameters
    if args.skip_fk:
        print("Loading FK parameters from saved results...")
        try:
            fk_params = load_fk_parameters(args.output_dir, args.fit_time)
            print(f"Loaded FK parameters: Dw={fk_params['Dw']:.6f}, rho={fk_params['rho']:.6f}")
            
            # For visualization, run forward model with loaded parameters
            # Create a temporary dictionary with the structure expected by the forward function
            temp_params = [
                fk_params['NxT1_pct'],
                fk_params['NyT1_pct'],
                fk_params['NzT1_pct'],
                fk_params['Dw'],
                fk_params['rho'],
                fk_params['thresholdT1c'],
                fk_params['thresholdFlair']
            ]
            
            # Prepare settings for the forward model
            settings = {
                "model_time": fk_params['model_time'],
                "resolution_factor": 0.7
            }
            
            # Create solver just for the forward model
            edema, necrotic, enhancing = prepare_tumor_regions(fit_tumor)
            
            solver = FK_CMAESSolver(
                settings=settings,
                wm=fit_wm,
                gm=fit_gm,
                edema=edema,
                enhancing=enhancing,
                necrotic=necrotic,
                segm=fit_tumor,
                workers=args.workers
            )
            
            # Run forward model to get predicted tumor
            fk_tumor = solver.forward(temp_params)
            
            if fk_tumor is None:
                print("Warning: Failed to generate tumor with loaded FK parameters.")
                fk_tumor = np.zeros_like(fit_tumor)
            
        except Exception as e:
            print(f"Error loading FK parameters: {str(e)}")
            print("Cannot continue without FK parameters. Exiting.")
            sys.exit(1)
    else:
        # Fit FK parameters
        # Prepare tumor regions
        edema, necrotic, enhancing = prepare_tumor_regions(fit_tumor)
        
        # Initialize settings
        settings = initialize_settings(edema, fit_gm.shape)
        
        # Create and run the FK parameter solver
        fk_solver = FK_CMAESSolver(
            settings=settings,
            wm=fit_wm,
            gm=fit_gm,
            edema=edema,
            enhancing=enhancing,
            necrotic=necrotic,
            segm=fit_tumor,
            workers=args.workers
        )
        
        # Run FK parameter optimization
        print(f"Running FK parameter optimization with {args.fk_generations} generations...")
        fk_tumor, fk_results = fk_solver.run(args.fk_generations)
        
        # Extract optimized parameters
        opt_params = fk_results.get("opt_params", None)
        
        if opt_params is not None:
            fk_params = {
                'Dw': opt_params[3],            # Diffusion coefficient
                'rho': opt_params[4],           # Proliferation rate
                'model_time': settings["model_time"],  # Fixed model time (30.0)
                'NxT1_pct': opt_params[0],      # Tumor position X
                'NyT1_pct': opt_params[1],      # Tumor position Y
                'NzT1_pct': opt_params[2],      # Tumor position Z
                'thresholdT1c': opt_params[5],  # T1c threshold
                'thresholdFlair': opt_params[6] # FLAIR threshold
            }
        else:
            print("Warning: Failed to extract optimized parameters from FK results")
            print("Cannot continue without FK parameters. Exiting.")
            sys.exit(1)
        
        # After FK parameter optimization, extract loss history
        fk_loss_history = []
        fk_param_history = {
            'Dw': [],
            'rho': [],
            'NxT1_pct': [],
            'NyT1_pct': [],
            'NzT1_pct': []
        }
        
        # Extract loss and parameter history from trace
        if 'trace' in fk_results:
            for gen_data in fk_results['trace']:
                if len(gen_data) > 5:  # Ensure we have generation data
                    # Best loss is at index 5
                    best_loss = gen_data[5]
                    
                    # Convert to float if possible
                    try:
                        if isinstance(best_loss, (list, tuple, np.ndarray)):
                            best_loss = float(best_loss[0])
                        else:
                            best_loss = float(best_loss)
                        fk_loss_history.append(best_loss)
                    except (IndexError, TypeError, ValueError):
                        print(f"Warning: Skipping invalid loss value: {best_loss}")
                        continue
                    
                    # Extract best parameters for this generation
                    best_params = gen_data[4]  # Best parameters at index 4
                    if isinstance(best_params, (list, tuple, np.ndarray)) and len(best_params) >= 5:
                        try:
                            fk_param_history['NxT1_pct'].append(float(best_params[0]))
                            fk_param_history['NyT1_pct'].append(float(best_params[1]))
                            fk_param_history['NzT1_pct'].append(float(best_params[2]))
                            fk_param_history['Dw'].append(float(best_params[3]))
                            fk_param_history['rho'].append(float(best_params[4]))
                        except (TypeError, ValueError):
                            print(f"Warning: Could not convert parameters to float: {best_params}")
        
        # Save results
        fk_dir = save_results(args.output_dir, 'fk', args.fit_time, fk_results, fk_params)
        
        # Generate visualization of loss progression
        if fk_loss_history:
            fk_loss_plot = visualize_training_progression(
                args.output_dir, 'fk', fk_loss_history, args.fk_generations
            )
            all_results['training_plots'].append(fk_loss_plot)
        
        # Generate visualization of parameter evolution (excluding model_time)
        if all(len(fk_param_history[k]) > 0 for k in fk_param_history):
            fk_param_plot = visualize_parameter_evolution(
                args.output_dir, 'fk', 
                [fk_param_history[k] for k in fk_param_history.keys()],
                list(fk_param_history.keys())
            )
            all_results['training_plots'].append(fk_param_plot)
    
    # Create tumor visualization for FK fitting
    visualize_results(args.output_dir, 'fk', args.fit_time, fit_wm, fit_tumor, fk_tumor, fk_params)
    
    # Generate 3D tumor visualization
    fk_tumor_plot = visualize_3d_tumor(
        args.output_dir, 'fk', args.fit_time, fit_wm, fit_tumor, fk_tumor
    )
    all_results['tumor_plots'].append(fk_tumor_plot)
    
    # Store FK parameters in results
    all_results['fk_params'] = fk_params
    
    # Calculate and store tumor volume and Dice score
    fit_tumor_volume = calculate_tumor_volume(fit_tumor)
    fit_dice = dice_coefficient(fk_tumor > 0.5, fit_tumor > 0)
    all_results['dice_scores'].append(fit_dice)
    all_results['tumor_volumes'].append(fit_tumor_volume)
    
    print(f"\nFK parameters phase completed.")
    print(f"Parameters: Dw={fk_params.get('Dw', 'N/A')}, rho={fk_params.get('rho', 'N/A')}, "
          f"model_time={fk_params.get('model_time', 'N/A')}")
    
    # Phase 2: Get time scale parameter (either fit, load, or use manual value)
    print(f"\n--- Phase 2: Time Scale Parameter ---")
    
    # Load data for the second time point if not in time_series_only mode or if we need to fit scale
    if not time_series_only or not (args.skip_scale or use_manual_scale):
        print(f"Loading data for week {args.scale_time}...")
        scale_wm, scale_gm, scale_tumor, scale_affine = load_data(
            args.data_dir, args.patient_id, args.scale_time
        )
    
    # Calculate physical time difference
    physical_time_diff = args.scale_time - args.fit_time
    
    # Handle time scale parameter based on mode
    if use_manual_scale:
        # Use manually specified time scale
        time_scale = args.manual_scale
        print(f"Using manually specified time scale: {time_scale:.6f}")
        
        # Save the manual scale parameter
        scale_params = {'time_scale': time_scale}
        save_results(args.output_dir, 'scale', args.scale_time, {'manual': True}, scale_params)
        
        # If we're not in time_series_only mode, generate prediction for visualization
        if not time_series_only:
            scale_tumor_pred = predict_tumor(
                fk_params=fk_params,
                time_scale=time_scale,
                reference_wm=fit_wm,
                reference_gm=fit_gm,
                reference_tumor=fit_tumor,
                reference_time=args.fit_time,
                target_time=args.scale_time
            )
            
            if scale_tumor_pred is None:
                print("Warning: Failed to generate tumor with specified time scale parameter.")
                scale_tumor_pred = np.zeros_like(scale_tumor)
    elif args.skip_scale:
        # Skip time scale optimization and load from saved results
        print("Loading time scale parameter from saved results...")
        try:
            time_scale = load_time_scale(args.output_dir, args.scale_time)
            print(f"Loaded time scale: {time_scale:.6f}")
            
            # If we're not in time_series_only mode, generate prediction for visualization
            if not time_series_only:
                scale_tumor_pred = predict_tumor(
                    fk_params=fk_params,
                    time_scale=time_scale,
                    reference_wm=fit_wm,
                    reference_gm=fit_gm,
                    reference_tumor=fit_tumor,
                    reference_time=args.fit_time,
                    target_time=args.scale_time
                )
                
                if scale_tumor_pred is None:
                    print("Warning: Failed to generate tumor with loaded time scale parameter.")
                    scale_tumor_pred = np.zeros_like(scale_tumor)
            
        except Exception as e:
            print(f"Error loading time scale parameter: {str(e)}")
            print("Cannot continue without time scale parameter. Exiting.")
            sys.exit(1)
    else:
        # Fit time scale parameter
        # Create and run the improved time scale solver
        time_solver = AdaptiveTimeScaleSolver(
            fk_params=fk_params,
            reference_wm=fit_wm,
            reference_gm=fit_gm,
            reference_tumor=fit_tumor,
            target_tumor=scale_tumor,
            physical_time_diff=physical_time_diff,
            workers=args.workers
        )
        
        print(f"Running adaptive time scale optimization with {args.scale_generations} generations...")
        scale_tumor_pred, time_scale, scale_results = time_solver.run(args.scale_generations, param_range=[1, 20.0])
        
        # After time scale optimization, extract loss history
        scale_loss_history = []
        if 'search_history' in scale_results:
            for time_scale_val, loss in scale_results['search_history']:
                if 0 <= loss <= 1:
                    scale_loss_history.append(loss)
        
        # Save results
        scale_dir = save_results(args.output_dir, 'scale', args.scale_time, scale_results, 
                               {'time_scale': time_scale})
        
        # Generate visualization of loss progression
        if scale_loss_history:
            scale_loss_plot = visualize_training_progression(
                args.output_dir, 'scale', scale_loss_history, len(scale_loss_history)
            )
            all_results['training_plots'].append(scale_loss_plot)
    
    # Store time scale parameter in results
    all_results['time_scale'] = {'time_scale': time_scale}
    
    # Create tumor visualization for time scale if not in time_series_only mode
    if not time_series_only:
        visualize_results(args.output_dir, 'scale', args.scale_time, scale_wm, scale_tumor, 
                        scale_tumor_pred, {'time_scale': time_scale})
        
        # Generate 3D tumor visualization
        scale_tumor_plot = visualize_3d_tumor(
            args.output_dir, 'scale', args.scale_time, scale_wm, scale_tumor, scale_tumor_pred
        )
        all_results['tumor_plots'].append(scale_tumor_plot)
        
        # Calculate and store tumor volume and Dice score
        scale_tumor_volume = calculate_tumor_volume(scale_tumor)
        scale_dice = dice_coefficient(scale_tumor_pred > 0.5, scale_tumor > 0)
        all_results['dice_scores'].append(scale_dice)
        all_results['tumor_volumes'].append(scale_tumor_volume)
    
    print(f"\nTime scale parameter phase completed.")
    print(f"Time scale: {time_scale:.4f}")
    
    # If time_series_only, generate time series
    if time_series_only or args.generate_time_series:
        print("\n--- Generating tumor progression time series ---")
        
        # Determine time range for the series
        start_time = args.start_time if args.start_time is not None else min(args.fit_time, 0)
        end_time = args.end_time if args.end_time is not None else max(args.test_time) + 10
        time_step = args.time_step
        
        print(f"Generating time series from week {start_time} to week {end_time} (step={time_step})...")
        
        # Generate predictions for all time points
        time_series_predictions = generate_time_series_predictions(
            fk_params=fk_params,
            time_scale=time_scale,
            reference_wm=fit_wm,
            reference_gm=fit_gm,
            reference_tumor=fit_tumor,
            reference_time=args.fit_time,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step
        )
        
        # Save predictions as NIfTI files
        time_series_dir, saved_files = save_time_series(
            args.output_dir, 
            args.patient_id, 
            time_series_predictions, 
            fit_affine
        )
        
        # Generate time series visualization
        time_series_plot = visualize_time_series(
            args.output_dir, 
            args.patient_id,
            fit_wm,  # Use the reference white matter as background
            time_series_predictions,
            fit_time=args.fit_time,
            scale_time=args.scale_time
        )
        
        all_results['tumor_plots'].append(time_series_plot)
        print(f"Time series generation completed. Files saved to {time_series_dir}")
        
        # If in time_series_only mode, exit here
        if time_series_only:
            print("\n" + "="*80)
            print(f"Time series generation completed for {args.patient_id}")
            print("="*80)
            print(f"FK Parameters: Dw={fk_params.get('Dw', 'N/A')}, rho={fk_params.get('rho', 'N/A')}")
            print(f"Time Scale: {time_scale:.4f}")
            print(f"Time series range: Week {start_time} to Week {end_time} (step={time_step})")
            print(f"Number of time points: {len(time_series_predictions)}")
            print(f"Results saved to: {args.output_dir}")
            print("="*80)
            return all_results
    
    # If in visualization only mode, exit here
    # Around line 1629, replace the if/elif for viz_only with:

    # If in visualization only mode, make sure we generate test visualizations
    if viz_only:
        print("\n--- Visualization Only Mode: Generating Test Visualizations ---")
        
        # Generate test visualizations for each test time
        test_results = []
        test_dice_scores = []
        test_tumor_volumes = []
        
        for test_time in args.test_time:
            print(f"\nGenerating visualization for week {test_time}...")
            
            try:
                # Load test data
                test_wm, test_gm, test_tumor, test_affine = load_data(
                    args.data_dir, args.patient_id, test_time
                )
                
                # Predict tumor at test time
                test_tumor_pred = predict_tumor(
                    fk_params=fk_params,
                    time_scale=time_scale,
                    reference_wm=fit_wm,
                    reference_gm=fit_gm,
                    reference_tumor=fit_tumor,
                    reference_time=args.fit_time,
                    target_time=test_time
                )
                
                if test_tumor_pred is not None:
                    # Calculate test Dice coefficient
                    test_dice = dice_coefficient(
                        test_tumor_pred > fk_params['thresholdFlair'], 
                        test_tumor > 0
                    )
                    print(f"Test prediction Dice coefficient: {test_dice:.4f}")
                    
                    # Create visualization
                    test_results_dict = {
                        'dice': test_dice,
                        'time_scale': time_scale
                    }
                    
                    visualize_results(args.output_dir, 'test', test_time, test_wm, test_tumor, 
                                    test_tumor_pred, test_results_dict)
                    
                    # Generate 3D tumor visualization
                    test_tumor_plot = visualize_3d_tumor(
                        args.output_dir, 'test', test_time, test_wm, test_tumor, test_tumor_pred
                    )
                    all_results['tumor_plots'].append(test_tumor_plot)
                    
                    # Calculate and store tumor volume
                    test_tumor_volume = calculate_tumor_volume(test_tumor)
                    
                    test_dice_scores.append(test_dice)
                    test_tumor_volumes.append(test_tumor_volume)
                    
                    # Update all results lists
                    all_results['dice_scores'].append(test_dice)
                    all_results['tumor_volumes'].append(test_tumor_volume)
                    
                else:
                    print(f"Test prediction for week {test_time} failed.")
            
            except Exception as e:
                print(f"Error during test visualization for week {test_time}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create a visualization comparing performance across all timepoints
        all_timepoints = [args.fit_time, args.scale_time] + args.test_time[:len(test_dice_scores)]
        all_dice_scores = [fit_dice, scale_dice] + test_dice_scores
        all_tumor_volumes = [fit_tumor_volume, scale_tumor_volume] + test_tumor_volumes
        
        if len(all_timepoints) > 2:
            timepoint_comparison = visualize_comparison_across_timepoints(
                args.output_dir, args.patient_id, 
                all_timepoints, 
                all_dice_scores,
                all_tumor_volumes
            )
            all_results['tumor_plots'].append(timepoint_comparison)
    
    # Phase 3: Test prediction on multiple test time points (full pipeline only)
    print(f"\n--- Phase 3: Testing prediction on {len(args.test_time)} time points ---")
    
    test_results = []
    test_dice_scores = []
    test_tumor_volumes = []
    test_predictions = []
    
    for test_time in args.test_time:
        print(f"\nTesting prediction for week {test_time}...")
        
        try:
            # Load test data
            test_wm, test_gm, test_tumor, test_affine = load_data(
                args.data_dir, args.patient_id, test_time
            )
            
            # Predict tumor at test time
            test_tumor_pred = predict_tumor(
                fk_params=fk_params,
                time_scale=time_scale,
                reference_wm=fit_wm,
                reference_gm=fit_gm,
                reference_tumor=fit_tumor,
                reference_time=args.fit_time,
                target_time=test_time
            )
            
            if test_tumor_pred is not None:
                # Calculate test Dice coefficient
                test_dice = dice_coefficient(test_tumor_pred > 0.5, test_tumor > 0)
                print(f"Test prediction Dice coefficient: {test_dice:.4f}")
                
                # Create visualization
                test_results_dict = {
                    'dice': test_dice,
                    'time_scale': time_scale
                }
                
                test_dir = save_results(args.output_dir, 'test', test_time, test_results_dict)
                visualize_results(args.output_dir, 'test', test_time, test_wm, test_tumor, 
                                 test_tumor_pred, test_results_dict)
                
                # Generate 3D tumor visualization
                test_tumor_plot = visualize_3d_tumor(
                    args.output_dir, 'test', test_time, test_wm, test_tumor, test_tumor_pred
                )
                all_results['tumor_plots'].append(test_tumor_plot)
                
                # Calculate and store tumor volume
                test_tumor_volume = calculate_tumor_volume(test_tumor)
                
                # Append results
                test_results.append({
                    'time': test_time,
                    'dice': test_dice,
                    'volume': test_tumor_volume,
                    'plot': test_tumor_plot
                })
                
                test_dice_scores.append(test_dice)
                test_tumor_volumes.append(test_tumor_volume)
                test_predictions.append(test_tumor_pred)
                
                # Update all results lists
                all_results['dice_scores'].append(test_dice)
                all_results['tumor_volumes'].append(test_tumor_volume)
                
                print(f"Test prediction for week {test_time} completed. Results saved to {test_dir}")
            else:
                print(f"Test prediction for week {test_time} failed.")
        
        except Exception as e:
            print(f"Error during test phase for week {test_time}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Update summary with test results
    all_results['summary']['test_dice_scores'] = test_dice_scores
    all_results['summary']['test_tumor_volumes'] = test_tumor_volumes
    
    # Create a visualization comparing performance across all timepoints
    all_timepoints = [args.fit_time, args.scale_time] + args.test_time[:len(test_dice_scores)]
    all_dice_scores = [fit_dice, scale_dice] + test_dice_scores
    all_tumor_volumes = [fit_tumor_volume, scale_tumor_volume] + test_tumor_volumes
    
    if len(all_timepoints) > 2:
        timepoint_comparison = visualize_comparison_across_timepoints(
            args.output_dir, args.patient_id, 
            all_timepoints, 
            all_dice_scores,
            all_tumor_volumes
        )
        all_results['tumor_plots'].append(timepoint_comparison)
    
    # Generate time series visualization if requested but not already done
    if args.generate_time_series and not time_series_only:
        print("\n--- Generating tumor progression time series ---")
        
        # Collect all timepoints and predictions in order
        time_series_timepoints = []
        time_series_predictions = {}
        
        # Start with fit time
        time_series_timepoints.append(args.fit_time)
        time_series_predictions[args.fit_time] = fk_tumor
        
        # Add scale time if different from fit time
        if args.scale_time != args.fit_time:
            time_series_timepoints.append(args.scale_time)
            time_series_predictions[args.scale_time] = scale_tumor_pred
        
        # Add all test timepoints and predictions
        for test_time, pred in zip(args.test_time, test_predictions):
            if test_time not in time_series_timepoints:  # Avoid duplicates
                time_series_timepoints.append(test_time)
                time_series_predictions[test_time] = pred
        
        # Generate time series visualization
        time_series_plot = visualize_time_series(
            args.output_dir, 
            args.patient_id, 
            fit_wm,  # Use the reference white matter as background
            time_series_predictions,
            fit_time=args.fit_time,
            scale_time=args.scale_time
        )
        
        all_results['tumor_plots'].append(time_series_plot)
    
    # Final report
    print("\n" + "="*80)
    print(f"Tumor Growth Prediction Pipeline completed for {args.patient_id}")
    print("="*80)
    print(f"FK Parameters: Dw={fk_params.get('Dw', 'N/A')}, rho={fk_params.get('rho', 'N/A')}")
    print(f"Time Scale: {time_scale:.4f}")
    print(f"Fit time (week {args.fit_time}) - Dice: {fit_dice:.4f}")
    print(f"Scale time (week {args.scale_time}) - Dice: {scale_dice:.4f}")
    
    for test_time, test_dice in zip(args.test_time[:len(test_dice_scores)], test_dice_scores):
        print(f"Test time (week {test_time}) - Dice: {test_dice:.4f}")
    
    print("\nResults saved to output directory.")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    main()