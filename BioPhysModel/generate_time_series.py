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
from TumorGrowthToolkit.FK import Solver as FKSolver

def load_parameters(output_dir, patient_id, fit_time, scale_time):
    """
    Load optimized FK parameters and time scale from saved results.
    """
    print(f"Loading parameters for {patient_id}...")
    
    # Load FK parameters
    fk_dir = os.path.join(output_dir, patient_id, "fk")
    fk_params_file = os.path.join(fk_dir, f"fk_params_t{fit_time}.npy")
    
    if not os.path.exists(fk_params_file):
        raise FileNotFoundError(f"FK parameters file not found: {fk_params_file}")
    
    fk_params = np.load(fk_params_file, allow_pickle=True).item()
    print(f"Loaded FK parameters: Dw={fk_params.get('Dw', 'N/A')}, rho={fk_params.get('rho', 'N/A')}")
    
    # Load time scale parameter
    scale_dir = os.path.join(output_dir, patient_id, "scale")
    scale_params_file = os.path.join(scale_dir, f"scale_params_t{scale_time}.npy")
    
    if not os.path.exists(scale_params_file):
        raise FileNotFoundError(f"Time scale parameters file not found: {scale_params_file}")
    
    scale_params = np.load(scale_params_file, allow_pickle=True).item()
    time_scale = scale_params.get('time_scale', None)
    
    if time_scale is None:
        raise ValueError("Time scale parameter not found in scale parameters file")
    
    print(f"Loaded time scale: {time_scale:.4f}")
    
    return fk_params, time_scale

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
    
    return wm_data, gm_data, tumor_data, affine, wm_nifti

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
        'segm': reference_tumor
    }
    
    # Run forward model
    solver = FKSolver(parameters)
    result = solver.solve()
    
    if not result.get('success', False):
        print(f"Prediction failed: {result.get('error', 'Unknown error')}")
        return None
    
    return result['final_state']

def categorize_tumor(tumor_density, thresholdT1c, thresholdFlair):
    """
    Categorize tumor density into different regions based on thresholds.
    Returns a labeled tumor mask:
    0: background
    1: necrotic - assuming all high-density regions are necrotic for simplicity
    2: edema - regions between the flair and t1c thresholds
    4: enhancing - assuming no explicit enhancing regions for simplicity
    """
    # Initialize empty mask
    labeled_mask = np.zeros_like(tumor_density, dtype=np.uint8)
    
    # Apply thresholds for different regions
    edema_mask = np.logical_and(tumor_density >= thresholdFlair, tumor_density < thresholdT1c)
    enhancing_mask = tumor_density >= thresholdT1c
    
    # Assign values to the mask
    labeled_mask[edema_mask] = 2  # Edema = 2
    labeled_mask[enhancing_mask] = 4  # Enhancing = 4
    
    return labeled_mask

def save_tumor_mask(mask, output_path, reference_nifti):
    """Save tumor mask as .nii.gz using reference image for metadata."""
    # Create nifti image with same metadata as reference
    mask_nifti = nib.Nifti1Image(mask, reference_nifti.affine, reference_nifti.header)
    
    # Save to disk
    nib.save(mask_nifti, output_path)
    
    return output_path

def find_representative_slice(tumor_masks):
    """
    Find a representative slice across all tumor masks that shows good tumor visibility.
    
    Args:
        tumor_masks: List of 3D tumor mask arrays
        
    Returns:
        int: Index of the representative slice
    """
    # Combine all masks to find where tumor exists across time
    combined_mask = np.zeros_like(tumor_masks[0])
    for mask in tumor_masks:
        combined_mask = np.logical_or(combined_mask, mask > 0)
    
    # Find the slice with maximum tumor coverage
    slice_sums = np.sum(combined_mask, axis=(0, 1))
    
    # If no tumor found, use middle slice
    if np.max(slice_sums) == 0:
        return combined_mask.shape[2] // 2
    
    # Return the slice with maximum tumor coverage
    return np.argmax(slice_sums)

def visualize_timeseries(patient_id, time_series_dir, reference_wm, start_time, end_time, output_dir):
    """
    Create a visualization of tumor progression over time on a representative slice.
    
    Args:
        patient_id: ID of the patient
        time_series_dir: Directory containing tumor mask time series
        reference_wm: White matter data to use as background
        start_time: Start week of time series
        end_time: End week of time series
        output_dir: Directory to save visualization
        
    Returns:
        str: Path to saved visualization
    """
    print(f"Creating time series visualization for {patient_id}...")
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load tumor masks for all weeks
    tumor_masks = []
    weeks = []
    
    for week in range(start_time, end_time + 1):
        mask_path = os.path.join(time_series_dir, f"tumor_week_{week:03d}.nii.gz")
        if os.path.exists(mask_path):
            mask_nifti = nib.load(mask_path)
            mask_data = mask_nifti.get_fdata()
            tumor_masks.append(mask_data)
            weeks.append(week)
    
    if not tumor_masks:
        print("No tumor masks found for visualization")
        return None
    
    # Find representative slice
    rep_slice = find_representative_slice(tumor_masks)
    print(f"Using representative slice: {rep_slice}")
    
    # Determine how many weeks to visualize (ensure it's not too crowded)
    if len(weeks) > 10:
        # Sample approximately 10 evenly spaced timepoints
        indices = np.linspace(0, len(weeks)-1, 10, dtype=int)
        selected_masks = [tumor_masks[i] for i in indices]
        selected_weeks = [weeks[i] for i in indices]
    else:
        selected_masks = tumor_masks
        selected_weeks = weeks
    
    # Create the visualization
    fig_width = min(20, 2 * len(selected_weeks))  # Limit width to 20 inches
    plt.figure(figsize=(fig_width, 5))
    
    for i, (week, mask) in enumerate(zip(selected_weeks, selected_masks)):
        plt.subplot(1, len(selected_masks), i+1)
        
        # Create a color-coded overlay:
        # - Red: Enhancing tumor (label 4)
        # - Green: Edema (label 2)
        
        # Create RGB image
        rgb_overlay = np.zeros((*reference_wm[:, :, rep_slice].shape, 3))
        
        # Add white matter as grayscale background
        background = reference_wm[:, :, rep_slice]
        # Normalize to 0-1 range for better visualization
        if np.max(background) > 0:
            background = background / np.max(background)
        
        for c in range(3):
            rgb_overlay[:, :, c] = background
        
        # Add tumor regions as colored overlays
        edema_mask = mask[:, :, rep_slice] == 2
        enhancing_mask = mask[:, :, rep_slice] == 4
        
        # Green for edema
        rgb_overlay[edema_mask, 0] *= 0.3  # Reduce red
        rgb_overlay[edema_mask, 1] = 0.8   # Strong green
        rgb_overlay[edema_mask, 2] *= 0.3  # Reduce blue
        
        # Red for enhancing
        rgb_overlay[enhancing_mask, 0] = 0.8  # Strong red
        rgb_overlay[enhancing_mask, 1] *= 0.3  # Reduce green
        rgb_overlay[enhancing_mask, 2] *= 0.3  # Reduce blue
        
        plt.imshow(rgb_overlay)
        plt.title(f"Week {week}")
        plt.axis('off')
    
    plt.suptitle(f"{patient_id} Tumor Progression (Slice {rep_slice})", fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(viz_dir, f"{patient_id}_tumor_progression.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")
    return output_path

def process_patient(data_dir, output_base_dir, patient):
    """Process a single patient to generate time series of tumor masks."""
    patient_id = patient["id"]
    fit_time = patient["fit_time"]
    scale_time = patient["scale_time"]
    test_time = patient["test_time"]
    
    print(f"\n{'='*80}")
    print(f"Processing {patient_id}")
    print(f"Training timepoint: {fit_time}, Scaling timepoint: {scale_time}, Test timepoint: {test_time}")
    print(f"{'='*80}\n")
    
    try:
        # Load parameters from the fitted model
        fk_params, time_scale = load_parameters(output_base_dir, patient_id, fit_time, scale_time)
        
        # Load reference data (from fit_time)
        reference_wm, reference_gm, reference_tumor, reference_affine, reference_nifti = load_data(
            data_dir, patient_id, fit_time
        )
        
        # Determine time range for the series
        start_time = min(fit_time, scale_time, test_time)
        end_time = max(fit_time, scale_time, test_time)
        
        # Create output directory for time series
        time_series_dir = os.path.join(output_base_dir, patient_id, "time_series")
        os.makedirs(time_series_dir, exist_ok=True)
        
        print(f"Generating time series from week {start_time} to week {end_time}")
        
        # Process each week in the range
        successful_weeks = []
        failed_weeks = []
        
        for week in range(start_time, end_time + 1):
            try:
                print(f"Processing week {week}...")
                
                # Predict tumor density
                tumor_density = predict_tumor(
                    fk_params=fk_params,
                    time_scale=time_scale,
                    reference_wm=reference_wm,
                    reference_gm=reference_gm,
                    reference_tumor=reference_tumor,
                    reference_time=fit_time,
                    target_time=week
                )
                
                if tumor_density is None:
                    print(f"Failed to predict tumor for week {week}")
                    failed_weeks.append(week)
                    continue
                
                # Categorize tumor using thresholds from fk_params
                thresholdT1c = fk_params.get('thresholdT1c', 0.675)  # Default if not found
                thresholdFlair = fk_params.get('thresholdFlair', 0.25)  # Default if not found
                
                print(f"Categorizing tumor using thresholds - T1c: {thresholdT1c:.4f}, Flair: {thresholdFlair:.4f}")
                tumor_mask = categorize_tumor(tumor_density, thresholdT1c, thresholdFlair)
                
                # Save the tumor mask
                output_path = os.path.join(time_series_dir, f"tumor_week_{week:03d}.nii.gz")
                save_tumor_mask(tumor_mask, output_path, reference_nifti)
                
                print(f"Saved tumor mask for week {week} to {output_path}")
                successful_weeks.append(week)
                
            except Exception as e:
                print(f"Error processing week {week}: {str(e)}")
                failed_weeks.append(week)
        
        # Create visualization of tumor progression
        if successful_weeks:
            viz_path = visualize_timeseries(
                patient_id=patient_id,
                time_series_dir=time_series_dir,
                reference_wm=reference_wm,
                start_time=start_time,
                end_time=end_time,
                output_dir=output_base_dir
            )
        
        # Print summary
        print(f"\nSuccessfully generated {len(successful_weeks)} of {end_time - start_time + 1} masks")
        if failed_weeks:
            print(f"Failed weeks: {failed_weeks}")
        
        return {
            "patient_id": patient_id,
            "success": len(failed_weeks) == 0,
            "total_weeks": end_time - start_time + 1,
            "successful_weeks": len(successful_weeks),
            "failed_weeks": len(failed_weeks),
            "visualization": viz_path if successful_weeks else None
        }
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "patient_id": patient_id,
            "success": False,
            "error": str(e)
        }

def main():
    """Main function to generate time series for all patients."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate time series of tumor masks from fitted models')
    parser.add_argument('--data_dir', type=str, 
                        default="/scratch/bcsl/jyin15/Tumor/BioPhysModel/parcelled_and_masks",
                        help='Base directory containing patient data')
    parser.add_argument('--output_dir', type=str, 
                        default="/scratch/bcsl/jyin15/Tumor/BioPhysModel/demo",
                        help='Directory containing fitted models and where to save results')
    parser.add_argument('--patient_ids', type=str, nargs='+',
                        default=["Patient-036"],
                        help='List of patient IDs to process')
    
    args = parser.parse_args()
    
    # Patients and their timepoints
    patients = [
        {"id": "Patient-036", "fit_time": 0, "scale_time": 2, "test_time": 20}
    ]
    
    # Filter patients by command line arguments if provided
    if args.patient_ids:
        patients = [p for p in patients if p["id"] in args.patient_ids]
    
    print(f"\n{'='*80}")
    print(f"Tumor Growth Time Series Generator")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patients to process: {[p['id'] for p in patients]}")
    print(f"{'='*80}\n")
    
    # Process each patient
    results = []
    for patient in patients:
        result = process_patient(args.data_dir, args.output_dir, patient)
        results.append(result)
        print(f"{'='*80}\n")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"Time Series Generation Summary")
    print(f"{'='*80}")
    
    successful_patients = sum(1 for r in results if r["success"])
    print(f"Successfully processed {successful_patients}/{len(patients)} patients")
    
    for result in results:
        patient_id = result["patient_id"]
        if result.get("success", False):
            print(f"  - {patient_id}: Generated {result.get('successful_weeks', 0)}/{result.get('total_weeks', 0)} tumor masks")
            if result.get("visualization"):
                print(f"    Visualization: {result.get('visualization')}")
        else:
            print(f"  - {patient_id}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\nTumor mask time series saved in respective patient directories under 'time_series/'")
    print(f"Visualizations saved in '{args.output_dir}/visualizations/'")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()