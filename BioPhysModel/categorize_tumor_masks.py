#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

def find_all_fk_params(patient_dir):
    """
    Find all FK parameter files for a patient.
    
    Args:
        patient_dir: Path to patient directory
        
    Returns:
        List of tuples (fit_time, param_file_path)
    """
    fk_dir = os.path.join(patient_dir, 'fk')
    
    if not os.path.exists(fk_dir):
        print(f"FK directory not found: {fk_dir}")
        return []
    
    # Look for all parameter files
    param_files = glob.glob(os.path.join(fk_dir, 'fk_params_t*.npy'))
    
    if not param_files:
        print(f"No FK parameter files found in {fk_dir}")
        return []
    
    # Extract time points from filenames
    params_info = []
    
    for param_file in param_files:
        filename = os.path.basename(param_file)
        try:
            # Extract time point from filename (format: fk_params_tXX.npy)
            fit_time = int(filename.split('_t')[1].split('.')[0])
            params_info.append((fit_time, param_file))
        except (IndexError, ValueError):
            print(f"Could not extract time point from {filename}, skipping")
    
    return sorted(params_info)  # Sort by fit_time

def load_fk_parameters(param_file):
    """
    Load FK parameters from a specific file.
    
    Args:
        param_file: Path to parameter file
        
    Returns:
        Dictionary with thresholdT1c and thresholdFlair
    """
    print(f"Loading FK parameters from {param_file}...")
    
    try:
        params = np.load(param_file, allow_pickle=True).item()
        
        # Check for required threshold parameters
        if 'thresholdT1c' not in params:
            raise KeyError(f"'thresholdT1c' parameter missing in {param_file}")
        
        if 'thresholdFlair' not in params:
            raise KeyError(f"'thresholdFlair' parameter missing in {param_file}")
        
        # Extract thresholds
        thresholds = {
            'thresholdT1c': params['thresholdT1c'],
            'thresholdFlair': params['thresholdFlair']
        }
        
        print(f"Loaded thresholds: T1c={thresholds['thresholdT1c']:.4f}, Flair={thresholds['thresholdFlair']:.4f}")
        return thresholds
        
    except Exception as e:
        print(f"Error loading FK parameters from {param_file}: {str(e)}")
        raise

def categorize_tumor(tumor_density, thresholdT1c, thresholdFlair):
    """
    Categorize tumor density into regions based on thresholds.
    
    Args:
        tumor_density: 3D array of tumor density (0-1)
        thresholdT1c: Threshold for enhancing tumor
        thresholdFlair: Threshold for edema
        
    Returns:
        3D array with categorical labels: 0=background, 2=edema, 4=enhancing
    """
    # Initialize empty mask with zeros (background)
    categorical_mask = np.zeros_like(tumor_density, dtype=np.uint8)
    
    # Apply thresholds for different regions
    edema_mask = np.logical_and(tumor_density > thresholdFlair, tumor_density < thresholdT1c)
    enhancing_mask = tumor_density >= thresholdT1c
    
    # Assign labels to the mask
    categorical_mask[edema_mask] = 2  # Edema = 2
    categorical_mask[enhancing_mask] = 4  # Enhancing = 4
    
    # Count voxels in each region
    edema_voxels = np.sum(edema_mask)
    enhancing_voxels = np.sum(enhancing_mask)
    total_voxels = edema_voxels + enhancing_voxels
    
    print(f"Categorized mask: {total_voxels} total tumor voxels "
          f"({edema_voxels} edema, {enhancing_voxels} enhancing)")
    
    return categorical_mask

def convert_density_to_categorical(input_path, output_path, thresholds):
    """
    Convert a continuous tumor density map to a categorical mask.
    
    Args:
        input_path: Path to continuous tumor map nii.gz
        output_path: Path to save categorical mask
        thresholds: Dictionary with thresholdT1c and thresholdFlair
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input image
        input_nii = nib.load(input_path)
        tumor_density = input_nii.get_fdata()
        
        # Check if already categorical
        unique_values = np.unique(tumor_density)
        if len(unique_values) <= 5 and np.all(np.equal(np.mod(unique_values, 1), 0)):
            print(f"WARNING: File appears to be already categorical with values: {unique_values}")
            print(f"Continuing with categorization anyway...")
        
        # Ensure values are in 0-1 range (normalize if needed)
        if np.max(tumor_density) > 1.0:
            print(f"Warning: Input has values > 1.0, normalizing to 0-1 range")
            tumor_density = tumor_density / np.max(tumor_density)
        
        # Apply thresholds to categorize tumor
        categorical_mask = categorize_tumor(
            tumor_density, 
            thresholds['thresholdT1c'], 
            thresholds['thresholdFlair']
        )
        
        # Create new nifti image with same metadata
        categorical_nii = nib.Nifti1Image(
            categorical_mask, 
            input_nii.affine, 
            input_nii.header
        )
        
        # Save output
        nib.save(categorical_nii, output_path)
        print(f"Saved categorical mask to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        raise

def process_patient_with_all_params(output_dir, patient_id):
    """
    Process all tumor maps for a patient using all available parameter sets.
    
    Args:
        output_dir: Base output directory
        patient_id: Patient ID
        
    Returns:
        Number of successfully processed files
    """
    print(f"\n{'='*80}")
    print(f"Processing {patient_id}")
    print(f"{'='*80}")
    
    patient_dir = os.path.join(output_dir, patient_id)
    
    # Find all parameter files
    params_info = find_all_fk_params(patient_dir)
    
    if not params_info:
        print(f"No FK parameter files found for {patient_id}")
        return 0
    
    print(f"Found {len(params_info)} parameter sets: {[f't{t}' for t, _ in params_info]}")
    
    # Find all tumor map files in time_series directory
    time_series_dir = os.path.join(patient_dir, 'time_series')
    if not os.path.exists(time_series_dir):
        print(f"Time series directory not found: {time_series_dir}")
        return 0
    
    # Find all nii.gz files in time_series
    tumor_maps = glob.glob(os.path.join(time_series_dir, 'tumor_week_*.nii.gz'))
    
    if not tumor_maps:
        print(f"No tumor map files found in {time_series_dir}")
        return 0
    
    print(f"Found {len(tumor_maps)} tumor map files")
    
    total_processed = 0
    
    # Process with each parameter set
    for fit_time, param_file in params_info:
        # Load parameters for this fit time
        try:
            thresholds = load_fk_parameters(param_file)
            
            # Create output directory for this parameter set
            categorical_dir = os.path.join(patient_dir, f'categorical_masks_t{fit_time}')
            os.makedirs(categorical_dir, exist_ok=True)
            
            print(f"\nProcessing with parameters from t{fit_time}...")
            processed = 0
            
            # Process each file
            for input_path in sorted(tumor_maps):
                # Create output filename based on input
                filename = os.path.basename(input_path)
                output_path = os.path.join(categorical_dir, filename)
                
                print(f"Processing {filename}...")
                convert_density_to_categorical(input_path, output_path, thresholds)
                processed += 1
            
            print(f"Successfully processed {processed} files with t{fit_time} parameters")
            print(f"Categorical masks saved to: {categorical_dir}")
            total_processed += processed
            
        except Exception as e:
            print(f"Error processing with t{fit_time} parameters: {str(e)}")
            print("Continuing with next parameter set...")
            continue
    
    return total_processed

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert continuous tumor maps to categorical masks')
    parser.add_argument('--output_dir', type=str, 
                        default=".",
                        help='Base directory containing patient results')
    parser.add_argument('--patient_ids', type=str, nargs='+',
                        default=["Patient-036", "Patient-059", "Patient-068", "Patient-078"],
                        help='List of patient IDs to process')
    parser.add_argument('--fit_times', type=int, nargs='+',
                        help='Specific fit times to use (optional)')
    parser.add_argument('--auto_detect_params', action='store_true',
                        help='Automatically detect all parameter files')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Tumor Map Categorization Tool")
    print(f"Output directory: {args.output_dir}")
    print(f"Patients to process: {args.patient_ids}")
    
    if args.auto_detect_params:
        print(f"Mode: Auto-detect all parameter files")
    elif args.fit_times:
        print(f"Mode: Using specified fit times: {args.fit_times}")
        # Ensure fit_times matches patient_ids length
        if len(args.fit_times) != len(args.patient_ids):
            print(f"ERROR: Number of fit times ({len(args.fit_times)}) doesn't match "
                f"number of patients ({len(args.patient_ids)})")
            print(f"Please provide one fit time for each patient ID.")
            sys.exit(1)
    else:
        print(f"WARNING: Neither --auto_detect_params nor --fit_times specified.")
        print(f"Defaulting to auto-detect mode.")
        args.auto_detect_params = True
    
    print(f"{'='*80}\n")
    
    # Process each patient
    total_processed = 0
    
    try:
        for i, patient_id in enumerate(args.patient_ids):
            if args.auto_detect_params:
                # Process with all parameter sets
                processed = process_patient_with_all_params(args.output_dir, patient_id)
            else:
                # Process with specific fit time
                fit_time = args.fit_times[i]
                # (This branch would need implementation if you want to use specific fit times)
                print(f"ERROR: Specific fit time mode not implemented yet.")
                sys.exit(1)
                
            total_processed += processed
        
        print(f"\n{'='*80}")
        print(f"Categorization completed. Processed {total_processed} tumor maps.")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Categorization failed")
        print(f"Reason: {str(e)}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()