#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def load_tumor_mask(mask_path):
    """
    Load a tumor mask file and return its data.
    
    Args:
        mask_path: Path to the tumor mask file
        
    Returns:
        Tuple of: (mask_data, voxel_dimensions)
    """
    try:
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        
        # Get voxel dimensions from the affine matrix
        voxel_dims = np.array([
            np.sqrt(np.sum(mask_nii.affine[:3, 0] ** 2)),
            np.sqrt(np.sum(mask_nii.affine[:3, 1] ** 2)),
            np.sqrt(np.sum(mask_nii.affine[:3, 2] ** 2))
        ])
        
        return mask_data, voxel_dims
    except Exception as e:
        print(f"Error loading mask {mask_path}: {str(e)}")
        return None, None

def calculate_tumor_volumes(mask_data, voxel_dims):
    """
    Calculate volumes for different tumor regions.
    
    Args:
        mask_data: 3D array of tumor mask data
        voxel_dims: Voxel dimensions in mm
        
    Returns:
        Dictionary of volumes in mm³ (edema, enhancing, total)
    """
    # Calculate voxel volume in mm³
    voxel_volume = np.prod(voxel_dims)
    
    # Count voxels in each region
    # Standard segmentation: 1=necrotic, 2=edema, 4=enhancing
    edema_voxels = np.sum(mask_data == 2)
    enhancing_voxels = np.sum(mask_data == 4)
    necrotic_voxels = np.sum(mask_data == 1)
    
    # If no standard labels found, try binary approach
    if (edema_voxels + enhancing_voxels + necrotic_voxels) == 0:
        # Binary approach: any non-zero value is tumor
        total_voxels = np.sum(mask_data > 0)
        edema_voxels = total_voxels
        enhancing_voxels = 0
        necrotic_voxels = 0
    else:
        total_voxels = edema_voxels + enhancing_voxels + necrotic_voxels
    
    # Calculate volumes
    volumes = {
        'edema': edema_voxels * voxel_volume,
        'enhancing': enhancing_voxels * voxel_volume,
        'necrotic': necrotic_voxels * voxel_volume,
        'total': total_voxels * voxel_volume
    }
    
    return volumes

def extract_week_number(mask_path):
    """
    Extract week number from the mask file path.
    
    Args:
        mask_path: Path to the tumor mask file
        
    Returns:
        Week number as an integer
    """
    filename = os.path.basename(mask_path)
    # Expected format: tumor_week_XXX.nii.gz
    try:
        week_str = filename.split('_')[2].split('.')[0]
        return int(week_str)
    except (IndexError, ValueError):
        # Try alternative approach - extract any number
        import re
        matches = re.findall(r'\d+', filename)
        if matches:
            return int(matches[0])
        else:
            print(f"Couldn't extract week number from {filename}, using 0")
            return 0

def process_patient(patient_dir):
    """
    Process all tumor masks for a patient and calculate volume progression.
    
    Args:
        patient_dir: Patient directory containing time_series subfolder
        
    Returns:
        DataFrame with volume data for all time points
    """
    # Find time series directory
    time_series_dir = os.path.join(patient_dir, "time_series")
    
    if not os.path.exists(time_series_dir):
        print(f"Time series directory not found: {time_series_dir}")
        return None
    
    # Find all tumor mask files
    mask_files = glob.glob(os.path.join(time_series_dir, "tumor_week_*.nii.gz"))
    
    if not mask_files:
        print(f"No tumor mask files found in {time_series_dir}")
        return None
    
    print(f"Found {len(mask_files)} tumor mask files")
    
    # Process each mask file
    volumes_data = []
    
    for mask_path in mask_files:
        # Extract week number
        week = extract_week_number(mask_path)
        
        # Load mask
        mask_data, voxel_dims = load_tumor_mask(mask_path)
        
        if mask_data is None:
            continue
        
        # Calculate volumes
        volumes = calculate_tumor_volumes(mask_data, voxel_dims)
        
        # Add to data list
        volumes_data.append({
            'week': week,
            'edema_volume': volumes['edema'],
            'enhancing_volume': volumes['enhancing'],
            'necrotic_volume': volumes['necrotic'],
            'total_volume': volumes['total']
        })
    
    # Convert to DataFrame and sort by week
    if volumes_data:
        df = pd.DataFrame(volumes_data)
        df = df.sort_values('week')
        return df
    else:
        return None

def visualize_volume_progression(df, patient_id, output_dir):
    """
    Create visualization of tumor volume progression over time.
    
    Args:
        df: DataFrame with volume data
        patient_id: ID of the patient
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    if df is None or len(df) < 2:
        print(f"Not enough data points for visualization for {patient_id}")
        return None
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a multi-plot figure
    plt.figure(figsize=(15, 10))
    
    # Plot total volume
    plt.subplot(2, 2, 1)
    plt.plot(df['week'], df['total_volume'], 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Volume (mm³)', fontsize=12)
    plt.title('Total Tumor Volume Progression', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot volume by region
    plt.subplot(2, 2, 2)
    if np.sum(df['enhancing_volume']) > 0 or np.sum(df['necrotic_volume']) > 0:
        plt.plot(df['week'], df['edema_volume'], 'g-o', linewidth=2, markersize=6, label='Edema')
        plt.plot(df['week'], df['enhancing_volume'], 'r-o', linewidth=2, markersize=6, label='Enhancing')
        plt.plot(df['week'], df['necrotic_volume'], 'k-o', linewidth=2, markersize=6, label='Necrotic')
        plt.legend()
    else:
        plt.plot(df['week'], df['total_volume'], 'b-o', linewidth=2, markersize=6, label='Total Tumor')
        plt.legend()
    
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Volume (mm³)', fontsize=12)
    plt.title('Tumor Volume by Region', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot weekly growth rate
    plt.subplot(2, 2, 3)
    
    # Calculate week-to-week volume change
    df['volume_change'] = df['total_volume'].diff()
    df['weeks_elapsed'] = df['week'].diff()
    df['weekly_growth_rate'] = df['volume_change'] / df['weeks_elapsed']
    
    # Filter out the first row (which has NaN due to diff)
    growth_data = df.dropna()
    
    if len(growth_data) > 0:
        plt.plot(growth_data['week'], growth_data['weekly_growth_rate'], 'm-o', linewidth=2, markersize=6)
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Growth Rate (mm³/week)', fontsize=12)
        plt.title('Weekly Growth Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'Insufficient data for growth rate', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
    
    # Plot percentage of total volume by region
    plt.subplot(2, 2, 4)
    
    if np.sum(df['enhancing_volume']) > 0 or np.sum(df['necrotic_volume']) > 0:
        # Calculate percentages
        df['edema_pct'] = df['edema_volume'] / df['total_volume'] * 100
        df['enhancing_pct'] = df['enhancing_volume'] / df['total_volume'] * 100
        df['necrotic_pct'] = df['necrotic_volume'] / df['total_volume'] * 100
        
        plt.stackplot(df['week'], 
                      df['edema_pct'], 
                      df['enhancing_pct'], 
                      df['necrotic_pct'],
                      labels=['Edema', 'Enhancing', 'Necrotic'],
                      colors=['green', 'red', 'black'],
                      alpha=0.7)
        plt.legend(loc='upper left')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Percentage of Total Volume', fontsize=12)
        plt.title('Tumor Composition', fontsize=14)
        plt.ylim(0, 100)
    else:
        plt.text(0.5, 0.5, 'No region data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
    
    plt.suptitle(f'{patient_id} Tumor Volume Progression', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    # Save the visualization
    output_path = os.path.join(viz_dir, f"{patient_id}_volume_progression.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Volume progression visualization saved to {output_path}")
    
    # Save data to CSV for reference
    csv_path = os.path.join(viz_dir, f"{patient_id}_volume_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Volume data saved to {csv_path}")
    
    return output_path

def visualize_all_patients_comparison(patients_data, output_dir):
    """
    Create a comparison visualization of tumor volume progression across all patients.
    
    Args:
        patients_data: Dictionary mapping patient IDs to their volume DataFrames
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create the comparison visualization
    plt.figure(figsize=(15, 8))
    
    # Plot total volume for all patients
    for patient_id, df in patients_data.items():
        if df is not None and len(df) >= 2:
            plt.plot(df['week'], df['total_volume'], 'o-', linewidth=2, markersize=6, label=patient_id)
    
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Volume (mm³)', fontsize=12)
    plt.title('Tumor Volume Progression Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(viz_dir, "all_patients_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All patients comparison visualization saved to {output_path}")
    
    return output_path

def main():
    """Main function to process all patients and generate visualizations."""
    parser = argparse.ArgumentParser(description='Generate tumor volume progression visualizations')
    parser.add_argument('--base_dir', type=str, 
                        default="/scratch/bcsl/jyin15/Tumor/BioPhysModel/demo",
                        help='Base directory containing patient results')
    parser.add_argument('--patient_ids', type=str, nargs='+',
                        default=["Patient-068", "Patient-036", "Patient-059", "Patient-078"],
                        help='List of patient IDs to process')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Tumor Volume Progression Visualization Generator")
    print(f"Base directory: {args.base_dir}")
    print(f"{'='*80}\n")
    
    # Process each patient
    all_patients_data = {}
    
    for patient_id in args.patient_ids:
        print(f"\nProcessing {patient_id}...")
        patient_dir = os.path.join(args.base_dir, patient_id)
        
        if not os.path.exists(patient_dir):
            print(f"Patient directory not found: {patient_dir}")
            continue
        
        # Process patient's tumor masks
        df = process_patient(patient_dir)
        
        if df is not None:
            # Create visualization
            output_path = visualize_volume_progression(df, patient_id, args.base_dir)
            
            if output_path:
                print(f"Generated volume progression visualization for {patient_id}")
                all_patients_data[patient_id] = df
        else:
            print(f"No volume data generated for {patient_id}")
    
    # Create comparison visualization if multiple patients have data
    if len(all_patients_data) > 1:
        comparison_path = visualize_all_patients_comparison(all_patients_data, args.base_dir)
        print(f"\nGenerated comparison visualization for all patients")
    
    print(f"\n{'='*80}")
    print("Tumor volume progression visualization completed")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()