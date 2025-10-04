#!/usr/bin/env python3

import os
import subprocess
import sys
import time
from pathlib import Path

# Define patients and their timepoints with correct ID format (with leading zeros)
patients = [
    {"id": "Patient-025", "fit_time": 30, "scale_time": 43, "test_time": 55}
]

# Base directories
data_dir = "/scratch/bcsl/jyin15/Tumor/BioPhysModel/parcelled_and_masks"
output_base_dir = "/scratch/bcsl/jyin15/Tumor/BioPhysModel/demo"

# Create the main output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

def run_pipeline(patient):
    """Run the tumor growth pipeline for a specific patient."""
    patient_id = patient["id"]
    fit_time = patient["fit_time"]
    scale_time = patient["scale_time"]
    test_time = patient["test_time"]
    
    # Create patient-specific output directory
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    manualscale = 1
    
    # Command to run the pipeline
    cmd = [
        "python", "TumorGrowthPipeline.py",
        "--data_dir", data_dir,
        "--patient_id", patient_id,
        "--output_dir", patient_output_dir,
        "--fit_time", str(fit_time),
        "--scale_time", str(scale_time),
        "--test_time", str(test_time),
        "--fk_generations", "13",
        "--scale_generations", "20",
        "--workers", "8",
        "--skip_fk",
        "--skip_scale",
        "--manual_scale", str(manualscale),
        "--viz_only"
    ]
    
    print(f"\n{'='*80}")
    print(f"Running pipeline for {patient_id}")
    print(f"Timepoints: Train={fit_time}, Scale={scale_time}, Test={test_time}")
    print(f"Output directory: {patient_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command
    try:
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(f"[{patient_id}] {line.strip()}")
            sys.stdout.flush()
        
        process.stdout.close()
        return_code = process.wait()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if return_code == 0:
            print(f"\n{'='*80}")
            print(f"Successfully completed pipeline for {patient_id}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Error running pipeline for {patient_id}")
            print(f"Return code: {return_code}")
            print(f"{'='*80}\n")
            
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Exception running pipeline for {patient_id}: {str(e)}")
        print(f"{'='*80}\n")
        
def main():
    """Main function to run the pipeline for all patients."""
    print(f"\n{'='*80}")
    print(f"Starting tumor growth prediction for multiple patients")
    print(f"Data directory: {data_dir}")
    print(f"Output base directory: {output_base_dir}")
    print(f"{'='*80}\n")
    
    # Process each patient
    for patient in patients:
        run_pipeline(patient)
    
    print(f"\n{'='*80}")
    print(f"All pipeline runs completed")
    print(f"Results stored in: {output_base_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()