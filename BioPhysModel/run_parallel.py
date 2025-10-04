#!/usr/bin/env python3

import os
import subprocess
import sys
import time
import concurrent.futures
from pathlib import Path
import threading

# Define patients and their timepoints with correct ID format (with leading zeros)
patients = [
    {"id": "Patient-068", "fit_time": 35, "scale_time": 22, "test_time": 47},
    {"id": "Patient-036", "fit_time": 2, "scale_time": 0, "test_time": 5},
    {"id": "Patient-059", "fit_time": 68, "scale_time": 55, "test_time": 84},
    {"id": "Patient-078", "fit_time": 19, "scale_time": 5, "test_time": 29}
]

# Base directories
data_dir = "/scratch/bcsl/jyin15/Tumor/BioPhysModel/parcelled_and_masks"
output_base_dir = "/scratch/bcsl/jyin15/Tumor/BioPhysModel/demo"

# Create the main output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Lock for preventing output interleaving
print_lock = threading.Lock()

def stream_output(process, patient_id):
    """Stream process output with patient ID prefix."""
    for line in iter(process.stdout.readline, ''):
        with print_lock:
            print(f"[{patient_id}] {line.strip()}")
            sys.stdout.flush()

def run_pipeline(patient):
    """Run the tumor growth pipeline for a specific patient."""
    patient_id = patient["id"]
    fit_time = patient["fit_time"]
    scale_time = patient["scale_time"]
    test_time = patient["test_time"]
    
    # Create patient-specific output directory
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
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
        "--workers", "6"  # We could reduce this for parallel runs
    ]
    
    with print_lock:
        print(f"\n{'='*80}")
        print(f"Starting pipeline for {patient_id}")
        print(f"Timepoints: Train={fit_time}, Scale={scale_time}, Test={test_time}")
        print(f"Output directory: {patient_output_dir}")
        print(f"{'='*80}\n")
    
    # Run the command
    start_time = time.time()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Create a thread to stream the output
        output_thread = threading.Thread(target=stream_output, args=(process, patient_id))
        output_thread.daemon = True
        output_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        output_thread.join()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        with print_lock:
            if return_code == 0:
                print(f"\n{'='*80}")
                print(f"Successfully completed pipeline for {patient_id}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
                print(f"{'='*80}\n")
                return {"patient_id": patient_id, "success": True, "time": elapsed_time}
            else:
                print(f"\n{'='*80}")
                print(f"Error running pipeline for {patient_id}")
                print(f"Return code: {return_code}")
                print(f"{'='*80}\n")
                return {"patient_id": patient_id, "success": False, "return_code": return_code}
            
    except Exception as e:
        with print_lock:
            print(f"\n{'='*80}")
            print(f"Exception running pipeline for {patient_id}: {str(e)}")
            print(f"{'='*80}\n")
        return {"patient_id": patient_id, "success": False, "error": str(e)}
        
def main():
    """Main function to run the pipeline for all patients in parallel."""
    with print_lock:
        print(f"\n{'='*80}")
        print(f"Starting tumor growth prediction for multiple patients IN PARALLEL")
        print(f"Data directory: {data_dir}")
        print(f"Output base directory: {output_base_dir}")
        print(f"{'='*80}\n")
    
    # Maximum number of parallel processes (adjust based on system resources)
    max_workers = min(len(patients), 4)  # Limit to 4 parallel jobs or less
    
    results = []
    # Process patients in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patients for processing
        futures = {executor.submit(run_pipeline, patient): patient["id"] for patient in patients}
        
        # As jobs complete
        for future in concurrent.futures.as_completed(futures):
            patient_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                with print_lock:
                    print(f"Error processing {patient_id}: {str(e)}")
    
    # Print summary
    with print_lock:
        print(f"\n{'='*80}")
        print(f"All pipeline runs completed")
        print(f"Results stored in: {output_base_dir}")
        
        # Print summary of successful and failed runs
        successful = [r for r in results if r.get("success", False)]
        print(f"\nSuccessful runs: {len(successful)}/{len(patients)}")
        for s in successful:
            print(f"  - {s['patient_id']}: {s['time']/60:.2f} minutes")
            
        failed = [r for r in results if not r.get("success", False)]
        if failed:
            print(f"\nFailed runs: {len(failed)}/{len(patients)}")
            for f in failed:
                print(f"  - {f['patient_id']}: {f.get('error', '') or f.get('return_code', '')}")
        
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
