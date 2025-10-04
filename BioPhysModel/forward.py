import os
import numpy as np
import nibabel as nib
import time
from scipy import ndimage
import cmaesFK
import tools


# Function to load the brain matter and tumor segmentation data
def load_data(wm_path, gm_path, tumor_path):
    WM = nib.load(wm_path).get_fdata()
    GM = nib.load(gm_path).get_fdata()
    segmentation = nib.load(tumor_path).get_fdata()
    return WM, GM, segmentation


# Function to initialize solver settings
def initialize_settings(edema, gm_shape):
    settings = {}
    com = ndimage.center_of_mass(edema)

    settings["rho0"] = 0.06
    settings["dw0"] = 0.001
    settings["thresholdT1c"] = 0.675
    settings["thresholdFlair"] = 0.25
    settings["NxT1_pct0"] = float(com[0] / gm_shape[0])
    settings["NyT1_pct0"] = float(com[1] / gm_shape[1])
    settings["NzT1_pct0"] = float(com[2] / gm_shape[2])
    settings["parameterRanges"] = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3], [0.5, 0.85], [0.001, 0.5]]
    settings["workers"] = 9
    settings["sigma0"] = 0.02
    settings["resolution_factor"] = {0: 0.3, 0.7: 0.5}
    settings["generations"] = 12
    return settings


# Function to save results
def save_results(settings, resultDict, resultTumor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_settings.npy"), settings)
    np.save(os.path.join(output_dir, f"gen_{settings['generations']}_results.npy"), resultDict)
    tools.writeNii(resultTumor, path=os.path.join(output_dir, f"gen_{settings['generations']}_result.nii.gz"))


# Function to check if output files exist
def output_exists(output_dir, generations):
    settings_file = os.path.join(output_dir, f"gen_{generations}_settings.npy")
    results_file = os.path.join(output_dir, f"gen_{generations}_results.npy")
    tumor_file = os.path.join(output_dir, f"gen_{generations}_result.nii.gz")
    
    return all(os.path.exists(f) for f in [settings_file, results_file, tumor_file])


# Main function to process a single patient-week directory
def process_patient_week(wm_path, gm_path, tumor_path, output_dir):
    # Check if output already exists
    settings = {"generations": 12}  # Default value for checking
    if output_exists(output_dir, settings["generations"]):
        print(f"Output already exists for {output_dir}. Skipping...")
        return

    # Load data
    WM, GM, segmentation = load_data(wm_path, gm_path, tumor_path)

    # Define tumor regions
    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4

    # Initialize solver settings
    settings = initialize_settings(edema, GM.shape)

    # Initialize CMA-ES solver
    solver = cmaesFK.CmaesSolver(settings, WM, GM, edema, enhancing, necrotic, segmentation)
    resultTumor, resultDict = solver.run()
    resultTumor = resultTumor.astype(int)

    # Save results
    save_results(settings, resultDict, resultTumor, output_dir)


# Traverse the directory structure
def traverse_and_process(data_dir, tumor_dir, output_dir):
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for patient in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient)
        tumor_patient_path = os.path.join(tumor_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for week in os.listdir(patient_path):
            week_path = os.path.join(patient_path, week)
            tumor_week_path = os.path.join(tumor_patient_path, week)
            if not os.path.isdir(week_path) or not os.path.isdir(tumor_week_path):
                continue

            # Paths to input files
            gm_path = os.path.join(week_path, "T1_pve_1.nii.gz")
            wm_path = os.path.join(week_path, "T1_pve_2.nii.gz")
            tumor_path = os.path.join(tumor_week_path, "DeepBraTumIA-segmentation", "native", "segmentation", "t1_seg_mask.nii.gz")

            # Ensure all paths exist
            if not (os.path.exists(wm_path) and os.path.exists(gm_path) and os.path.exists(tumor_path)):
                print(f"Skipping {week_path}: Missing required files.")
                error_count += 1
                continue

            # Output directory
            output_patient_week_dir = os.path.join(output_dir, patient, week)
            
            # Check if output already exists
            if output_exists(output_patient_week_dir, 12):
                print(f"Skipping {patient}/{week}: Output already exists.")
                skipped_count += 1
                continue

            print(f"Processing {patient}/{week}...")
            try:
                process_patient_week(wm_path, gm_path, tumor_path, output_patient_week_dir)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {patient}/{week}: {str(e)}")
                error_count += 1

    return processed_count, skipped_count, error_count


if __name__ == "__main__":
    data_dir = "/scratch/bcog/jyin15/Tumor/BioPhysModel/parcelled"
    tumor_dir = "/projects/bcsl/shirui/LUMEIER"
    output_dir = "/scratch/bcog/jyin15/Tumor/processed"

    start_time = time.time()
    processed, skipped, errors = traverse_and_process(data_dir, tumor_dir, output_dir)
    elapsed_time = round((time.time() - start_time) / 60, 2)
    
    print(f"\nProcessing completed in {elapsed_time} minutes.")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors encountered: {errors}")