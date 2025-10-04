#!/bin/bash
#SBATCH --job-name=copy_seg_masks        # Job name
#SBATCH --account=bcsl-delta-cpu         # Your SLURM account
#SBATCH --partition=cpu                  # Partition (queue) to use
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=4G                         # Memory allocation
#SBATCH --time=2:00:00                   # Maximum runtime
#SBATCH --output=copy_masks.%j.out       # Standard output log (%j = job ID)
#SBATCH --error=copy_masks.%j.err        # Standard error log (%j = job ID)

# --- Configuration Variables ---
BASE_DIR="/projects/bcsl/shirui/LUMEIER"            
OUTPUT_DIR="/scratch/bcsl/jyin15/Tumor/BioPhysModel/playground/data"

# --- Validate Paths ---
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory $BASE_DIR does not exist."
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory $OUTPUT_DIR does not exist."
    exit 1
fi

# --- Copy Segmentation Masks ---
echo "Starting copy of segmentation mask files..."
count=0

# Find all segmentation mask files
find "$BASE_DIR" -type f -path "*/Patient-091/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz" | while read -r seg_file; do
    # Extract patient and week info from the path
    relative_path="${seg_file#$BASE_DIR/}"
    patient_dir=$(echo "$relative_path" | cut -d'/' -f1)
    week_dir=$(echo "$relative_path" | cut -d'/' -f2)
    
    # Construct the destination directory path
    dest_dir="$OUTPUT_DIR/$patient_dir/$week_dir"
    
    # Check if destination directory exists (if FSL FAST was run for this patient/week)
    if [[ -d "$dest_dir" ]]; then
        # Check if destination already has the file
        if [[ -f "$dest_dir/seg_mask.nii.gz" ]]; then
            echo "Skipping: $dest_dir/seg_mask.nii.gz already exists"
        else
            echo "Copying: $seg_file -> $dest_dir/seg_mask.nii.gz"
            cp "$seg_file" "$dest_dir/seg_mask.nii.gz"
            ((count++))
        fi
    else
        echo "Warning: Destination directory $dest_dir does not exist, creating it..."
        mkdir -p "$dest_dir"
        echo "Copying: $seg_file -> $dest_dir/seg_mask.nii.gz"
        cp "$seg_file" "$dest_dir/seg_mask.nii.gz"
        ((count++))
    fi
done

echo "Process complete. Copied $count segmentation mask files."
