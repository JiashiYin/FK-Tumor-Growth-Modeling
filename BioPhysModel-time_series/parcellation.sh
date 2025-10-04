#!/bin/bash
#SBATCH --job-name=parcel               # Job name
#SBATCH --account=bcsl-delta-gpu       # Your SLURM account
#SBATCH --partition=gpuA40x4                 # Partition (queue) to use
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64G                       # Memory allocation (adjust as needed)
#SBATCH --time=10:00:00                 # Maximum runtime
#SBATCH --output=parcel.%j.out          # Standard output log (%j = job ID)
#SBATCH --error=parcel.%j.err           # Standard error log (%j = job ID)

# Set threading behavior
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Configuration Variables ---
BASE_DIR="/projects/bcsl/shirui/LUMEIER"            
OUTPUT_DIR="/scratch/bcsl/jyin15/Tumor/BioPhysModel/parcelled_and_masks"  
THREADS=4

# --- Validate Paths ---
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory $BASE_DIR does not exist."
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory $OUTPUT_DIR..."
    mkdir -p "$OUTPUT_DIR"
fi

# --- Conda Environment Activation ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fslmaths-env

# --- FSL Configuration ---
export FSLDIR=$CONDA_PREFIX
source $FSLDIR/etc/fslconf/fsl.sh

if ! command -v fast &> /dev/null; then
    echo "Error: FSL FAST is not available. Ensure FSL is installed and configured."
    exit 1
fi

echo "Activated Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "FSLDIR is set to: $FSLDIR"
echo "Starting FSL FAST segmentation..."

# --- Loop Over All T1 Skull Strip Files ---
find "$BASE_DIR" -type f -path "*/Patient-*/week-*/DeepBraTumIA-segmentation/atlas/skull_strip/t1_skull_strip.nii.gz" | while read -r t1_file; do
    relative_path="${t1_file#$BASE_DIR/}"
    # Extract patient and week info from the path
    patient_dir=$(echo "$relative_path" | cut -d'/' -f1)
    week_dir=$(echo "$relative_path" | cut -d'/' -f2)
    patient_week_dir="$patient_dir/$week_dir"
    subject_output_dir="$OUTPUT_DIR/$patient_week_dir"

    # Check if output files already exist
    if [[ -f "$subject_output_dir/T1_seg.nii.gz" && \
          -f "$subject_output_dir/T1_pve_0.nii.gz" && \
          -f "$subject_output_dir/T1_pve_1.nii.gz" && \
          -f "$subject_output_dir/T1_pve_2.nii.gz" ]]; then
        echo "Skipping $t1_file - output files already exist"
        continue
    fi

    # Create the output directory if it doesn't exist
    if [[ ! -d "$subject_output_dir" ]]; then
        echo "Creating output directory: $subject_output_dir"
        mkdir -p "$subject_output_dir"
    fi

    # Run FSL FAST segmentation
    echo "Processing $t1_file (Output: $subject_output_dir)..."
    fast -t 1 -n 3 -o "$subject_output_dir/T1" "$t1_file" &

    # Limit the number of concurrent jobs
    while (( $(jobs -r | wc -l) >= THREADS )); do
        sleep 1
    done
done

# Wait for all jobs to finish
wait

echo "All images processed."