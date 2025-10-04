#!/bin/bash
#SBATCH --job-name=fitting            # Job name
#SBATCH --account=bcog-delta-cpu        # Your SLURM account
#SBATCH --partition=cpu                 # Partition (queue) to use
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64G                       # Memory allocation (adjust as needed)
#SBATCH --time=24:00:00                 # Maximum runtime
#SBATCH --output=fitting.%j.out         # Standard output log (%j = job ID)
#SBATCH --error=fitting.%j.err          # Standard error log (%j = job ID)
#SBATCH --mail-type=END,FAIL            # Email notifications for job completion or failure
#SBATCH --mail-user=jyin15@illinois.edu # Email address for notifications

# Set threading behavior
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh

conda activate base


# Set paths
DATA_DIR="/scratch/bcog/jyin15/Tumor/parcelled"
TUMOR_DIR="/projects/bcsl/shirui/LUMEIER"
OUTPUT_DIR="/scratch/bcog/jyin15/Tumor/processed"

# Run the Python script
python3 fitting.py
