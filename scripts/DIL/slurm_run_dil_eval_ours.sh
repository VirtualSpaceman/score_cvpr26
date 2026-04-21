#!/bin/bash
#SBATCH --job-name=merge_domain_ours        # A name for your job
#SBATCH --output=slurm_logs/%x_%A_%a.out    # Standard output log: jobname_jobID_taskID.out
#SBATCH --error=slurm_logs/%x_%A_%a.err     # Standard error log
#SBATCH --partition=l40s,h100,a5000         # The partition to run on
#SBATCH --gres=gpu:1                        # Request 1 GPU per job
#SBATCH --cpus-per-task=16                  # Request 16 CPU cores per job
#SBATCH --mem=100G                          # Request 100GB of memory per job
#SBATCH --time=12:00:00                     # Time limit for each job (HH:MM:SS)

# --- Define the size of the job array ---
# Number of models (1) * number of datasets (3) * number of merge fns (6) = 18 jobs
#SBATCH --array=1-1

# --- Preparation ---
# Exit the script if any command fails
set -e

# Create log directory if it doesn't exist to prevent errors
mkdir -p slurm_logs

# Activate your Conda environment (ensure it's correctly installed)
source ~/miniconda3/bin/activate
source activate task_vectors

# --- Job Information ---
echo "------------------------------------------------"
echo "Starting Slurm Job"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "------------------------------------------------"

# --- Define the Experiment Grid ---
models=("ViT-B-32")
# datasets=("ImageNetR" "DomainNet" "FedISIC")
datasets=("ImageNetR")
# merge_fns=("saliency")
merge_fns=("saliency_spectrum")

# --- Map Slurm Task ID to 3D Experiment Parameters ---
# This math maps the linear task ID (1-12) to a 3D grid of parameters.
# We first make the task ID 0-indexed for easier modulo arithmetic.
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))

num_datasets=${#datasets[@]}
num_merge_fns=${#merge_fns[@]}

# Calculate the index for each parameter
model_index=$((task_id_zero_based / (num_datasets * num_merge_fns)))
dataset_index=$(((task_id_zero_based / num_merge_fns) % num_datasets))
merge_fn_index=$((task_id_zero_based % num_merge_fns))

# Assign the parameters for the current job
model=${models[$model_index]}
dataset=${datasets[$dataset_index]}
merge_fn=${merge_fns[$merge_fn_index]}

# Keep only the top p parameter of each layer 
ALPHA=0.1

# --- Execute the Job ---
echo "Running experiment with the following parameters:"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Merge Function: ${merge_fn}"

# Run the Python script with the parameters for this specific job
python3 merge_domain_splitted_leave_one_out.py --model ${model} --dataset ${dataset} --merge-fn ${merge_fn} --seed 5 --alpha $ALPHA

echo "Job finished successfully."