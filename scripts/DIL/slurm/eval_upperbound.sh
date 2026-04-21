#!/bin/bash
#SBATCH --job-name=upperbound_merge                  # A name for your job
#SBATCH --output=slurm_logs/upperbound_merge/%x_%A_%a.out    # Standard output log: jobname_jobID_taskID.out
#SBATCH --error=slurm_logs/upperbound_merge/%x_%A_%a.err     # Standard error log
#SBATCH --partition=h100,l40s,a5000         # The partition to run on (please change if needed)
#SBATCH --gres=gpu:1                        # Request 1 GPU per job
#SBATCH --cpus-per-task=16                  # Request 16 CPU cores per job
#SBATCH --mem=100G                          # Request 100GB of memory per job
#SBATCH --time=12:00:00                     # Time limit for each job (HH:MM:SS)
#SBATCH --array=1-24

# --- 1. Configuration Section ---
# Define the models and datasets to create combinations for.
MODELS=("ViT-B-32" "ViT-B-16" "ViT-L-14")
# MODELS=("ViT-B-32" "ViT-B-16")
# MODELS=("ViT-L-14")
DATASETS=("DomainNet" "ImageNetR" "PACS" "OfficeHome" "NICOpp" "RetinaDomains" "FedISIC" "TerraIncognita")

# --- 2. Automatically Set Slurm Array Size ---
num_models=${#MODELS[@]}
num_datasets=${#DATASETS[@]}

# --- 3. Preparation ---
set -e
mkdir -p slurm_logs/upperbound_merge
# Activate your Conda environment (ensure it's correctly installed)
source ~/miniconda3/bin/activate
source activate task_vectors

# --- 4. Job Execution ---
echo "------------------------------------------------"
echo "Starting Slurm Job"

echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "------------------------------------------------"

# Map the linear Slurm task ID to the correct parameters from our arrays.
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))

# This math maps the linear task ID to a 2D grid of parameters (model x dataset).
model_index=$(( task_id_zero_based / num_datasets ))
dataset_index=$(( task_id_zero_based % num_datasets ))

# Get the parameters for the current job
model=${MODELS[$model_index]}
dataset=${DATASETS[$dataset_index]}

# --- Execute the Job ---
echo "Running experiment with the following parameters:"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Seed: 5"

# Run the target Python script with the parameters for this specific job
# The '&' is not needed as Slurm manages the jobs in parallel.
python3 eval_domain_splitted_leave_one_out.py \
    --model "${model}" \
    --seed 5 \
    --dataset "${dataset}" \

echo "Job finished successfully."
