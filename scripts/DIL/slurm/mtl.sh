#!/bin/bash
#SBATCH --job-name=merge_dil_mtl           # A name for your job
#SBATCH --output=slurm_logs/dil-mtl/%x_%A_%a.out    # Standard output log: jobname_jobID_taskID.out
#SBATCH --error=slurm_logs/dil-mtl/%x_%A_%a.err     # Standard error log
#SBATCH --partition=l40s,h100,rtx8000        # The partition to run on
#SBATCH --gres=gpu:1                        # Request 1 GPU per job
#SBATCH --cpus-per-task=16                  # Request 16 CPU cores per job
#SBATCH --mem=100G                          # Request 100GB of memory per job
#SBATCH --time=12:00:00                     # Time limit for each job (HH:MM:SS)

# Total Jobs = (num_models) * (num_datasets) * (num_merge_fns)
# Example: 3 model(s) * 8 datasets * 2 merge_fns = 48 job(s)
#SBATCH --array=1-48

# --- 1. Configuration Section ---
# --- Model Selection ---
# To run only a single model, comment the line above and uncomment one of these:
# models=("ViT-B-32")
# models=("ViT-B-16")
# models=("ViT-L-14")
# To run two models, use this line: 
# models=("ViT-B-32" "ViT-B-16")
# To run all models, use this line:
models=("ViT-B-32" "ViT-B-16" "ViT-L-14")

# --- Dataset Selection ---
# datasets=("ImageNetR" "DomainNet")
datasets=("RetinaDomains" "PACS" "OfficeHome" "NICOpp" "FedISIC" "TerraIncognita" "ImageNetR" "DomainNet")
# datasets=("ImageNetR")

# --- Other Parameters ---
# merge_fns=("randmix" "pcb" "magmax" "tsv" "isoc" "ties" "saliency_spectrum" "avg" "saliency_ties")
# merge_fns=("saliency_ties")
merge_fns=("isoc_changeb_v1" "isoc_changeb_v2")

# --- 2. Preparation ---
set -e
mkdir -p slurm_logs/dil-mtl
source ~/miniconda3/bin/activate
source activate task_vectors

# --- 3. Job Information ---
echo "------------------------------------------------"
echo "Starting Slurm Job"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "------------------------------------------------"


# --- 4. Map Slurm Task ID to 3D Experiment Parameters ---
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
num_datasets=${#datasets[@]}
num_merge_fns=${#merge_fns[@]}

# This math maps the linear task ID to a 3D grid of parameters.
model_index=$(( task_id_zero_based / (num_datasets * num_merge_fns) ))
dataset_index=$(( (task_id_zero_based / num_merge_fns) % num_datasets ))
merge_fn_index=$(( task_id_zero_based % num_merge_fns ))

# Assign the parameters for the current job
model=${models[$model_index]}
dataset=${datasets[$dataset_index]}
merge_fn=${merge_fns[$merge_fn_index]}


# --- 5. Execute the Job ---
echo "Running experiment with the following parameters:"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Merge Function: ${merge_fn}"

# ALPHA=0.5 -> better for medical tasks, OfficeHome and TerraIncognita
ALPHA=0.2
BETA=2.0

# Run the Python script with the parameters for this specific job
python3 merge_domain_splitted_DIL_MTL.py \
    --model "${model}" \
    --merge-fn "${merge_fn}" \
    --dataset "${dataset}" \
    --seed 5 \
    --alpha $ALPHA \
    --beta $BETA

echo "Job finished successfully."
