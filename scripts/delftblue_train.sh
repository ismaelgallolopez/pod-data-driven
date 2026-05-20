#!/bin/bash
#SBATCH --job-name=pod-pinn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

# Create logs directory
mkdir -p logs

# Load modules
module load 2025 python/3.11.9 cuda/12.1

# Activate virtual environment
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Print GPU info for confirmation
echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
nvidia-smi

# Set environment
export PYTHONPATH=$SLURM_SUBMIT_DIR

# Run training
cd $SLURM_SUBMIT_DIR
python main.py
