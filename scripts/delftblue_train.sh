#!/bin/bash
#SBATCH --job-name=pod-pinn-train
#SBATCH --partition=gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/delftblue/train_%j.out
#SBATCH --error=logs/delftblue/train_%j.err

module load miniconda3
module load cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pod-pinn

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$SLURM_SUBMIT_DIR

cd $SLURM_SUBMIT_DIR

mkdir -p logs/delftblue

nvidia-smi
python main.py
echo "Training finished: $(date)"
