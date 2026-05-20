#!/bin/bash
# One-time setup script for DelftBlue login node
# Run interactively: bash scripts/delftblue_setup.sh

echo "=========================================="
echo "Pod-Data-Driven PINN Setup for DelftBlue"
echo "=========================================="

# Load modules
module load 2023r1 python/3.10.4 cuda/11.7

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 11.7 support
echo "Installing PyTorch with CUDA 11.7..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
echo "Installing dependencies..."
pip install numpy scipy matplotlib

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Transfer data files to the cluster:"
echo "   scp -r data/ <netid>@login.delftblue.tudelft.nl:~/pod-data-driven/"
echo ""
echo "2. Submit training job:"
echo "   sbatch scripts/delftblue_train.sh"
echo ""
echo "3. Check job status:"
echo "   squeue -u <netid>"
echo ""
echo "4. Monitor output:"
echo "   tail -f logs/train_<jobid>.log"
echo ""
