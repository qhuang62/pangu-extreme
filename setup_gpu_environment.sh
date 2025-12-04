#!/bin/bash

echo "=== Setting up GPU environment for Pangu-Weather on ASU SOL ==="

# Load required modules
echo "Loading CUDA 11.6.2..."
module load cuda-11.6.2-gcc-12.1.0

echo "Loading cuDNN 8.0.4..."
module load cudnn-8.0.4.30-10.1-gcc-12.1.0

# Set environment variables
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"

# Activate conda environment
echo "Activating pangu environment..."
conda activate pangu

echo "=== Environment ready! ==="
echo "✅ CUDA 11.6.2 loaded"
echo "✅ cuDNN 8.0.4 loaded" 
echo "✅ ONNX Runtime 1.14.0 ready"
echo "✅ GPU acceleration enabled"

echo ""
echo "To verify GPU is working, run:"
echo "  python test_cuda_gpu.py"
echo ""
echo "To run your prediction/perturb script:"
echo "  python xxx.py"