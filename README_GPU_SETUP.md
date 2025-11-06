# GPU Setup for Pangu-Weather on ASU SOL HPC

## Quick Start

1. **Setup GPU environment** (run this every time):
   ```bash
   cd /scratch/qhuang62/pangu-extreme
   source setup_gpu_environment.sh
   ```

2. **Verify GPU is working**:
   ```bash
   python test_cuda_gpu.py
   ```
   You should see: `🚀 GPU acceleration is ACTIVE!`

3. **Run the notebook**:
   ```bash
   jupyter notebook pangu_sandy_2012_init12.ipynb
   ```

## Working Configuration

- **CUDA**: 11.6.2
- **cuDNN**: 8.0.4.30
- **ONNX Runtime**: 1.14.0
- **Hardware**: NVIDIA A100-SXM4-80GB

## If GPU Setup Fails

Check the modules are loaded:
```bash
module list
# Should show:
#   1) jupyter/latest
#   2) cuda-11.6.2-gcc-12.1.0
#   3) cudnn-8.0.4.30-10.1-gcc-12.1.0
```

If not, run:
```bash
module load cuda-11.6.2-gcc-12.1.0
module load cudnn-8.0.4.30-10.1-gcc-12.1.0
```

## Environment Variables

The setup script will automatically set:
- `CUDA_HOME`: Points to CUDA 11.6.2 installation
- `PATH`: Includes CUDA bin directory
- `LD_LIBRARY_PATH`: Includes CUDA and cuDNN libraries