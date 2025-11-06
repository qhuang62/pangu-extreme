#!/usr/bin/env python3

import os
import onnxruntime as ort

print("=== CUDA GPU Test for ONNX Runtime ===")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Test creating a CUDA session
try:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    
    cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}
    
    session = ort.InferenceSession(
        '/scratch/qhuang62/pangu-extreme/model/pangu_weather_24.onnx', 
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
    )
    print('✅ CUDA session created successfully!')
    print(f'Active providers: {session.get_providers()}')
    
    # Check if CUDA is actually being used
    if 'CUDAExecutionProvider' in session.get_providers():
        print('🚀 GPU acceleration is ACTIVE!')
    else:
        print('⚠️  Falling back to CPU execution')
        
except Exception as e:
    print(f'❌ CUDA session failed: {e}')
    print('Falling back to CPU test...')
    try:
        session = ort.InferenceSession(
            '/scratch/qhuang62/pangu-extreme/model/pangu_weather_24.onnx', 
            providers=['CPUExecutionProvider']
        )
        print('✅ CPU session works')
    except Exception as e2:
        print(f'❌ Even CPU failed: {e2}')

print("\n=== Environment Check ===")
print(f"PATH: {os.environ.get('PATH', '').split(':')[0:3]}...")  # First 3 paths
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '').split(':')[0:3]}...")  # First 3 paths