#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_PATH:?Set CUDA_PATH to a CUDA 12.8 toolkit or compatible newer CUDA 12.x installation}"
PYTHON="${PYTHON:-python3}"
NVIDIA_ROOT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/nvidia")')"
export LD_LIBRARY_PATH="$NVIDIA_ROOT/cuda_nvrtc/lib:$NVIDIA_ROOT/cublas/lib:$NVIDIA_ROOT/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p build results
"$PYTHON" "${STUDY:-gpu_study.py}" "$@"
