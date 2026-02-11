#!/bin/bash

set -e

# Parameters with defaults
FLASH_ATTN_VERSION=$1
PYTHON_VERSION=$2
TORCH_VERSION=$3
ROCM_VERSION=$4

echo "Building Flash Attention for ROCm with parameters:"
echo "  Flash-Attention: $FLASH_ATTN_VERSION"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  ROCm: $ROCM_VERSION"

# Set ROCm and PyTorch versions
MATRIX_ROCM_VERSION=$(echo $ROCM_VERSION | awk -F \. {'print $1 "." $2'})
MATRIX_TORCH_VERSION=$(echo $TORCH_VERSION | awk -F \. {'print $1 "." $2'})

echo "Derived versions:"
echo "  ROCm Matrix: $MATRIX_ROCM_VERSION"
echo "  Torch Matrix: $MATRIX_TORCH_VERSION"

# Install PyTorch for ROCm
echo "Installing PyTorch $TORCH_VERSION for ROCm $MATRIX_ROCM_VERSION..."
if [[ $TORCH_VERSION == *"dev"* ]]; then
  pip install --force-reinstall --no-cache-dir --pre torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/nightly/rocm${MATRIX_ROCM_VERSION}
else
  pip install --force-reinstall --no-cache-dir torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/rocm${MATRIX_ROCM_VERSION}
fi

# Install additional dependencies
echo "Installing build dependencies..."
pip install ninja packaging

# Verify installation
echo "Verifying installations..."
python -V
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('ROCm:', torch.version.hip if hasattr(torch.version, 'hip') else 'Not found')"

# Display ROCm information
if command -v rocm-smi &> /dev/null; then
  echo "ROCm SMI:"
  rocm-smi --showproductname || true
fi

if command -v rocminfo &> /dev/null; then
  echo "Available ROCm devices:"
  rocminfo | grep -E "Name:|Marketing Name:" || true
fi

# Checkout flash-attn
echo "Checking out flash-attention v$FLASH_ATTN_VERSION..."
git clone https://github.com/Dao-AILab/flash-attention.git -b "v$FLASH_ATTN_VERSION"

# Determine MAX_JOBS based on system resources
NUM_THREADS=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "System resources:"
echo "  CPU threads: $NUM_THREADS"
echo "  RAM: ${RAM_GB}GB"

# Calculate MAX_JOBS based on available resources
# ROCm builds are memory intensive, so we use conservative estimates
if [[ -z "${MAX_JOBS:-}" ]]; then
  MAX_PRODUCT_CPU=$NUM_THREADS
  MAX_PRODUCT_RAM=$(awk -v ram="$RAM_GB" 'BEGIN {print int(ram / 4)}')
  MAX_JOBS=$((MAX_PRODUCT_CPU < MAX_PRODUCT_RAM ? MAX_PRODUCT_CPU : MAX_PRODUCT_RAM))
  
  # Ensure minimum values
  MAX_JOBS=$((MAX_JOBS < 1 ? 1 : MAX_JOBS))
  
  # Cap at 8 to avoid overwhelming the system
  MAX_JOBS=$((MAX_JOBS > 8 ? 8 : MAX_JOBS))
fi

echo "Build parallelism settings:"
echo "  MAX_JOBS: $MAX_JOBS"

# Detect GPU architecture if available
if command -v rocminfo &> /dev/null; then
  GPU_ARCH=$(rocminfo | grep -o -m 1 'gfx[0-9a-z]*' | head -n 1)
  if [ -n "$GPU_ARCH" ]; then
    echo "Detected GPU architecture: $GPU_ARCH"
    export PYTORCH_ROCM_ARCH=$GPU_ARCH
  fi
fi

# Set default architectures if not detected
# Common ROCm architectures: gfx90a (MI200), gfx942 (MI300), gfx1030, gfx1100
if [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
  # Build for multiple common architectures
  export PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1030;gfx1100"
  echo "No GPU detected, building for multiple architectures: $PYTORCH_ROCM_ARCH"
fi

# Build wheels with Composable Kernel (CK) backend (default for AMD)
echo "Building wheels with Composable Kernel backend..."
cd flash-attention

LOCAL_VERSION_LABEL="rocm${MATRIX_ROCM_VERSION//./}torch${MATRIX_TORCH_VERSION}"

# Disable Triton backend to use Composable Kernel (CK) backend
export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_LOCAL_VERSION=${LOCAL_VERSION_LABEL}

MAX_JOBS=$MAX_JOBS time python setup.py bdist_wheel --dist-dir=dist

wheel_name=$(basename $(ls dist/*.whl | head -n 1))
echo "Built wheel: $wheel_name"

# Optional: Also build Triton backend if requested
if [ "${BUILD_TRITON_BACKEND:-false}" = "true" ]; then
  echo "Building additional wheel with Triton backend..."
  
  # Clean previous build
  python setup.py clean
  rm -rf build dist
  
  LOCAL_VERSION_LABEL_TRITON="rocm${MATRIX_ROCM_VERSION//./}torch${MATRIX_TORCH_VERSION}triton"
  
  export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
  export FLASH_ATTN_LOCAL_VERSION=${LOCAL_VERSION_LABEL_TRITON}
  
  MAX_JOBS=$MAX_JOBS time python setup.py bdist_wheel --dist-dir=dist_triton
  
  wheel_name_triton=$(basename $(ls dist_triton/*.whl | head -n 1))
  echo "Built Triton wheel: $wheel_name_triton"
fi

echo "Build complete!"
