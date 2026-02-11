# ROCm Build Setup for Flash Attention Wheels

This document describes the ROCm build infrastructure added to the flash-attention-prebuild-wheels repository.

## Overview

ROCm (Radeon Open Compute) support has been added to enable building Flash Attention wheels for AMD GPUs. The implementation follows the same pattern as existing CUDA builds but uses ROCm Docker containers and AMD-specific build configurations.

## Created Files

### 1. `build_rocm.sh`
Build script for creating Flash Attention wheels with ROCm support.

**Features:**
- Installs PyTorch for ROCm from the official PyTorch ROCm index
- Supports both Composable Kernel (CK) and Triton backends
- Auto-detects AMD GPU architecture (`gfx90a`, `gfx942`, etc.)
- Falls back to building for multiple common architectures when no GPU is detected
- Configures build parallelism based on available system resources
- Creates wheels with version labels like `rocm61torch2.5`

**Usage:**
```bash
./build_rocm.sh <flash-attn-version> <python-version> <torch-version> <rocm-version>

# Example:
./build_rocm.sh 2.8.3 3.11 2.5.1 6.2
```

**Environment Variables:**
- `BUILD_TRITON_BACKEND=true` - Also build Triton backend variant (optional)
- `PYTORCH_ROCM_ARCH` - Specify target GPU architectures (auto-detected if not set)
- `MAX_JOBS` - Override parallel build jobs

### 2. `.github/workflows/_build_rocm.yml`
Reusable GitHub Actions workflow for ROCm builds.

**Features:**
- Uses official ROCm PyTorch Docker containers (`rocm/pytorch:rocm*`)
- Supports multiple Python versions via deadsnakes PPA
- Includes GPU device passthrough for container (`/dev/kfd`, `/dev/dri`)
- Tests wheel installation before upload
- Optionally builds both CK and Triton backend wheels
- Applies `auditwheel repair` for distribution compatibility

**Inputs:**
- `flash-attn-version` - Flash Attention version to build (required)
- `python-version` - Python version (required)
- `torch-version` - PyTorch version (required)
- `rocm-version` - ROCm version (required, e.g., "6.1" or "6.2")
- `runner` - Runner type (default: "ubuntu-22.04")
- `is-upload` - Whether to upload to GitHub Releases (default: true)
- `build-triton-backend` - Also build Triton backend (default: false)

### 3. Updated `create_matrix.py`
Added ROCm build matrix configuration.

**ROCm Matrix:**
```python
ROCM_MATRIX = {
    "flash-attn-version": ["2.6.3", "2.7.4", "2.8.3"],
    "python-version": ["3.10", "3.11", "3.12"],
    "torch-version": ["2.5.1", "2.6.0"],
    "rocm-version": ["6.1", "6.2"],
}
```

**Note:** Set `"rocm": ROCM_MATRIX` in the `main()` function to enable ROCm builds.

### 4. Updated `.github/workflows/build.yml`
Integrated ROCm builds into the main build workflow.

**Changes:**
- Added `build_wheels_rocm` job that uses the `_build_rocm.yml` workflow
- Updated `update_release_notes` and `update_docs` jobs to include ROCm builds in dependencies

## ROCm Architecture Support

The build system supports the following AMD GPU architectures:

- **gfx90a** - AMD Instinct MI200 series (MI210, MI250, MI250X)
- **gfx942** - AMD Instinct MI300 series (MI300A, MI300X)
- **gfx1030** - AMD Radeon RX 6000 series
- **gfx1100** - AMD Radeon RX 7000 series

When building in a GitHub Actions runner without AMD GPUs, the build script compiles for all common architectures to ensure broad compatibility.

## Backend Options

Flash Attention 2 on ROCm supports two backends:

### Composable Kernel (CK) - Default
- Native AMD backend optimized for ROCm
- Better integration with AMD GPU features
- Default choice for production builds
- Controlled by `FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"`

### Triton
- OpenAI Triton backend
- Python-friendly kernel implementation
- Useful for experimental features
- Controlled by `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`

Wheels are labeled to distinguish backends (e.g., `rocm62torch2.5` vs `rocm62torch2.5triton`).

## Wheel Naming Convention

ROCm wheels follow this naming pattern:
```
flash_attn-{version}+rocm{rocm_version}torch{torch_version}-cp{python_ver}-cp{python_ver}-linux_x86_64.whl

Example:
flash_attn-2.8.3+rocm62torch2.5-cp311-cp311-linux_x86_64.whl
```

For Triton backend:
```
flash_attn-2.8.3+rocm62torch2.5triton-cp311-cp311-linux_x86_64.whl
```

## Enabling ROCm Builds

To enable ROCm builds in the GitHub Actions workflow:

1. Open `create_matrix.py`
2. Find the `main()` function
3. Change `"rocm": False` to `"rocm": ROCM_MATRIX`
4. Adjust the matrix values as needed for your requirements
5. Commit and create a release tag (e.g., `v0.1.0`)

The workflow will automatically:
- Create a GitHub release
- Build wheels for all matrix combinations
- Upload wheels to the release
- Update documentation

## Testing ROCm Wheels

After building, test the wheel:

```bash
pip install flash_attn-2.8.3+rocm62torch2.5-cp311-cp311-linux_x86_64.whl
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import flash_attn_2_cuda; print('Flash Attention ROCm loaded successfully')"
```

## Requirements

### For GitHub Actions:
- Ubuntu 22.04 runners (GitHub-hosted or self-hosted)
- Docker support
- Sufficient disk space (~20GB per build)
- ROCm-capable GPU (optional, builds work without GPU)

### For Local Builds:
- ROCm installation (6.1+ recommended)
- ROCm-enabled PyTorch
- Build dependencies: `git`, `ninja-build`, `cmake`, `python3-dev`
- AMD GPU (optional, for architecture detection)

## Docker Container Details

The workflow uses official ROCm PyTorch containers that match the published tags, for example:
```
rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
```

Device passthrough options:
```yaml
--device=/dev/kfd 
--device=/dev/dri 
--group-add video
--cap-add=SYS_PTRACE
--security-opt seccomp=unconfined
--ipc=host
```

These options enable GPU access from within the container, though the builds can complete without physical AMD GPUs present.

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Flash Attention on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html)
- [ROCm PyTorch Docker Images](https://hub.docker.com/r/rocm/pytorch)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

## Troubleshooting

### Build Failures

1. **Out of memory**: Reduce `MAX_JOBS` in the build script
2. **Architecture mismatch**: Set `PYTORCH_ROCM_ARCH` explicitly
3. **PyTorch version issues**: Verify ROCm version compatibility with PyTorch

### Common Issues

- **Container GPU access**: Ensure Docker has proper device permissions
- **Python version not found**: The deadsnakes PPA may not have all versions immediately
- **Wheel incompatibility**: Use `auditwheel repair` output for broader compatibility

## Future Enhancements

Potential improvements:
- Self-hosted runners with AMD GPUs for faster builds
- ROCm version auto-detection
- Multi-architecture wheel bundling
- Performance benchmarking integration
- Support for newer ROCm versions as they're released
