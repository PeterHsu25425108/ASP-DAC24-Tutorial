#!/bin/bash

set -e  # Exit on error

echo "=============================================="
echo "ASP-DAC24-Tutorial Dependency Installation"
echo "Installing to system Python: /usr/bin/python3"
echo "=============================================="

# Use workspace for pip cache to avoid filling root filesystem
export PIP_CACHE_DIR=/workspace/.pip-cache
mkdir -p /workspace/.pip-cache

# Force deactivate conda
source /opt/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true

# Disable conda auto-activation
conda config --set auto_activate_base false 2>/dev/null || true

# Clean up space before starting
echo ""
echo "[0/4] Cleaning up disk space..."
apt-get clean
apt-get autoremove -y 2>/dev/null || true
rm -rf /tmp/* /var/tmp/* /var/cache/apt/archives/* 2>/dev/null || true
pip3 cache purge 2>/dev/null || true
rm -rf /root/.cache/pip 2>/dev/null || true
conda clean --all -y 2>/dev/null || true
rm -rf /opt/miniconda3/pkgs/* 2>/dev/null || true

# Check available space
echo ""
echo "Available disk space:"
df -h / | grep -E "Filesystem|overlay"

# Update package lists
echo ""
echo "[1/4] Updating package lists..."
apt-get update

# Install system dependencies
echo ""
echo "[2/4] Installing system dependencies..."
apt-get install -y gnupg2 ca-certificates

# Add graph-tool repository
echo ""
echo "Adding graph-tool repository for Ubuntu 24.04 (noble)..."
if ! grep -q "downloads.skewed.de" /etc/apt/sources.list; then
    echo "deb [trusted=yes] https://downloads.skewed.de/apt noble main" >> /etc/apt/sources.list
fi
apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 || true

# Update again after adding repository
apt-get update

# Install required system packages
echo ""
echo "Installing required system packages..."
apt-get install -y \
    python3-matplotlib \
    python3-graph-tool \
    python3-pip \
    libcairo2 \
    libcairo2-dev \
    libpython3-dev \
    libboost-all-dev

# Clean up after apt installations
apt-get clean
rm -rf /var/cache/apt/archives/*

# CUDA toolkit installation DISABLED to save disk space (~7GB)
# Root filesystem has limited space - PyTorch has built-in CUDA support
echo ""
echo "CUDA toolkit installation SKIPPED - PyTorch has built-in CUDA support"
echo "Root filesystem usage: $(df -h / | grep overlay | awk '{print $5}')"

# Install Python packages using system pip
echo ""
echo "[3/4] Installing Python packages to system Python3..."
echo "Using pip cache in workspace: $PIP_CACHE_DIR"

# Install packages one by one to handle dependencies properly
# NOTE: Changed numpy from 1.24.4 to 1.26.0+ for Python 3.12 compatibility
echo "Installing NumPy..."
/usr/bin/pip3 install --break-system-packages --no-cache-dir numpy>=1.26.0

echo "Installing PyTorch..."
/usr/bin/pip3 install --break-system-packages --no-cache-dir torch==2.2.0

echo "Installing torchdata (compatible version for DGL 2.3.0)..."
# DGL 2.1.0 requires torchdata 0.6.x or 0.7.x, NOT 0.11.x
/usr/bin/pip3 install --break-system-packages --no-cache-dir 'torchdata<0.8,>=0.6'

echo "Installing PyYAML (required by DGL)..."
/usr/bin/pip3 install --break-system-packages --no-cache-dir PyYAML

echo "Installing DGL..."
# /usr/bin/pip3 install --break-system-packages --no-cache-dir dgl==2.3.0
# /usr/bin/pip3 install --break-system-packages --no-cache-dir 'dgl-cu121==2.3.0' -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
/usr/bin/pip3 install --break-system-packages dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

echo "Installing remaining packages..."
/usr/bin/pip3 install --break-system-packages --no-cache-dir \
    pycairo \
    pandas \
    scikit-learn \
    pydantic

echo "Installing snakeviz for profiling (optional)..."
/usr/bin/pip3 install --break-system-packages --no-cache-dir snakeviz

# Clean pip cache in root after installation
rm -rf /root/.cache/pip 2>/dev/null || true

# Fix graph-tool library conflict with PyTorch
echo ""
echo "Fixing graph-tool library conflict..."
if [ -f /usr/local/lib/python3.12/dist-packages/torch/lib/libgomp-a34b3233.so.1 ]; then
    echo "Setting up library preload for graph-tool compatibility..."
    # Add to bashrc for persistent fix
    if ! grep -q "LD_PRELOAD.*libgomp" /root/.bashrc 2>/dev/null; then
        echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1' >> /root/.bashrc
    fi
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
fi

# Add conda deactivation to bashrc
if ! grep -q "conda deactivate" /root/.bashrc 2>/dev/null; then
    echo 'conda deactivate 2>/dev/null || true' >> /root/.bashrc
fi

# Clean up workspace cache to free root filesystem
echo "Cleaning workspace cache..."
rm -rf /workspace/.pip-cache/* 2>/dev/null || true

# Verify installations
echo ""
echo "[4/4] Verifying installations..."
echo "=============================================="

# Check Python version
echo ""
echo "Python version:"
/usr/bin/python3 --version

# Verify system packages
echo ""
echo "Checking system packages..."
SYSTEM_PACKAGES=(
    "python3-matplotlib"
    "python3-graph-tool"
    "python3-pip"
    "libcairo2"
    "libcairo2-dev"
)

for package in "${SYSTEM_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $package"; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is NOT installed"
    fi
done

# Verify Python packages
echo ""
echo "Checking Python packages..."
PYTHON_PACKAGES=(
    "torch"
    "dgl"
    "cairo"
    "pandas"
    "sklearn"
    "numpy"
    "pydantic"
    "torchdata"
)

# Verify snakeviz installation
if /usr/bin/python3 -c "import snakeviz" 2>/dev/null; then
    echo "✓ snakeviz is installed"
else
    echo "✗ snakeviz is NOT installed"
fi

for package in "${PYTHON_PACKAGES[@]}"; do
    if /usr/bin/python3 -c "import $package" 2>/dev/null; then
        VERSION=$(/usr/bin/python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ $package (version: $VERSION) is installed"
    else
        echo "✗ $package is NOT installed"
    fi
done

# Special check for graph_tool (different import name, needs LD_PRELOAD)
echo ""
echo "Checking graph-tool..."
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
if /usr/bin/python3 -c "import graph_tool" 2>/dev/null; then
    GT_VERSION=$(/usr/bin/python3 -c "import graph_tool; print(graph_tool.__version__)" 2>/dev/null || echo "unknown")
    echo "✓ graph_tool (version: $GT_VERSION) is installed"
else
    echo "✗ graph_tool is NOT installed"
fi

# Detailed version check
echo ""
echo "=============================================="
echo "Detailed Package Versions:"
echo "=============================================="
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
/usr/bin/python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch: NOT INSTALLED ({e})')

try:
    import dgl
    print(f'DGL: {dgl.__version__}')
except ImportError as e:
    print(f'DGL: NOT INSTALLED ({e})')

try:
    import torchdata
    print(f'TorchData: {torchdata.__version__}')
except ImportError as e:
    print(f'TorchData: NOT INSTALLED ({e})')

try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'NumPy: NOT INSTALLED ({e})')

try:
    import pandas
    print(f'Pandas: {pandas.__version__}')
except ImportError as e:
    print(f'Pandas: NOT INSTALLED ({e})')

try:
    import sklearn
    print(f'Scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'Scikit-learn: NOT INSTALLED ({e})')

try:
    import pydantic
    print(f'Pydantic: {pydantic.__version__}')
except ImportError as e:
    print(f'Pydantic: NOT INSTALLED ({e})')

try:
    import cairo
    print(f'PyCairo: {cairo.version}')
except ImportError as e:
    print(f'PyCairo: NOT INSTALLED ({e})')

try:
    import graph_tool
    print(f'Graph-tool: {graph_tool.__version__}')
except ImportError as e:
    print(f'Graph-tool: NOT INSTALLED ({e})')
"

echo ""
echo "=============================================="
echo "Final disk usage:"
df -h / | grep -E "Filesystem|overlay"
echo ""
echo "IMPORTANT NOTES:"
echo "1. Always deactivate conda: conda deactivate"
echo "2. To use graph-tool, run: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1"
echo "3. Both have been added to /root/.bashrc"
echo "=============================================="
echo "Installation and verification complete!"
echo "=============================================="