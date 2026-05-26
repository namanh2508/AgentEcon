#!/bin/bash
# AgentEcon - Quick Start Script
# This script helps you get started with the AgentEcon experiment system

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="/root/agentecon"

echo "========================================"
echo "AgentEcon - Setting Up Your Environment"
echo "========================================"

# Check if running in WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    echo "[OK] Running in WSL"
else
    echo "[WARN] Not running in WSL. Some features may not work."
fi

# Create symbolic link to project in WSL home
echo ""
echo "[1/5] Setting up project directory..."
if [ ! -d "$WSL_DIR" ]; then
    ln -s "$PROJECT_DIR" "$WSL_DIR"
    echo "Created symlink: $WSL_DIR -> $PROJECT_DIR"
else
    echo "Symlink already exists"
fi

# Create Python virtual environment
echo ""
echo "[2/5] Creating Python virtual environment..."
cd "$WSL_DIR"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
else
    echo "Virtual environment exists"
fi

# Install dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy statsmodels yfinance pyyaml requests matplotlib seaborn

echo "Core dependencies installed"
echo "Note: For GPU support, install vllm and torch separately"

# Download data
echo ""
echo "[4/5] Downloading financial data..."
source .venv/bin/activate
python scripts/download_data.py --n-queries 10

# Run smoke test
echo ""
echo "[5/5] Running smoke test..."
source .venv/bin/activate
python -c "
import sys
sys.path.insert(0, '.')
from src.data.asset_loader import AssetLoader
from src.models.predictors import LogitQRESampler
from src.oracle.fabric_oracle import FabricOracleClient

print('Testing components...')

# Test AssetLoader
loader = AssetLoader('data')
print('  AssetLoader: OK')

# Test QRE Sampler
qre = LogitQRESampler(grid_size=1024, tau=1.0)
qre.set_seed(1234)
sample = qre.sample([100.0, 101.0, 102.0])
print(f'  QRE Sampler: OK (sample bin={sample})')

# Test Oracle
oracle = FabricOracleClient()
oracle.submit_query('test', 'XAU_USD', 1850.50, 512)
oracle.submit_prediction('test', 'validator_0', 512)
print('  Oracle Client: OK')

print('')
print('All smoke tests passed!')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Download full data: python scripts/download_data.py"
echo "  3. Run experiments: make run-pilot"
echo "  4. Generate figures: make figures"
echo ""
echo "For help: make help"
