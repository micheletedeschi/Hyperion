#!/bin/bash
# Installation script for Hyperion V2 on Apple Silicon Mac (M4)

echo "🚀 Hyperion V2 Installation for Apple Silicon Mac M4"
echo "=================================================="

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only!"
    echo "   Detected architecture: $(uname -m)"
    exit 1
fi

echo "✅ Apple Silicon detected: $(uname -m)"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python 3.9+ required. Current version: $PYTHON_VERSION"
    echo "   Please install Python 3.9+ first."
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install MLX (Apple's ML framework)
echo ""
echo "🍎 Installing MLX for Apple Silicon optimization..."
pip install mlx mlx-lm

# Install PyTorch with MPS support
echo ""
echo "🔥 Installing PyTorch with Metal Performance Shaders support..."
pip install torch torchvision torchaudio

# Install TA-Lib (requires brew)
echo ""
echo "📊 Installing TA-Lib..."
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

brew install ta-lib
export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
pip install ta-lib

# Install main requirements
echo ""
echo "📦 Installing main requirements..."
pip install -r requirements.txt

# Install additional Apple Silicon optimized packages
echo ""
echo "🍎 Installing additional optimizations..."
pip install tensorflow-macos  # Optional, for ensemble models
pip install tensorflow-metal   # GPU acceleration for TensorFlow

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data models logs checkpoints optimization_results backtest_results plots

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cat > .env << EOL
# Exchange API Configuration
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_SECRET=your_secret_key_here

# Optional: Other API Keys
ALPHAVANTAGE_API_KEY=
NEWSAPI_KEY=

# System Configuration
DEBUG=False
LOG_LEVEL=INFO

# Apple Silicon Optimization
USE_MLX=True
USE_MPS=True
EOL
    echo "⚠️  Please edit .env file with your API keys!"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "
import torch
import mlx
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ MPS available: {torch.backends.mps.is_available()}')
print(f'✅ MLX installed successfully')

# Test imports
try:
    import flaml
    print('✅ FLAML installed')
except:
    print('❌ FLAML not installed')

try:
    import ray
    print('✅ Ray installed')
except:
    print('❌ Ray not installed')

try:
    import stable_baselines3
    print('✅ Stable-Baselines3 installed')
except:
    print('❌ Stable-Baselines3 not installed')
"

echo ""
echo "✨ Installation complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Hyperion V2, run:"
echo "  python main.py"
echo ""
echo "🚀 Happy trading!"