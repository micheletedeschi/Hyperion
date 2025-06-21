# 🚀 Installation Guide - Hyperion3

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Platform-Specific Instructions](#platform-specific-instructions)
4. [Development Installation](#development-installation)
5. [Docker Installation](#docker-installation)
6. [Troubleshooting](#troubleshooting)
7. [Verification](#verification)

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for data and models
- **OS**: Linux, macOS, or Windows 10/11

### Recommended Requirements
- **Python**: 3.10 or 3.11
- **RAM**: 32GB for large model training
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD with 20GB+ free space
- **OS**: Ubuntu 20.04+ or macOS 12+

### Supported Platforms
- ✅ **macOS**: Intel and Apple Silicon (M1/M2/M3)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- ✅ **Windows**: Windows 10/11 with WSL2 recommended

## Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/hyperion3.git
cd hyperion3
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv hyperion_env
source hyperion_env/bin/activate  # On Windows: hyperion_env\Scripts\activate

# Or using conda
conda create -n hyperion python=3.10
conda activate hyperion
```

### 3. Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt

# Or minimal installation (core features only)
pip install -r requirements-minimal.txt
```

### 4. Verify Installation
```bash
python main.py --help
```

## Platform-Specific Instructions

### 🍎 macOS Installation

#### Apple Silicon (M1/M2/M3)
```bash
# Use the optimized installation script
chmod +x install_mac.sh
./install_mac.sh
```

#### Intel Mac
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.11
pip3 install -r requirements.txt
```

#### Additional macOS Dependencies
```bash
# Install TA-Lib (Technical Analysis Library)
brew install ta-lib

# Install other system dependencies
brew install cmake pkg-config
```

### 🐧 Linux Installation

#### Ubuntu/Debian
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y build-essential cmake
sudo apt install -y libta-lib-dev

# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Install Python dependencies
pip3 install -r requirements.txt
```

#### CentOS/RHEL/Fedora
```bash
# Install system dependencies
sudo dnf install -y python3-devel python3-pip
sudo dnf install -y gcc gcc-c++ make cmake
sudo dnf install -y ta-lib-devel

# Install Python dependencies
pip3 install -r requirements.txt
```

### 🪟 Windows Installation

#### Using WSL2 (Recommended)
```bash
# Install WSL2 and Ubuntu
wsl --install

# Follow Linux installation instructions inside WSL2
```

#### Native Windows
```powershell
# Install Python 3.11 from python.org
# Install Microsoft C++ Build Tools
# Install Git for Windows

# Clone and install
git clone https://github.com/your-username/hyperion3.git
cd hyperion3
python -m venv hyperion_env
hyperion_env\Scripts\activate
pip install -r requirements.txt
```

## Development Installation

### For Contributors and Developers

```bash
# Clone the repository
git clone https://github.com/your-username/hyperion3.git
cd hyperion3

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking
- `jupyter`: Notebook support
- `pre-commit`: Git hooks

## Docker Installation

### Using Docker Compose (Recommended)
```bash
# Clone repository
git clone https://github.com/your-username/hyperion3.git
cd hyperion3

# Build and run with Docker Compose
docker-compose up --build
```

### Manual Docker Build
```bash
# Build the image
docker build -t hyperion3 .

# Run the container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  hyperion3
```

### Docker with GPU Support
```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t hyperion3-gpu .

# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  hyperion3-gpu
```

## Environment Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# MLOps Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your_wandb_key_here

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### Configuration Files
Copy and modify configuration templates:

```bash
# Copy example configuration
cp config.json.example config.json

# Edit configuration
nano config.json
```

## Troubleshooting

### Common Issues and Solutions

#### 1. TA-Lib Installation Issues
```bash
# If TA-Lib fails to install
pip install --upgrade setuptools wheel
pip install --no-cache-dir TA-Lib

# Alternative: use conda
conda install -c conda-forge ta-lib
```

#### 2. PyTorch Installation Issues
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Apple Silicon Issues
```bash
# If you encounter architecture issues
arch -arm64 pip install package_name

# Or use Rosetta
arch -x86_64 pip install package_name
```

#### 4. Memory Issues
```bash
# Reduce batch size in config.json
{
  "training": {
    "batch_size": 16  # Reduce from 32 or 64
  }
}
```

#### 5. Permission Issues (Linux/macOS)
```bash
# Fix permissions
sudo chown -R $USER:$USER /path/to/hyperion3
chmod +x install_mac.sh
```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Clean installation
pip uninstall -y -r requirements.txt
pip install --no-cache-dir -r requirements.txt

# Or use pip-tools
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

### Performance Issues

#### GPU not detected
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# For Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### Slow training
- Reduce batch size
- Use fewer features
- Enable GPU acceleration
- Use faster models (LightGBM, XGBoost)

## Verification

### Basic Functionality Test
```bash
# Test basic installation
python -c "import hyperion3; print('✅ Hyperion3 installed successfully')"

# Test professional interface
python main_professional.py

# Run minimal example
python example_minimal.py
```

### Model Testing
```bash
# Test individual models
python -m pytest tests/test_models.py

# Test specific model
python test_sac_fix.py
```

### Data Pipeline Test
```bash
# Test data downloader
python -c "from hyperion3.data.downloader import DataDownloader; print('✅ Data pipeline OK')"

# Test feature engineering
python -c "from hyperion3.data.feature_engineering import FeatureEngineer; print('✅ Features OK')"
```

### Complete System Test
```bash
# Run integration tests
python -m pytest tests/

# Run example experiment
python scripts/run_experiment.py
```

## Post-Installation Setup

### 1. Download Sample Data
```bash
# Download sample cryptocurrency data
python scripts/download_sample_data.py
```

### 2. Configure API Keys
```bash
# Add your exchange API keys to .env file
echo "BINANCE_API_KEY=your_key" >> .env
echo "BINANCE_SECRET_KEY=your_secret" >> .env
```

### 3. Initialize MLOps
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Initialize tracking database
python scripts/init_mlops.py
```

### 4. Run First Experiment
```bash
# Run a simple experiment
python main.py

# Or use the professional interface
python main_professional.py
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the Issues**: Look at [GitHub Issues](https://github.com/your-username/hyperion3/issues)
2. **Documentation**: Read the complete documentation in `/docs`
3. **Community**: Join our community discussions
4. **Support**: Contact the development team

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](QUICKSTART.md)
2. Review the [Architecture Guide](ARCHITECTURE.md)
3. Explore the [API Reference](API_REFERENCE.md)
4. Try the [Example Notebooks](../examples/)
5. Join the [Development Guide](DEVELOPMENT_GUIDE.md) for contributing

---

**🎉 Congratulations! You now have Hyperion3 installed and ready to use.**

For your first trading strategy, see [QUICKSTART.md](QUICKSTART.md) or run:
```bash
python main_professional.py
```
3. (Optional) Install the lighter test requirements:
   ```bash
   pip install -r requirements-test.txt
   ```

## Apple Silicon

Mac users with Apple Silicon can run the helper script which installs MLX and other optimisations:

```bash
bash install_mac.sh
```

The script creates a `.env` file and verifies the environment. Edit that file with your API keys after the installation completes.

## Manual `.env` file

If you set up the environment manually, create a `.env` file in the project root with your API credentials:

```
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_SECRET=your_secret_key_here
```

Additional keys (e.g. `ALPHAVANTAGE_API_KEY`) can be added as needed.

## Verifying the installation

Run the test suite to make sure the environment works:

```bash
pytest
```

If some packages are missing you can install them using `requirements-test.txt` as shown above.

