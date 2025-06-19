from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperion3",
    version="3.0.0",
    packages=find_packages(),
    install_requires=[
        # Core Dependencies 
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        
        # Deep Learning Frameworks
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        
        # Reinforcement Learning
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.28.0",
        
        # Time Series & Financial Analysis
        "ta>=0.10.2",
        "yfinance>=0.2.18",
        "ccxt>=4.0.0",
        "python-binance>=1.0.17",
        
        # AutoML & Optimization
        "flaml>=2.0.0",
        "optuna>=3.2.0",
        
        # Classic ML Models
        "lightgbm>=4.0.0",
        "xgboost>=1.7.0",
        "catboost>=1.2.2",
        
        # Transformers & Advanced Models
        "transformers>=4.30.0",
        "einops>=0.6.1",
        
        # Visualization & Monitoring
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "rich>=13.3.0",
        
        # MLOps & Experiment Tracking
        "mlflow>=2.4.0",
        "tensorboard>=2.13.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "requests>=2.31.0",
        
        # Development & Testing
        "pytest>=7.3.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    python_requires=">=3.8",
    author="Hyperion Team",
    author_email="HyperionGanador@proton.me",
    description="Hyperion3 - Framework de Trading Algorítmico Forjado en la Obsesión",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ganador1/Hyperion",
    project_urls={
        "Bug Reports": "https://github.com/Ganador1/Hyperion/issues",
        "Documentation": "https://github.com/Ganador1/Hyperion/docs",
        "Source": "https://github.com/Ganador1/Hyperion",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="trading, algorithmic-trading, machine-learning, deep-learning, reinforcement-learning, cryptocurrency, finance, automl, transformers",
    entry_points={
        "console_scripts": [
            "hyperion=main:main",
            "hyperion-professional=main_professional:main",
        ],
    },
)
