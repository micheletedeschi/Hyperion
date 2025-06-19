# Installation Guide

This document explains how to set up Hyperion V2 on your system.

## Prerequisites

- Python 3.9 or newer
- Recommended: create a virtual environment

## Quick Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the main dependencies:
   ```bash
   pip install -r requirements.txt
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

