# Installation

This guide covers the installation and setup of NullStrike for structural identifiability analysis.

## Prerequisites

NullStrike requires Python 3.8 or higher and has the following dependencies:

### System Requirements

- **Python**: 3.8+
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 4GB minimum (8GB+ recommended for large models)
- **Storage**: 500MB for installation + space for results

### Required Python Packages

The core dependencies include:

- **SymPy**: Symbolic mathematics and computation
- **NumPy**: Numerical arrays and linear algebra  
- **Matplotlib**: Visualization and plotting
- **NetworkX**: Graph analysis and visualization
- **SciPy**: Scientific computing utilities

## Installation Methods

### Method 1: Development Installation (Recommended)

For development and customization, install from the source repository:

```bash
# Clone the repository
git clone https://github.com/vipulsinghal02/NullStrike.git
cd NullStrike

# Install in development mode
pip install -e .
```

This method allows you to:

- Modify the source code and see changes immediately
- Add custom models and options easily
- Contribute to the project development
- Access the latest features and bug fixes

### Method 2: Direct Installation

For production use or if you don't need to modify the code:

```bash
# Install directly from the repository
pip install git+https://github.com/vipulsinghal02/NullStrike.git
```

### Method 3: Local Installation

If you have a local copy of the source code:

```bash
cd /path/to/NullStrike
pip install .
```

## Verification

After installation, verify that NullStrike is working correctly:

### 1. Check Installation

```bash
# Verify the command is available
nullstrike --help
```

Expected output:
```
usage: nullstrike [-h] [--parameters-only] model_name [options_file]

NullStrike: Structural identifiability analysis with nullspace analysis

positional arguments:
  model_name         Name of the model to analyze
  options_file       Configuration file (optional)

optional arguments:
  -h, --help         Show this help message and exit
  --parameters-only  Run parameters-only analysis
```

### 2. Run Test Analysis

Test with a built-in example:

```bash
# Run analysis on the two-compartment model
nullstrike C2M
```

This should:

1. Create a `results/` directory
2. Generate analysis files including visualizations
3. Display progress information in the terminal
4. Complete without errors

### 3. Check Python API

Test the Python interface:

```python
# Test in Python interpreter
from nullstrike.cli.complete_analysis import main

# This should work without import errors
print("NullStrike successfully installed!")
```

## Troubleshooting

### Common Installation Issues

=== "Import Errors"

    **Problem**: `ModuleNotFoundError` when importing NullStrike
    
    **Solutions**:
    ```bash
    # Ensure you're in the correct environment
    which python
    which pip
    
    # Reinstall in development mode
    pip uninstall nullstrike
    pip install -e .
    ```

=== "Permission Errors"

    **Problem**: Permission denied during installation
    
    **Solutions**:
    ```bash
    # Use user installation
    pip install --user -e .
    
    # Or use virtual environment (recommended)
    python -m venv nullstrike_env
    source nullstrike_env/bin/activate  # Linux/macOS
    # nullstrike_env\Scripts\activate   # Windows
    pip install -e .
    ```

=== "SymPy Computation Issues"

    **Problem**: Slow symbolic computation or memory errors
    
    **Solutions**:
    - Increase system RAM or use smaller models for testing
    - Check that SymPy version is compatible (≥1.8)
    - For large models, consider using `--parameters-only` flag

=== "Visualization Issues"

    **Problem**: Matplotlib plots not displaying or saving
    
    **Solutions**:
    ```bash
    # For headless systems, ensure proper backend
    export MPLBACKEND=Agg
    
    # Install additional backends if needed
    pip install PyQt5  # or tkinter support
    ```

### Platform-Specific Notes

=== "macOS"

    ```bash
    # If using Homebrew Python
    brew install python@3.9
    
    # Ensure pip is up to date
    python -m pip install --upgrade pip
    ```

=== "Ubuntu/Debian"

    ```bash
    # Install Python development headers
    sudo apt update
    sudo apt install python3-dev python3-pip
    
    # Install build tools if needed
    sudo apt install build-essential
    ```

=== "Windows"

    Use Windows Subsystem for Linux (WSL) or ensure:
    ```cmd
    # Install Visual C++ Build Tools if compilation errors occur
    # Use Anaconda/Miniconda for easier dependency management
    conda install sympy numpy matplotlib networkx scipy
    pip install -e .
    ```

## Virtual Environment Setup (Recommended)

For isolated installation:

```bash
# Create virtual environment
python -m venv nullstrike_env

# Activate environment
source nullstrike_env/bin/activate  # Linux/macOS
# nullstrike_env\Scripts\activate   # Windows

# Install NullStrike
pip install -e .

# Deactivate when done
deactivate
```

## Development Dependencies

For contributing to NullStrike development:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or manually install testing/docs tools
pip install pytest pytest-cov sphinx mkdocs mkdocs-material
```

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)**: Get started with immediate examples
2. **[First Analysis](quickstart.md)**: Step-by-step tutorial
4. **[Examples](examples.md)**: Explore practical use cases

## Getting Help

If you encounter issues:

- Check the [Troubleshooting section](#troubleshooting) above
- Review the [GitHub Issues](https://github.com/vipulsinghal02/NullStrike/issues)
- Ask questions in [GitHub Discussions](https://github.com/vipulsinghal02/NullStrike/discussions)
- Contact the maintainers

---

!!! success "Installation Complete"
    
    If you can run `nullstrike C2M` successfully, you're ready to start analyzing your own models!