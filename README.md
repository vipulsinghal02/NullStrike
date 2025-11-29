# NullStrike

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-green.svg)](https://vipulsinghal02.github.io/NullStrike/)

**NullStrike** is a structural identifiability analysis tool that extends StrikePy (Python implementation of STRIKE-GOLDD) with **nullspace analysis** capabilities. In addition to identifying individual parameters and states that are unidentifiable, NullStrike determines which **parameter combinations** are unidentifiable. 

<!-- ## What Makes NullStrike Different

Traditional structural identifiability analysis tells you *which parameters cannot be identified*. NullStrike goes further by discovering *which parameter combinations CAN be identified*, providing actionable insights for experimental design and parameter estimation. -->

### Nullspace Analysis

The mathematical foundation is:

```
Observability Matrix: O = [h, Lf h, Lf² h, ..., Lfⁿ h]ᵀ
Nullspace: N = nullspace(O) ← unidentifiable directions  
Identifiable Combinations: I = nullspace(N) ← what you CAN estimate
```

If the nullspace has dimension *k*, then *(total_parameters - k)* independent parameter combinations are identifiable. 

## Quick Start

### Installation

```bash
# Clone and install in development mode
git clone https://github.com/vipulsinghal02/NullStrike.git
cd NullStrike
pip install -e .
```

### Usage

```bash
# Analyze built-in models
nullstrike C2M                    # Two-compartment pharmacokinetic model
nullstrike Bolie                  # Bolie glucose-insulin model  
nullstrike calibration_single     # Calibration example

# Custom analysis
nullstrike my_model options_my_model
```

### Python API

```python
from nullstrike.cli.complete_analysis import main

# Run complete analysis
results = main('C2M', 'options_C2M')
# Results include parameter combinations, visualizations, and detailed reports
```

## Example Results

### Two-Compartment Pharmacokinetic Model

For a typical PK model with parameters `k12`, `k21`, `V1`, `V2`:

**Traditional Analysis**: *"All parameters are unidentifiable"*

**NullStrike Analysis**: 

- **Identifiable combinations**: `k12×V1`, `k21×V2`, `(k12+k21+k10)×V1`

- **Nullspace dimension**: 1 (out of 4 parameters)

- **3 independent combinations** can be reliably estimated

### Visualization Outputs

NullStrike generates comprehensive visual analysis:

- **3D Manifold Plots**: Parameter constraint surfaces

- **2D Projections**: Pairwise parameter relationships  

- **Graph Networks**: Parameter dependency visualization

- **Detailed Reports**: Mathematical interpretations

## Mathematical Foundation

NullStrike implements structural identifiability analysis using:

### STRIKE-GOLDD Algorithm
Computes the observability-identifiability matrix using Lie derivatives:

$$\mathcal{O} = \begin{bmatrix} 
\mathcal{L}_f^0 h \\
\mathcal{L}_f^1 h \\
\vdots \\
\mathcal{L}_f^n h 
\end{bmatrix}$$

### Nullspace Analysis  
Identifies unidentifiable and identifiable directions:

$$\begin{align}
\mathcal{N} &= \text{nullspace}(\mathcal{O}) \quad \text{(unidentifiable directions)} \\
\mathcal{I} &= \text{nullspace}(\mathcal{N}) \quad \text{(identifiable combinations)}
\end{align}$$

### Symbolic Computation
Symbolic analysis ensures exact mathematical relationships without numerical approximation errors.

## Project Structure

```
NullStrike/
├── src/nullstrike/
│   ├── core/              # Original StrikePy functionality
│   ├── analysis/          # Enhanced nullspace analysis  
│   ├── visualization/     # 3D manifolds and graphs
│   └── cli/              # Command-line interface
├── custom_models/         # User-defined dynamical systems
├── custom_options/        # Model-specific analysis options  
├── results/              # Generated analysis outputs
└── docs/                 # Comprehensive documentation
```

## Model Definition

Define your dynamical system in `custom_models/my_model.py`:

```python
import sympy as sym

# States  
x1, x2 = sym.symbols('x1 x2')
x = [[x1], [x2]]

# Parameters
p1, p2, p3 = sym.symbols('p1 p2 p3') 
p = [[p1], [p2], [p3]]

# Outputs (measurements)
h = [x1]

# Known inputs
u1 = sym.Symbol('u1')
u = [u1]

# Dynamics dx/dt = f
f = [[p1*x1 + p2*x2 + u1], [-p3*x1]]

# Required for StrikePy
variables_locales = locals().copy()
```

Configure analysis in `custom_options/options_my_model.py`:

```python
modelname = 'my_model'
checkObser = 1           # Check state observability 
maxLietime = 100         # Max time per Lie derivative (seconds)
nnzDerU = [0]           # Known input derivative limits
prev_ident_pars = []    # Previously identified parameters
```

## Advanced Features

- **Checkpointing System**: Intelligent caching avoids recomputation

- **Batch Processing**: Analyze multiple models efficiently  

- **Parameter Manifold Visualization**: 3D constraint surfaces

- **Graph Analysis**: Network representation of dependencies

- **Comprehensive Reports**: Mathematical interpretations and actionable insights

## Documentation

**[Full Documentation](https://vipulsinghal02.github.io/NullStrike/)** includes:

- **[Quick Start](https://vipulsinghal02.github.io/NullStrike/quickstart/)**: Get started in 5 minutes
- **[Installation](https://vipulsinghal02.github.io/NullStrike/installation/)**: Setup instructions
- **[Theory](https://vipulsinghal02.github.io/NullStrike/theory/)**: Mathematical foundations
- **[Examples](https://vipulsinghal02.github.io/NullStrike/examples/)**: Step-by-step tutorials
<!-- - **[Reference](https://vipulsinghal02.github.io/NullStrike/reference/)**: Complete API and CLI reference -->
<!-- - **[Contributing](https://vipulsinghal02.github.io/NullStrike/contributing/)**: Development guide -->

## Development

### Running Tests

```bash
pytest                    # Run test suite
pytest --cov=nullstrike  # With coverage reporting
```

### Code Quality

```bash
black src/               # Format code
flake8 src/              # Lint code  
mypy src/                # Type checking
```

### Building Documentation

```bash
pip install -e ".[docs]"
mkdocs serve            # Local documentation server
mkdocs build            # Build static site
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://vipulsinghal02.github.io/NullStrike/dev/contributing/) for details on:

- Setting up the development environment

- Code style and testing requirements

- Submitting pull requests

- Reporting issues

## License

NullStrike is released under the [GNU General Public License v3.0](LICENSE).

The core STRIKE-GOLDD algorithm implementation is adapted from StrikePy, which implements the MATLAB STRIKE-GOLDD toolbox in Python. See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed attribution.

## Citation

If you use NullStrike in your research, please cite:

```bibtex
@software{nullstrike2025,
  title = {NullStrike: Enhanced Structural Identifiability Analysis with Nullspace Parameter Combinations},
  author = {Vipul Singhal},
  year = {2025},
  url = {https://github.com/vipulsinghal02/NullStrike}
}
```

## Getting Help

- **Documentation**: [https://vipulsinghal02.github.io/NullStrike/](https://vipulsinghal02.github.io/NullStrike/)

- **Issues**: [GitHub Issues](https://github.com/vipulsinghal02/NullStrike/issues)

- **Discussions**: [GitHub Discussions](https://github.com/vipulsinghal02/NullStrike/discussions)

---


**NullStrike**: *From "parameters are unidentifiable" to "these combinations ARE identifiable"*
