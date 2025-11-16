# Configuration Files

Configuration files control how NullStrike analyzes your models. This guide explains all configuration options and how to customize the analysis for your specific needs.

## Configuration File Structure

Configuration files are Python files located in the `custom_options/` directory. They must be named with the pattern `options_[model_name].py`.

### Basic Template

```python
# ===========================================================================
# CONFIGURATION FILE FOR [MODEL_NAME]
# ===========================================================================

import sympy as sym
from math import inf

# ---------------------------------------------------------------------------
# (1) MODEL IDENTIFICATION
# ---------------------------------------------------------------------------
modelname = 'your_model_name'  # Must match model file name

# ---------------------------------------------------------------------------
# (2) ANALYSIS OPTIONS  
# ---------------------------------------------------------------------------
checkObser = 1                 # Check state observability (1=yes, 0=no)
maxLietime = inf              # Max time for Lie derivative computation (seconds)
nnzDerU = [0]                 # Non-zero known input derivatives
nnzDerW = [inf]               # Non-zero unknown input derivatives

# ---------------------------------------------------------------------------
# (3) KNOWN/IDENTIFIABLE PARAMETERS
# ---------------------------------------------------------------------------
prev_ident_pars = []          # Previously identified parameters

# ---------------------------------------------------------------------------
# (4) VISUALIZATION CONFIGURATION
# ---------------------------------------------------------------------------
MANIFOLD_PLOTTING = {
    # Configuration dictionary (detailed below)
}
```

## Core Analysis Options

### Model Name

```python
modelname = 'calibration_single'
```

**Purpose**: Identifies which model file to analyze.

**Rules**:
- Must exactly match the filename in `custom_models/` (without `.py`)
- Case-sensitive
- No spaces or special characters

### Observability Analysis

```python
checkObser = 1  # or 0
```

**Options**:
- `1`: Analyze state observability (can initial conditions be determined?)
- `0`: Skip state observability analysis

**Recommendation**: Use `1` unless you're only interested in parameter identifiability.

### Computation Time Limits

```python
maxLietime = inf  # No time limit
# or
maxLietime = 300  # 5 minutes maximum per Lie derivative
```

**Purpose**: Prevents infinite loops in symbolic computation.

**Guidelines**:
- `inf`: No limit (use for small/medium models)
- `60-600`: Reasonable limits for larger models (seconds)
- `10-30`: Quick testing during development

### Input Derivative Information

```python
nnzDerU = [0]      # No derivatives of known inputs
nnzDerW = [inf]    # Unlimited derivatives of unknown inputs
```

**`nnzDerU`**: Number of non-zero derivatives for each known input
- `[0]`: No input derivatives used
- `[1]`: First derivative available (velocity)
- `[2]`: Up to second derivative available (acceleration)
- `[0, 1]`: Different limits for multiple inputs

**`nnzDerW`**: Number of non-zero derivatives for unknown inputs
- `[inf]`: No limit (most common)
- `[0]`: Unknown inputs are constant
- `[2]`: Up to second derivatives

### Prior Knowledge

```python
# No prior knowledge
prev_ident_pars = []

# Some parameters known to be identifiable
x3 = sym.Symbol('x3')
x5 = sym.Symbol('x5')
prev_ident_pars = [x3, x5]
```

**Purpose**: Specify parameters already known to be identifiable from prior analysis.

**Use cases**:
- Parameters identified in previous studies
- Parameters measured directly
- Parameters fixed by experimental design

## Visualization Configuration

The `MANIFOLD_PLOTTING` dictionary controls all visualization aspects.

### Basic Configuration

```python
MANIFOLD_PLOTTING = {
    # Parameter ranges for plotting
    "var_ranges": {},
    
    # Parameters known to be positive
    "positive_symbols": [],
    
    # Default range for general parameters
    "default_var_range": (-5.0, 5.0, 100),
    
    # Default range for positive parameters
    "default_positive_range": (1e-3, 10.0, 120),
    
    # Use log scale for positive parameters
    "log_for_positive": True,
}
```

### Parameter Range Specification

#### Method 1: Simple Ranges

```python
"var_ranges": {
    "k1": (0.1, 10.0, 50),      # (min, max, num_points)
    "k2": (1e-4, 1e-1, 100),    # Linear spacing
}
```

#### Method 2: Advanced Ranges

```python
"var_ranges": {
    "k1": {
        "min": 0.1,
        "max": 10.0, 
        "num": 50,
        "scale": "linear"    # or "log"
    },
    "k2": {
        "min": 1e-4,
        "max": 1e-1,
        "num": 100,
        "scale": "log"
    }
}
```

### Positive Parameter Handling

```python
"positive_symbols": ["k1", "k2", "V1", "kcat"],
"default_positive_range": (1e-3, 10.0, 120),
"log_for_positive": True,
```

**Purpose**: Automatically use appropriate ranges and scaling for positive parameters.

**Benefits**:
- Log scaling reveals structure across orders of magnitude
- Avoids non-physical negative values
- Optimizes visualization for biological/chemical parameters

### Plot Customization

```python
MANIFOLD_PLOTTING = {
    # ... basic configuration ...
    
    # Number of z-slices for 3D plots
    "z_slices": 15,
    
    # Additional 3D combinations to plot
    "extra_triplets_3d": [
        ("k1", "k2", "V1"),      # Custom 3D plot
        ("kf", "kr", "kcat"),    # Another 3D plot
    ],
    
    # Additional 2D combinations to plot  
    "extra_pairs_2d": [
        ("k1", "k2"),            # Custom 2D plot
        ("V1", "kcat"),          # Another 2D plot
    ],
    
    # Limits on number of plots
    "max_triplets_3d": None,    # No limit
    "max_pairs_2d": 20,         # Maximum 20 2D plots
}
```

### Parameter Value Overrides

```python
"param_overrides": {
    "k3": 2.0,      # Fix k3 = 2.0 for visualization
    "V2": 0.5,      # Fix V2 = 0.5 for visualization
}
```

**Purpose**: Set fixed values for parameters not being plotted.

**Use cases**:
- Visualize subspaces of parameter space
- Use known parameter values from literature
- Focus on uncertain parameters

## Domain-Specific Configurations

### Pharmacokinetic Models

```python
modelname = 'two_compartment_pk'
checkObser = 1
maxLietime = 600  # PK models can be complex

MANIFOLD_PLOTTING = {
    "positive_symbols": [
        "k10", "k12", "k21",     # Rate constants
        "V1", "V2",              # Volumes  
        "CL", "Q"                # Clearances
    ],
    "var_ranges": {
        "k10": (0.01, 1.0, 50),     # Elimination rates
        "k12": (0.001, 0.1, 50),    # Distribution rates
        "V1": (1.0, 100.0, 50),     # Volumes (L)
    },
    "default_positive_range": (1e-3, 10.0, 100),
    "log_for_positive": True,
    "param_overrides": {
        "BW": 70.0,              # Body weight fixed at 70 kg
    }
}
```

### Chemical Reaction Systems

```python
modelname = 'enzyme_kinetics'
checkObser = 1
maxLietime = 300

MANIFOLD_PLOTTING = {
    "positive_symbols": [
        "kf", "kr", "kcat",      # Rate constants
        "Km", "Vmax",            # Michaelis-Menten parameters
        "E0", "S0"               # Initial concentrations
    ],
    "var_ranges": {
        "kf": {"min": 1e6, "max": 1e9, "num": 50, "scale": "log"},    # M^-1 s^-1
        "kr": {"min": 1e-2, "max": 1e2, "num": 50, "scale": "log"},   # s^-1  
        "kcat": {"min": 1e-2, "max": 1e2, "num": 50, "scale": "log"}, # s^-1
    },
    "log_for_positive": True,
    "extra_triplets_3d": [
        ("kf", "kr", "kcat"),    # Kinetic parameters
    ]
}
```

### Biological Systems

```python
modelname = 'gene_expression'
checkObser = 1
maxLietime = 180

MANIFOLD_PLOTTING = {
    "positive_symbols": [
        "k_transcr", "k_transl", # Synthesis rates
        "k_deg_m", "k_deg_p",    # Degradation rates
        "K_d", "n"               # Binding and cooperativity
    ],
    "var_ranges": {
        "k_transcr": (0.1, 100.0, 50),      # Transcription rate
        "k_deg_m": (0.01, 1.0, 50),         # mRNA degradation
        "n": (1.0, 4.0, 20),                # Hill coefficient
    },
    "log_for_positive": False,  # Some parameters better on linear scale
    "extra_pairs_2d": [
        ("k_transcr", "k_deg_m"),
        ("K_d", "n"),
    ]
}
```

## Advanced Configuration Options

### Multiple Experiment Analysis

```python
# For models with multiple experimental conditions
nnzDerU = [
    [0, 1],     # Experiment 1: no input, first derivative available
    [2, 0],     # Experiment 2: second derivative, no second input
]
nnzDerW = [
    [inf],      # Experiment 1: unlimited unknown input derivatives
    [0],        # Experiment 2: no unknown inputs
]
```

### Computational Optimization

```python
# For large models requiring optimization
maxLietime = 120          # Shorter time limit
checkObser = 0            # Skip observability if not needed

MANIFOLD_PLOTTING = {
    "max_triplets_3d": 5,    # Limit 3D plots to reduce computation
    "max_pairs_2d": 10,      # Limit 2D plots
    "z_slices": 8,           # Fewer slices for faster rendering
}
```

### Debugging Configuration

```python
# For debugging model issues
modelname = 'test_model'
checkObser = 1
maxLietime = 30           # Short limit to catch issues quickly

# No visualization during debugging
MANIFOLD_PLOTTING = {
    "max_triplets_3d": 0,    # No 3D plots
    "max_pairs_2d": 0,       # No 2D plots
}
```

## Configuration File Organization

### Naming Convention

```
custom_options/
├── options_C2M.py                    # Two-compartment model
├── options_calibration_single.py     # Single calibration model
├── options_enzyme_kinetics.py        # Enzyme model
├── options_my_custom_model.py        # Your model
└── options_default.py                # Default fallback
```

### Inheritance and Defaults

Create a base configuration for related models:

```python
# options_base_pk.py (base pharmacokinetic configuration)
base_pk_config = {
    "checkObser": 1,
    "maxLietime": 600,
    "positive_symbols": ["k10", "k12", "k21", "V1", "V2"],
    "default_positive_range": (1e-3, 10.0, 100),
    "log_for_positive": True,
}

# options_my_pk_model.py (inherits from base)
from .options_base_pk import base_pk_config

modelname = 'my_pk_model'
checkObser = base_pk_config["checkObser"]
maxLietime = base_pk_config["maxLietime"]

MANIFOLD_PLOTTING = {
    **base_pk_config,  # Inherit base configuration
    "var_ranges": {    # Add model-specific ranges
        "CL": (1.0, 20.0, 50),
        "Q": (0.1, 5.0, 50),
    }
}
```

## Troubleshooting Configuration Issues

### Common Problems

=== "Model Not Found"

    **Error**: `ImportError: No module named 'custom_models.my_model'`
    
    **Solutions**:
    - Check that `modelname` matches the file name exactly
    - Verify the model file exists in `custom_models/`
    - Ensure no typos in the model name

=== "Visualization Errors"

    **Error**: Empty plots or plotting failures
    
    **Solutions**:
    - Check parameter ranges are reasonable
    - Ensure positive parameters have positive ranges
    - Verify symbol names match model definition

=== "Computation Timeout"

    **Error**: Analysis hangs during Lie derivative computation
    
    **Solutions**:
    - Reduce `maxLietime` value
    - Simplify the model for testing
    - Use `--parameters-only` flag

### Validation Checklist

Before running analysis, verify:

- [ ] `modelname` matches model file exactly
- [ ] All parameter names in `prev_ident_pars` exist in model
- [ ] Parameter ranges in `var_ranges` are reasonable
- [ ] Positive parameters listed in `positive_symbols`
- [ ] Time limits appropriate for model complexity

## Configuration Examples

### Minimal Configuration

```python
# Simplest possible configuration
modelname = 'simple_model'
checkObser = 1
maxLietime = inf
nnzDerU = [0]
nnzDerW = [inf]
prev_ident_pars = []

MANIFOLD_PLOTTING = {
    "var_ranges": {},
    "positive_symbols": [],
    "default_var_range": (-5.0, 5.0, 100),
}
```

### Comprehensive Configuration

```python
# Full-featured configuration example
import sympy as sym
from math import inf

modelname = 'comprehensive_model'
checkObser = 1
maxLietime = 300
nnzDerU = [1, 0]  # Two inputs with different derivative availability
nnzDerW = [inf]

# Some parameters known to be identifiable
k1, V1 = sym.symbols('k1 V1')
prev_ident_pars = [k1, V1]

MANIFOLD_PLOTTING = {
    "var_ranges": {
        "k1": {"min": 0.1, "max": 10.0, "num": 50, "scale": "log"},
        "k2": (0.01, 1.0, 50),
        "V1": (10.0, 100.0, 30),
    },
    "positive_symbols": ["k1", "k2", "k3", "V1", "V2"],
    "default_positive_range": (1e-3, 10.0, 120),
    "log_for_positive": True,
    "default_var_range": (-10.0, 10.0, 100),
    "z_slices": 12,
    "extra_triplets_3d": [
        ("k1", "k2", "V1"),
        ("k2", "k3", "V2"),
    ],
    "extra_pairs_2d": [
        ("k1", "V1"),
        ("k2", "k3"),
    ],
    "max_triplets_3d": 10,
    "max_pairs_2d": 15,
    "param_overrides": {
        "fixed_param": 1.5,
    },
}
```

---

## Further Reading

- **[Model Definition Guide](models.md)**: How to define models for analysis
- **[CLI Usage](cli-usage.md)**: Command-line interface and options
- **[Results Interpretation](../results/interpretation.md)**: Understanding analysis output
- **[Advanced Features](../advanced/performance.md)**: Performance optimization

!!! tip "Configuration Strategy"
    
    Start with minimal configurations and gradually add complexity. Use the `--parameters-only` flag to test configurations quickly before generating full visualizations.