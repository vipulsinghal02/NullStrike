# Reference Documentation

Complete reference for NullStrike CLI commands, Python API, configuration options, and troubleshooting.

## CLI Reference

### Main Command

```bash
nullstrike [model_name] [options_file] [flags]
```

**Arguments:**
- `model_name` (optional): Name of model in `custom_models/`. Default: `calibration_single`
- `options_file` (optional): Name of options file in `custom_options/`. Default: `options_{model_name}` or `options_default`

**Flags:**
- `--parameters-only` or `-p`: Analyze only parameters (skip states/inputs), faster
- `--help` or `-h`: Show help message
- `--check`: Validate environment setup
- `--example`: Create example model and options files
- `--demo`: Run quick demo with C2M model

### Examples

```bash
# Use defaults
nullstrike

# Specific model with auto-detected options
nullstrike C2M

# Specific model and options
nullstrike C2M options_C2M

# Fast parameters-only analysis
nullstrike Bolie --parameters-only
nullstrike Bolie -p

# Utility commands
nullstrike --check
nullstrike --example
nullstrike --demo
nullstrike --help
```

---

## Python API Reference

### High-Level API

#### `main()`

Run complete analysis from CLI.

```python
from nullstrike.cli.complete_analysis import main

results = main('C2M', 'options_C2M')
```

**Returns:**
```python
{
    'model_name': str,
    'nullspace_dimension': int,
    'fully_identifiable': bool,
    'matrix_rank': int,
    'matrix_shape': tuple,
    'unidentifiable_patterns': list,
    'identifiable_info': dict,
    'model_info': dict
}
```

### Core Analysis Functions

#### `strike_goldd()`

Run STRIKE-GOLDD algorithm only (no nullspace analysis).

```python
from nullstrike.core import strike_goldd

strike_goldd('options_C2M')
```

**Parameters:**
- `options_file` (str, optional): Options module name

**Effects:**
- Computes observability-identifiability matrix
- Saves results to `results/obs_ident_matrix_*.txt`
- Prints identifiability results to console

#### `run_integrated_analysis()`

Run complete integrated analysis with nullspace.

```python
from nullstrike.analysis import run_integrated_analysis

results = run_integrated_analysis(
    model_name='C2M',
    options_file='options_C2M',
    analysis_scope='full'  # or 'parameters'
)
```

**Parameters:**
- `model_name` (str): Model to analyze
- `options_file` (str): Options configuration
- `analysis_scope` (str): `'full'` or `'parameters'`

**Returns:** Same as `main()`

#### `analyze_identifiable_combinations()`

Perform nullspace analysis on existing OIC matrix.

```python
from nullstrike.analysis.enhanced_subspace import analyze_identifiable_combinations

results = analyze_identifiable_combinations(
    O_matrix,           # sympy.Matrix
    param_symbols,      # list of sympy.Symbol
    state_symbols,      # list of sympy.Symbol
    input_symbols=[],   # list of sympy.Symbol (optional)
    analysis_scope='full'
)
```

**Returns:**
```python
{
    'nullspace_dimension': int,
    'fully_identifiable': bool,
    'nullspace_vectors': list,
    'unidentifiable_patterns': list,
    'identifiable_info': dict
}
```

### Checkpointing Functions

#### `save_checkpoint()`

Save analysis results to cache.

```python
from nullstrike.analysis.checkpointing import save_checkpoint

save_checkpoint(
    model_name='C2M',
    options_file='options_C2M',
    model_hash='abc123',
    oic_matrix=O,
    nullspace_results=results
)
```

#### `load_checkpoint()`

Load cached analysis results.

```python
from nullstrike.analysis.checkpointing import load_checkpoint

checkpoint = load_checkpoint(
    model_name='C2M',
    options_file='options_C2M',
    model_hash='abc123'
)

if checkpoint:
    O = checkpoint['oic_matrix']
    results = checkpoint['nullspace_results']
```

#### `compute_model_hash()`

Generate hash for model+options for cache invalidation.

```python
from nullstrike.analysis.checkpointing import compute_model_hash

model_hash = compute_model_hash(model_module, options_module)
```

### Visualization Functions

#### `build_identifiability_graph()`

Create NetworkX graph from nullspace analysis.

```python
from nullstrike.visualization.graphs import build_identifiability_graph

G = build_identifiability_graph(
    nullspace_results,
    param_symbols,
    state_symbols=[],
    input_symbols=[]
)
```

#### `visualize_identifiability_graph()`

Plot identifiability graph.

```python
from nullstrike.visualization.graphs import visualize_identifiability_graph

visualize_identifiability_graph(
    G,
    model_name='C2M',
    output_dir='results/C2M/graphs',
    show_all_vars=True
)
```

#### `visualize_nullspace_manifolds()`

Create 2D/3D manifold plots.

```python
from nullstrike.visualization.manifolds import visualize_nullspace_manifolds

visualize_nullspace_manifolds(
    nullspace_results,
    param_symbols,
    model_name='C2M',
    output_dir='results/C2M',
    config={'max_2d_plots': 10, 'max_3d_plots': 5}
)
```

---

## Configuration Reference

### Options File Format

**File location:** `custom_options/options_{model}.py`

```python
from math import inf

# === REQUIRED ===
modelname = 'model_name'  # Must match custom_models/ filename

# === ANALYSIS SETTINGS ===
checkObser = 1            # Check observability: 1=yes, 0=no
maxLietime = inf          # Max seconds per Lie derivative
nnzDerU = [inf]          # Known input derivatives (per input)
nnzDerW = [inf]          # Unknown input derivatives (per input)
prev_ident_pars = []     # List of already-identified parameter names

# === VISUALIZATION (OPTIONAL) ===
MANIFOLD_PLOTTING = {
    'enabled': True,
    'max_2d_plots': 10,
    'max_3d_plots': 5,
    'resolution': 50,
    'figsize': (10, 8)
}
```

### Configuration Options Explained

#### `modelname`
**Type:** `str`
**Required:** Yes
**Description:** Name of model file (without `.py`)
**Example:** `'C2M'` → looks for `custom_models/C2M.py`

#### `checkObser`
**Type:** `int` (0 or 1)
**Default:** `1`
**Description:** Whether to check state observability
**Values:**
- `1`: Check observability (recommended)
- `0`: Skip observability check (faster)

#### `maxLietime`
**Type:** `float` or `inf`
**Default:** `inf`
**Description:** Maximum seconds to compute each Lie derivative
**Recommendations:**
- `inf`: No limit (may take long for complex models)
- `100-300`: Reasonable limit for most models
- `60`: Quick analysis

#### `nnzDerU`
**Type:** `list[float]`
**Default:** `[inf]`
**Description:** Non-zero derivatives for each known input
**Values:**
- `[inf]`: Input can vary arbitrarily
- `[0]`: Input is constant
- `[1]`: Input + first derivative can vary
- Multiple values: One per input in order

#### `nnzDerW`
**Type:** `list[float]`
**Default:** `[inf]`
**Description:** Non-zero derivatives for unknown inputs
**Same as `nnzDerU` but for unknown inputs `w`

#### `prev_ident_pars`
**Type:** `list[str]`
**Default:** `[]`
**Description:** Parameter names already identified (excluded from analysis)
**Example:** `['V1', 'k10']` → These are known, analyze others

#### `MANIFOLD_PLOTTING`
**Type:** `dict`
**Default:** `{'enabled': True}`
**Description:** Visualization configuration

**Sub-options:**
- `enabled` (bool): Generate plots or not
- `max_2d_plots` (int): Maximum 2D pairwise plots
- `max_3d_plots` (int): Maximum 3D triple plots
- `resolution` (int): Grid resolution for surfaces
- `figsize` (tuple): Figure size in inches

---

## Model Definition Reference

### Model File Format

**File location:** `custom_models/{model_name}.py`

```python
import sympy as sym

# === STATES ===
x1, x2 = sym.symbols('x1 x2')
x = [[x1], [x2]]  # List of lists format

# === PARAMETERS ===
p1, p2, p3 = sym.symbols('p1 p2 p3')
p = [[p1], [p2], [p3]]  # List of lists format

# === OUTPUTS (measurements) ===
h = [x1, x2]  # List format (can measure multiple)

# === KNOWN INPUTS (optional) ===
u1 = sym.Symbol('u1')
u = [u1]  # List format

# === UNKNOWN INPUTS (optional) ===
w1 = sym.Symbol('w1')
w = [w1]  # List format, or [] if none

# === DYNAMICS (dx/dt = f) ===
f = [
    [p1*x1 + p2*x2 + u1],  # dx1/dt
    [-p3*x2]                # dx2/dt
]  # List of lists, must match x dimensions

# === REQUIRED FOR STRIKEPY ===
variables_locales = locals().copy()
```

### Model Components Explained

#### States (`x`)
**Format:** List of lists `[[x1], [x2], ...]`
**Description:** State variables of the dynamical system
**Example:**
```python
C1, C2 = sym.symbols('C1 C2')
x = [[C1], [C2]]
```

#### Parameters (`p`)
**Format:** List of lists `[[p1], [p2], ...]`
**Description:** Unknown parameters to analyze
**Example:**
```python
k1, k2, V = sym.symbols('k1 k2 V')
p = [[k1], [k2], [V]]
```

#### Outputs (`h`)
**Format:** List `[h1, h2, ...]`
**Description:** Measured outputs (functions of states/params)
**Example:**
```python
h = [x1]              # Measure first state only
h = [x1, x2]          # Measure both states
h = [x1 + x2]         # Measure sum
h = [x1, x2, x1*p1]   # Multiple measurements
```

#### Known Inputs (`u`)
**Format:** List `[u1, u2, ...]`
**Description:** Known external inputs (forcing functions)
**Example:**
```python
u1 = sym.Symbol('u1')
u = [u1]
# or
u = []  # No known inputs
```

#### Unknown Inputs (`w`)
**Format:** List `[w1, w2, ...]`
**Description:** Unknown disturbances or unmodeled inputs
**Example:**
```python
w = []  # Most common - no unknown inputs
# or
w1 = sym.Symbol('w1')
w = [w1]
```

#### Dynamics (`f`)
**Format:** List of lists `[[f1], [f2], ...]`
**Description:** Right-hand side of dx/dt = f
**Must:** Match dimensions of `x`
**Example:**
```python
f = [
    [p1*x1 - p2*x1*x2],  # dx1/dt
    [-p3*x2 + p2*x1*x2]  # dx2/dt
]
```

---

## Troubleshooting Guide

### Common Errors

#### "No module named 'custom_models'"

**Cause:** Python can't find model directory

**Solutions:**
```bash
# 1. Check you're in project root
pwd  # Should be .../NullStrike
cd /path/to/NullStrike

# 2. Check model exists
ls custom_models/my_model.py

# 3. Check for __init__.py
ls custom_models/__init__.py  # Should exist
```

#### "No OIC matrix found"

**Cause:** StrikePy hasn't run or results not saved

**Solutions:**
```bash
# Run full analysis (not just load results)
nullstrike my_model

# Check results directory
ls results/
```

#### "Model hash mismatch" or checkpoint invalidated

**Cause:** Model or options file changed

**This is normal:** Checkpoint invalidates automatically when you modify model/options

**To force recomputation:**
```bash
rm -rf checkpoints/my_model/
nullstrike my_model
```

#### Analysis stuck/taking too long

**Causes:** Complex symbolic computation

**Solutions:**
```python
# 1. Limit time in options
maxLietime = 300  # 5 minutes

# 2. Use parameters-only
```
```bash
nullstrike my_model -p
```

```python
# 3. Simplify model
# - Reduce number of parameters
# - Use simpler dynamics
```

#### Import errors in model file

**Cause:** Syntax error in model definition

**Check:**
```python
# Must import sympy
import sympy as sym

# Must end with this line
variables_locales = locals().copy()

# Check symbols defined
print(x, p, h, f)
```

### Performance Issues

#### Slow first run

**Normal:** Symbolic computation is inherently slow

**Expected times:**
- Simple model (2-3 params): 10-30 seconds
- Medium model (4-6 params): 30-120 seconds
- Complex model (8+ params): 2-10 minutes

**Solutions:**
- Use checkpointing (automatic)
- Limit `maxLietime`
- Simplify model if possible

#### Slow visualization

**Cause:** Many parameters → many plots

**Solutions:**
```python
MANIFOLD_PLOTTING = {
    'enabled': False  # Disable entirely
}
# or
MANIFOLD_PLOTTING = {
    'max_2d_plots': 3,  # Reduce number
    'max_3d_plots': 0   # Skip 3D
}
```

### Result Interpretation Issues

#### Unexpected nullspace dimension

**Check:**
1. Model definition is correct
2. Sufficient measurements in `h`
3. Review `results/detailed_analysis.txt`
4. Compare with theoretical expectations

#### Nullspace dimension = 0 but parameters seem unidentifiable

**Possible causes:**
- Global vs local identifiability
- Practical vs structural identifiability
- Initial conditions matter

**Action:** Review theory and model assumptions

#### Visualizations look wrong

**Check:**
1. Parameter ranges in manifold plots
2. Graph connectivity makes sense
3. Compare with text report

---

## File Organization

### Input Files
```
custom_models/
├── {model_name}.py          # Your model definition
└── __init__.py              # Required for Python imports

custom_options/
├── options_{model_name}.py  # Model-specific options
├── options_default.py       # Default options
└── __init__.py
```

### Output Files
```
results/
└── {model_name}/
    └── {timestamp}/
        ├── detailed_analysis.txt
        ├── graphs/
        │   ├── identifiability_graph_full.png
        │   └── identifiability_graph_params.png
        ├── manifolds_2d/
        │   └── {param1}_vs_{param2}.png
        └── manifolds_3d/
            └── {param1}_{param2}_{param3}.png

checkpoints/
└── {model_name}/
    └── checkpoint_{hash}.pkl
```

---

## Environment Setup

### Required Packages

```bash
pip install sympy numpy matplotlib networkx symbtools
```

### Check Installation

```bash
nullstrike --check
```

**Output:**
```
Checking environment...
✓ sympy
✓ numpy
✓ matplotlib
✓ networkx
✓ symbtools
✓ pathlib
✓ models/ directory
✓ results/ directory
✓ custom_options/ directory
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Basic analysis | `nullstrike model_name` |
| With options | `nullstrike model options_file` |
| Fast mode | `nullstrike model -p` |
| Validate setup | `nullstrike --check` |
| Create example | `nullstrike --example` |
| Get help | `nullstrike --help` |
| Demo | `nullstrike --demo` |

## Python Import Reference

```python
# High-level
from nullstrike.cli.complete_analysis import main

# Core analysis
from nullstrike.core import strike_goldd
from nullstrike.analysis import run_integrated_analysis
from nullstrike.analysis.enhanced_subspace import analyze_identifiable_combinations

# Checkpointing
from nullstrike.analysis.checkpointing import save_checkpoint, load_checkpoint, compute_model_hash

# Visualization
from nullstrike.visualization.graphs import build_identifiability_graph, visualize_identifiability_graph
from nullstrike.visualization.manifolds import visualize_nullspace_manifolds
```

---

**For more details, see:**
- [Theory](theory.md) - Mathematical foundations
- [Examples](examples.md) - Step-by-step tutorials
- [Quickstart](quickstart.md) - Get started quickly
- [QUICK_REFERENCE.md](https://github.com/vipulsinghal02/NullStrike/blob/main/QUICK_REFERENCE.md) - One-page cheat sheet
