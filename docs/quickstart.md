# Quick Start Guide

This guide provides immediate examples to help you understand structural identifiability analysis using the NullStrike package. 

## 30-Second Test

After [installation](installation.md), verify everything works:

```bash
# Run analysis on a simple two-compartment model
nullstrike C2M
```

NullStrike will:

1. Analyze the C2M pharmacokinetic model
2. Generate visualizations in `results/C2M/`
3. Show you which parameter combinations are identifiable

## 2-Minute Analysis

Let's examine what just happened and run a few more examples:

### Example 1: Two-Compartment Model (C2M)

```bash
nullstrike C2M
```

**What this analyzes**: A pharmacokinetic model with drug distribution between central and peripheral compartments.

**Key results**:

- Individual parameters `k12`, `k21`, `V1`, `V2` are unidentifiable

- Parameter combinations like `k12*V1` and `k21*V2` are identifiable

- Results saved in `results/C2M/`

### Example 2: Calibration Model

```bash
nullstrike calibration_single
```

**What this analyzes**: A biochemical reaction model with DNA-enzyme interactions.

**Key results**:

- Models protein production via an enzymatic reaction

- Shows which rate constants can be determined from GFP fluorescence data

- Generates 3D manifold plots of parameter constraints

### Example 3: Bolie Model

```bash
nullstrike Bolie
```

**What this analyzes**: A glucose-insulin interaction model from diabetes research.

**Key results**:

- Multi-parameter identifiability analysis

- Complex parameter interactions in metabolic networks

- Constraint surfaces showing feasible parameter regions

## Understanding the Output

After running any example, check the `results/` directory:

```
results/
└── C2M/                          # Model-specific results
    ├── analysis_report.txt       # Human-readable summary
    ├── nullspace_analysis.txt    # Technical details
    ├── observability_matrix.txt  # Mathematical matrices
    ├── visualizations/           # All plots and graphs
    │   ├── 3d_manifolds/        # 3D parameter surfaces
    │   ├── 2d_projections/      # 2D parameter relationships
    │   └── parameter_graph.png   # Network representation
    └── checkpoints/             # Cached computations
```

### Quick Result Interpretation

=== "Identifiable Parameters"

    ```
    IDENTIFIABLE PARAMETERS:
    - k10*V1 (combination of elimination rate and volume)
    ```
    
    These parameter combinations can be uniquely determined from data.

=== "Unidentifiable Parameters"

    ```
    UNIDENTIFIABLE PARAMETERS:
    - k12 (inter-compartment rate constant)
    - V1 (central compartment volume)
    ```
    
    These individual parameters cannot be determined, but their combinations might be.

=== "Parameter Combinations"

    ```
    IDENTIFIABLE COMBINATIONS:
    - k12*V1: links distribution rate to central volume
    - k21*V2: links return rate to peripheral volume
    ```
    
    These products/combinations are mathematically determinable from output data.

## 5-Minute Custom Analysis

Want to analyze your own model? Create a simple example:

### Step 1: Define Your Model

Create `custom_models/my_model.py`:

```python
import sympy as sym

# Define states (x), parameters (p), outputs (h)
x1, x2 = sym.symbols('x1 x2')
p1, p2, p3 = sym.symbols('p1 p2 p3')

# State vector
x = [[x1], [x2]]

# Parameter vector  
p = [[p1], [p2], [p3]]

# Output equations (what you can measure)
h = [x1]  # Only x1 is observable

# System dynamics (differential equations)
f = [[p1*x1 - p2*x1*x2], [p2*x1*x2 - p3*x2]]

# Required for NullStrike
u = []  # No inputs
w = []  # No unknown inputs
variables_locales = locals().copy()
```

### Step 2: Create Options File

Create `custom_options/options_my_model.py`:

```python
modelname = 'my_model'
checkObser = 1
from math import inf
maxLietime = inf
nnzDerU = [0] 
nnzDerW = [inf]
prev_ident_pars = []

# Manifold plotting configuration
MANIFOLD_PLOTTING = {
    "var_ranges": {},
    "positive_symbols": ["p1", "p2", "p3"],
    "default_positive_range": (1e-3, 10.0, 100),
    "log_for_positive": True,
}
```

### Step 3: Run Analysis

```bash
nullstrike my_model
```

## Command Line Options

NullStrike provides several analysis modes:

```bash
# Full analysis (default)
nullstrike C2M

# Use specific options file
nullstrike C2M options_C2M  

# Parameters-only analysis (faster, no visualizations)
nullstrike C2M --parameters-only

# Get help
nullstrike --help
```

## Python API Quick Start

Prefer programming? Use the Python interface:

```python
from nullstrike.cli.complete_analysis import main

# Run analysis programmatically
result = main('C2M', 'options_C2M')

# Access results
print("Analysis completed!")
print("Check results/ directory for output files")
```

## What's Different About NullStrike?

Traditional identifiability tools tell you **which parameters are unidentifiable**. NullStrike goes further by finding **which parameter combinations are identifiable** even when individual parameters aren't.

### Traditional Analysis
```
Parameters k12, k21, V1, V2 are unidentifiable.
Analysis complete.
```

### NullStrike Analysis
```
Individual parameters k12, k21, V1, V2 are unidentifiable.

BUT these combinations ARE identifiable:
- k12*V1 = 0.45 ± 0.02
- k21*V2 = 1.23 ± 0.05  
- (k12+k21+k10)*V1 = 2.1 ± 0.1

Visualizations show constraint manifolds in parameter space.
```

This information is crucial for:

- **Model calibration**: Focus on identifiable combinations
- **Experiment design**: Understand what data can/cannot determine
- **Parameter estimation**: Use proper constraints in fitting
- **Model reduction**: Eliminate unidentifiable directions

## Next Steps

Now that you've run your first analyses:

1. **[Understand the theory](theory.md)**: Learn the mathematical foundations
2. **[Deep dive tutorial](quickstart.md)**: Step-by-step walkthrough
3. **[Explore examples](examples.md)**: See more complex models
4. **[Model definition guide](reference.md)**: Learn to define your own models
5. **[Results interpretation](examples.md)**: Understand the output files

## Common Use Cases

=== "Pharmacokinetics"
    
    **Models**: Drug absorption, distribution, metabolism
    
    ```bash
    nullstrike C2M          # Two-compartment model
    nullstrike C2M_2outputs # Multiple measurements
    ```

=== "Systems Biology"
    
    **Models**: Gene expression, enzyme kinetics, signaling
    
    ```bash
    nullstrike calibration_single  # Enzyme kinetics
    nullstrike Bolie              # Glucose-insulin
    ```

=== "Engineering Systems"
    
    **Models**: Control systems, chemical reactors
    
    ```bash
    nullstrike SimpleExample  # Basic linear system
    ```

=== "Custom Applications"
    
    **Models**: Your domain-specific models
    
    ```bash
    # Define in custom_models/ and run
    nullstrike your_model
    ```

---

!!! tip "Pro Tips"
    
    - Start with built-in examples to understand the workflow
    - Use `--parameters-only` for quick testing during model development  
    - Check `results/*/analysis_report.txt` for human-readable summaries
    - Visualizations in `results/*/visualizations/` show parameter relationships
    - Each model run creates checkpoints for efficient re-analysis

!!! question "Need Help?"
    
    - **Stuck?** Check the [troubleshooting guide](installation.md#troubleshooting)
    - **Want more examples?** Browse the [examples section](examples.md)
    - **Questions?** Open an issue on [GitHub](https://github.com/vipulsinghal02/NullStrike/issues)