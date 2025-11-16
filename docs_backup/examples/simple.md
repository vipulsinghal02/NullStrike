# Simple Example: Linear Two-State System

This example walks through a complete NullStrike analysis using the simplest possible model - a linear two-state system. This tutorial is perfect for understanding the core concepts without complex mathematics.

## The Model

We'll analyze a simple linear system representing, for example, drug distribution between two compartments:

$$\begin{align}
\frac{dx_1}{dt} &= -k_1 x_1 + k_2 x_2 \\
\frac{dx_2}{dt} &= k_1 x_1 - k_2 x_2 \\
y &= x_1
\end{align}$$

Where:
- $x_1$: Amount in compartment 1
- $x_2$: Amount in compartment 2  
- $k_1$: Transfer rate from compartment 1 to 2
- $k_2$: Transfer rate from compartment 2 to 1
- $y$: Observable output (only compartment 1 is measured)

## Model Definition

Create the file `custom_models/simple_linear.py`:

```python
import sympy as sym

# ============================================================================
# MODEL: Simple Linear Two-Compartment System
# Description: Basic linear system for demonstrating NullStrike analysis
# ============================================================================

# State variables
x1 = sym.Symbol('x1')  # Amount in compartment 1
x2 = sym.Symbol('x2')  # Amount in compartment 2

# Parameter variables
k1 = sym.Symbol('k1')  # Transfer rate 1->2
k2 = sym.Symbol('k2')  # Transfer rate 2->1

# ============================================================================
# SYSTEM DEFINITION
# ============================================================================

# State vector
x = [[x1], [x2]]

# Parameter vector
p = [[k1], [k2]]

# Output vector (only x1 is observable)
h = [x1]

# No inputs
u = []

# No unknown inputs
w = []

# System dynamics
f = [
    [-k1*x1 + k2*x2],  # dx1/dt
    [k1*x1 - k2*x2]    # dx2/dt
]

# Required boilerplate
variables_locales = locals().copy()
```

## Configuration File

Create `custom_options/options_simple_linear.py`:

```python
# ============================================================================
# CONFIGURATION: Simple Linear System
# ============================================================================

import sympy as sym
from math import inf

# Model identification
modelname = 'simple_linear'

# Analysis options
checkObser = 1          # Check state observability
maxLietime = inf        # No time limit (simple model)
nnzDerU = [0]          # No input derivatives
nnzDerW = [inf]        # Unknown input derivatives
prev_ident_pars = []   # No pre-identified parameters

# Visualization configuration
MANIFOLD_PLOTTING = {
    "var_ranges": {
        "k1": (0.1, 2.0, 50),    # Transfer rate 1->2
        "k2": (0.1, 2.0, 50),    # Transfer rate 2->1
    },
    "positive_symbols": ["k1", "k2"],
    "default_positive_range": (0.1, 5.0, 100),
    "log_for_positive": False,  # Linear scale for simplicity
    "default_var_range": (-2.0, 2.0, 100),
    
    # Plot specific parameter combinations
    "extra_pairs_2d": [
        ("k1", "k2"),
    ],
    
    # Limit plots for this simple example
    "max_triplets_3d": 0,  # No 3D plots
    "max_pairs_2d": 5,     # Few 2D plots
}
```

## Running the Analysis

### Step 1: Quick Test

First, test that everything is set up correctly:

```bash
nullstrike simple_linear --parameters-only
```

Expected output:
```
=== NULLSTRIKE ANALYSIS: simple_linear ===
Loading model: simple_linear
Loading options: options_simple_linear

=== STRIKE-GOLDD Analysis ===
Computing Lie derivatives... Done
Building observability matrix... Done
Rank analysis... Done
Found 2 identifiable parameters out of 2

=== Nullspace Analysis ===
Computing nullspace basis... Done
Nullspace dimension: 0
All parameters are identifiable!

Analysis complete!
```

### Step 2: Full Analysis

Now run the complete analysis with visualizations:

```bash
nullstrike simple_linear
```

This generates all plots and detailed reports.

## Understanding the Results

### Mathematical Analysis

For this linear system, the observability matrix is:

$$\mathcal{O} = \begin{bmatrix}
x_1 \\
-k_1 x_1 + k_2 x_2
\end{bmatrix}$$

The Jacobian with respect to parameters is:

$$\mathcal{J} = \frac{\partial \mathcal{O}}{\partial [k_1, k_2]} = \begin{bmatrix}
0 & 0 \\
-x_1 & x_2
\end{bmatrix}$$

**Key insight**: The rank of $\mathcal{J}$ depends on the state values $x_1$ and $x_2$.

### Identifiability Results

The analysis reveals:

1. **Both parameters are identifiable** when $x_1 \neq 0$ and $x_2 \neq 0$
2. **The nullspace is empty** (dimension 0)
3. **No parameter combinations are needed** - individuals are identifiable

### Physical Interpretation

This result makes intuitive sense:

- **$k_1$ identifiable**: The rate $x_1$ decreases tells us about $k_1$
- **$k_2$ identifiable**: The contribution to $\frac{dx_1}{dt}$ from $x_2$ tells us about $k_2$
- **Both needed**: We need non-zero amounts in both compartments to see both rates

## Examining the Output Files

### Analysis Report

Check `results/simple_linear/analysis_report.txt`:

```
STRUCTURAL IDENTIFIABILITY ANALYSIS: simple_linear
================================================

MODEL SUMMARY:
- States: 2 (x1, x2)
- Parameters: 2 (k1, k2)
- Outputs: 1 (x1)

IDENTIFIABILITY RESULTS:

IDENTIFIABLE PARAMETERS:
- k1: Transfer rate from compartment 1 to 2
- k2: Transfer rate from compartment 2 to 1

UNIDENTIFIABLE PARAMETERS:
(None)

IDENTIFIABLE COMBINATIONS:
(All individual parameters are identifiable)

OBSERVABILITY ANALYSIS:
- Initial condition x1(0): Identifiable
- Initial condition x2(0): Not directly observable

SUMMARY:
This simple linear system has excellent identifiability properties.
Both rate parameters can be uniquely determined from measurements
of compartment 1, provided both compartments contain non-zero amounts.
```

### Technical Details

Check `results/simple_linear/nullspace_analysis.txt`:

```
NULLSPACE ANALYSIS RESULTS
========================

Observability Matrix:
[x1]
[-k1*x1 + k2*x2]

Identifiability Matrix (Jacobian):
[0, 0]
[-x1, x2]

Matrix Rank: 2 (when x1 ≠ 0 and x2 ≠ 0)
Nullspace Dimension: 0

INTERPRETATION:
- All parameters are structurally identifiable
- No parameter combinations are needed
- Identifiability depends on having non-zero states
```

### Visualizations

The visualization folder contains:

- `results/simple_linear/visualizations/2d_projections/k1_vs_k2.png`
- `results/simple_linear/visualizations/parameter_graph.png`

The parameter graph shows two isolated nodes (k1 and k2) since they're independently identifiable.

## Experimental Design Insights

### What This Analysis Tells Us

1. **Measurement strategy**: Measuring only $x_1$ is sufficient for parameter identification
2. **Initial conditions**: Need non-zero amounts in both compartments initially
3. **Experiment duration**: Need to observe the system long enough to see transfer dynamics
4. **Data quality**: Both parameters are identifiable, so focus on measurement precision

### Practical Recommendations

Based on the identifiability analysis:

1. **Design experiments** with appropriate initial conditions: $x_1(0) > 0$, $x_2(0) > 0$
2. **Collect time-series data** that captures the dynamic transfer between compartments
3. **Focus on measurement accuracy** of compartment 1 since that's what determines identifiability
4. **Consider adding measurements** of compartment 2 for improved parameter precision (though not necessary for identifiability)

## Modifying the Example

### Adding Complexity

Let's explore how the analysis changes with modifications:

#### Variant 1: Unknown Initial Condition

Modify the model to treat initial conditions as unknown parameters:

```python
# Add initial condition parameters
x1_0 = sym.Symbol('x1_0')  # Initial amount in compartment 1
x2_0 = sym.Symbol('x2_0')  # Initial amount in compartment 2

# Expanded parameter vector
p = [[k1], [k2], [x1_0], [x2_0]]
```

Re-run the analysis to see how this affects identifiability.

#### Variant 2: Two Outputs

Modify to observe both compartments:

```python
# Both compartments observable
h = [x1, x2]
```

This should improve the identifiability properties.

#### Variant 3: Input Addition

Add an external input to compartment 1:

```python
# Add input
u1 = sym.Symbol('u1')
u = [u1]

# Modify dynamics
f = [
    [-k1*x1 + k2*x2 + u1],  # Input to compartment 1
    [k1*x1 - k2*x2]
]
```

## Comparison with Complex Models

### Why Start Simple?

This simple example demonstrates that:

1. **Linear systems** often have good identifiability properties
2. **Clear mathematical structure** makes analysis interpretation straightforward  
3. **Physical intuition** aligns with mathematical results
4. **Baseline understanding** prepares you for complex nonlinear cases

### Next Steps

After mastering this simple example:

1. **Try the two-compartment model**: `nullstrike C2M` - similar structure but with elimination
2. **Explore nonlinear systems**: `nullstrike calibration_single` - see how nonlinearity affects identifiability
3. **Work with your own models**: Apply the principles to your research domain

## Exercise: Parameter Sensitivity

### Exploration Tasks

1. **Modify parameter ranges** in the configuration file and observe how it affects visualizations
2. **Change the observable output** to $h = [x_2]$ and see how identifiability changes
3. **Add an elimination term** like $-k_0 x_1$ to the first equation and analyze the new system
4. **Compare with analytical solutions** - solve the linear system analytically and verify that the identified parameters make sense

### Expected Learning

- Understanding how model structure affects identifiability
- Appreciation for the relationship between observability and identifiability
- Intuition for experimental design based on mathematical analysis
- Preparation for analyzing more complex systems

## Summary

This simple linear example demonstrates:

- **Basic NullStrike workflow**: Model definition → Configuration → Analysis → Results
- **Identifiability concepts**: What it means for parameters to be identifiable
- **Result interpretation**: Understanding analysis output and physical meaning
- **Experimental design**: How identifiability analysis guides experiments
- **Mathematical foundations**: Observability matrices, Jacobians, and rank analysis

The key insight is that **even simple systems provide valuable insights** when analyzed systematically. This foundation prepares you to tackle more complex nonlinear systems where the relationship between model structure and identifiability becomes much more subtle.

---

## Further Reading

- **Two-Compartment Model**: Similar structure with added complexity (see custom_models/C2M.py)
- **Calibration Model**: Nonlinear system with enzyme kinetics (see custom_models/calibration_single.py)  
- **[Theory Overview](../theory/overview.md)**: Mathematical foundations
- **[Results Interpretation](../results/interpretation.md)**: How to analyze output files

!!! tip "Building Intuition"
    
    Simple examples like this are invaluable for building intuition about identifiability analysis. Master the concepts here before moving to complex nonlinear systems.