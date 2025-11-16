# Your First Analysis: Step-by-Step

This guide walks you through performing a complete structural identifiability analysis using NullStrike, from model setup to results interpretation. We'll use the calibration model as our example.

## Overview

In this tutorial, you'll learn the key concepts of:

- **Structural identifiability**: Which parameters can be uniquely determined from data
- **Nullspace analysis**: Finding parameter combinations when individual parameters aren't identifiable  
- **Visualization**: Understanding results through graphs and manifolds
- **Results interpretation**: What the output files tell you about your model

## The Calibration Model

We'll analyze the `calibration_single` model, which represents a biochemical reaction system with DNA template and GFP production:

$$\begin{align}
\frac{d[\text{DNA}]}{dt} &= -k_f(E_{tot} - 10 + [\text{DNA}])[\text{DNA}] + (k_r + k_{cat})(10 - [\text{DNA}]) \\
\frac{d[\text{GFP}]}{dt} &= k_{cat}(10 - [\text{DNA}]) \\
y &= [\text{GFP}]
\end{align}$$

### Biological Context

This model represents an enzymatic reaction where:

1. **DNA template** binds to **enzyme** to form a complex
2. The complex can either **dissociate** (rate $k_r$) or proceed to **product formation** (rate $k_{cat}$)  
3. **GFP** is produced as the observable output
4. Total enzyme concentration is $E_{tot}$, with 10 units initially bound

### Model Components

- **States**: 
  - $[\text{DNA}]$: Free DNA template concentration
  - $[\text{GFP}]$: GFP product concentration
- **Parameters**: 
  - $E_{tot}$: Total enzyme concentration
  - $k_f$: Forward binding rate constant
  - $k_r$: Reverse dissociation rate constant  
  - $k_{cat}$: Catalytic rate constant
- **Output**: $y = [\text{GFP}]$ (observable GFP fluorescence)
- **Input**: None (autonomous system)

!!! note "Why This Model?"
    This model demonstrates **practical identifiability challenges** where multiple parameters affect the same observable output. Understanding which parameter combinations are identifiable is crucial for experimental design and model calibration.

## Step 1: Examine the Model Definition

The model is defined in `custom_models/calibration_single.py`:

```python
import sympy as sym

# State variables
DNA = sym.Symbol('DNA')  # Free DNA concentration
GFP = sym.Symbol('GFP')  # GFP concentration

# Parameters
kf = sym.Symbol('kf')    # Forward binding rate
kr = sym.Symbol('kr')    # Reverse dissociation rate  
kcat = sym.Symbol('kcat') # Catalytic rate
Etot = sym.Symbol('Etot') # Total enzyme concentration

# State vector [DNA, GFP]
x = [[DNA], [GFP]]

# Output vector (what we can measure)
h = [GFP]  # Only GFP is observable

# No external inputs or unknown inputs
u = []
w = []

# Parameter vector
p = [[Etot], [kf], [kr], [kcat]]

# System dynamics (the differential equations)
f = [
    [-kf*(Etot-10+DNA)*DNA + (kr+kcat)*(10-DNA)],  # d[DNA]/dt
    [kcat*(10-DNA)]                                  # d[GFP]/dt
]

# Required for NullStrike
variables_locales = locals().copy()
```

### Understanding the Dynamics

The dynamics encode the biochemical mechanism:

- **DNA binding**: $-k_f(E_{tot} - 10 + [\text{DNA}])[\text{DNA}]$ 
  - Available enzyme: $E_{tot} - 10 + [\text{DNA}]$ (since $10 - [\text{DNA}]$ is bound)
  - Binding rate proportional to free enzyme and free DNA

- **Complex dissociation**: $(k_r + k_{cat})(10 - [\text{DNA}])$
  - Bound complex: $10 - [\text{DNA}]$
  - Can dissociate ($k_r$) or proceed to product ($k_{cat}$)

- **Product formation**: $k_{cat}(10 - [\text{DNA}])$
  - Only the catalytic pathway produces GFP

## Step 2: Examine the Configuration

The analysis options are defined in `custom_options/options_calibration_single.py`:

```python
# Model identification
modelname = 'calibration_single'

# Analysis options
checkObser = 1           # Check observability
maxLietime = inf        # No time limit for Lie derivatives
nnzDerU = [0]          # No input derivatives
nnzDerW = [inf]        # Unknown input derivatives
prev_ident_pars = []   # No pre-identified parameters

# Manifold plotting configuration
MANIFOLD_PLOTTING = {
    "var_ranges": {},
    "positive_symbols": [],      # All parameters are positive
    "default_positive_range": (1e-3, 10.0, 120),
    "log_for_positive": True,
    "default_var_range": (-5.0, 5.0, 100),
    # ... (additional plotting options)
}
```

Key configuration points:

- **`checkObser = 1`**: Analyze state observability in addition to parameter identifiability
- **`maxLietime = inf`**: No time limit (use with caution for complex models)
- **`prev_ident_pars = []`**: No parameters assumed known a priori
- **`MANIFOLD_PLOTTING`**: Controls visualization parameters and ranges

## Step 3: Run the Analysis

Execute the analysis from the command line:

```bash
nullstrike calibration_single
```

What happens during execution:

1. **Model Loading**: NullStrike imports the model definition
2. **STRIKE-GOLDD Analysis**: Computes Lie derivatives and observability matrix
3. **Nullspace Computation**: Finds unidentifiable parameter directions
4. **Identifiable Combinations**: Determines which parameter combinations are identifiable
5. **Visualization Generation**: Creates 3D manifolds and 2D projections
6. **Report Generation**: Produces human-readable summaries

### Expected Output

During execution, you'll see progress information:

```
=== STRIKE-GOLDD Analysis ===
Loading model: calibration_single
Computing Lie derivatives...
Building observability matrix...
Rank analysis complete.

=== Nullspace Analysis ===
Computing nullspace...
Found 2 unidentifiable directions
Computing identifiable combinations...
Found 2 identifiable parameter combinations

=== Visualization ===
Generating 3D manifolds...
Creating 2D projections...
Building parameter dependency graph...

Analysis complete! Results saved to: results/calibration_single/
```

## Step 4: Examine the Results

After completion, check the `results/calibration_single/` directory:

```
results/calibration_single/
├── analysis_report.txt              # Human-readable summary
├── nullspace_analysis.txt           # Technical mathematical details
├── observability_matrix.txt         # Computed matrices
├── visualizations/
│   ├── 3d_manifolds/
│   │   ├── Etot_kf_kr_manifold.png
│   │   ├── Etot_kf_kcat_manifold.png
│   │   └── ...
│   ├── 2d_projections/
│   │   ├── Etot_vs_kf.png
│   │   ├── kr_vs_kcat.png
│   │   └── ...
│   └── parameter_graph.png
└── checkpoints/                     # Cached computations
    ├── observability_matrix.pkl
    └── analysis_state.pkl
```

### Key Result Files

=== "analysis_report.txt"

    **Human-readable summary** of the analysis:
    
    ```
    STRUCTURAL IDENTIFIABILITY ANALYSIS: calibration_single
    =====================================================
    
    MODEL SUMMARY:
    - States: 2 (DNA, GFP)
    - Parameters: 4 (Etot, kf, kr, kcat)  
    - Outputs: 1 (GFP)
    
    IDENTIFIABILITY RESULTS:
    
    UNIDENTIFIABLE PARAMETERS:
    - kf: Forward binding rate
    - kr: Reverse dissociation rate
    
    IDENTIFIABLE PARAMETERS:
    - Etot: Total enzyme concentration
    - kcat: Catalytic rate constant
    
    IDENTIFIABLE COMBINATIONS:
    - kf + kr: Total binding/dissociation rate
    
    PARAMETER CONSTRAINTS:
    The model constrains parameter relationships through the
    nullspace structure. See visualizations for geometric
    interpretation.
    ```

=== "nullspace_analysis.txt"

    **Technical mathematical details**:
    
    ```
    NULLSPACE ANALYSIS RESULTS
    =========================
    
    Observability Matrix Rank: 6
    Nullspace Dimension: 2
    
    Nullspace Basis Vectors:
    v1 = [0, 1, -1, 0]  # kf - kr combination
    v2 = [1, 0, 0, -1]  # Etot - kcat combination
    
    Identifiable Directions:
    - kf + kr (orthogonal to v1)
    - Etot + kcat (orthogonal to v2)
    
    Mathematical Interpretation:
    The model cannot distinguish between kf and kr individually,
    but can identify their sum. Similarly for other combinations.
    ```

=== "Visualizations"

    The `visualizations/` folder contains:
    
    - **3D Manifolds**: Show constraint surfaces in parameter space
    - **2D Projections**: Pairwise parameter relationships
    - **Parameter Graph**: Network showing identifiability dependencies

## Step 5: Interpret the Mathematical Results

### Understanding Identifiability

The analysis reveals the mathematical structure of what's identifiable:

#### Individual Parameters

- **Identifiable**: $E_{tot}$, $k_{cat}$
  - These can be uniquely determined from GFP time-series data
  - The total enzyme affects the reaction capacity
  - The catalytic rate directly controls GFP production rate

- **Unidentifiable**: $k_f$, $k_r$ 
  - These cannot be individually determined
  - The model output depends on their combination, not individual values
  - Multiple $(k_f, k_r)$ pairs produce identical GFP trajectories

#### Parameter Combinations

- **Identifiable combination**: $k_f + k_r$
  - This sum represents the total rate of complex turnover
  - Observable in the equilibrium between free and bound DNA
  - Critical for understanding reaction dynamics

### Biological Interpretation

The identifiability structure reveals biological constraints:

1. **Enzyme saturation effects**: $E_{tot}$ is identifiable because it determines the maximum possible reaction rate

2. **Product formation rate**: $k_{cat}$ is identifiable because it directly controls observable GFP accumulation

3. **Binding equilibrium**: Individual binding constants $k_f$, $k_r$ are unidentifiable because only their balance (sum) affects the observable output

4. **Experimental design implications**: To determine $k_f$ and $k_r$ individually, you'd need additional measurements (e.g., direct observation of free DNA concentration)

## Step 6: Explore the Visualizations

### 3D Manifolds

Open `visualizations/3d_manifolds/Etot_kf_kr_manifold.png`:

This shows the constraint surface in $(E_{tot}, k_f, k_r)$ space where the model produces identical outputs. Points on this surface are indistinguishable from data.

**Interpretation**:
- The surface represents parameter combinations that produce identical GFP time-series
- Perpendicular directions to the surface are identifiable
- Tangent directions are unidentifiable

### 2D Projections  

Examine `visualizations/2d_projections/kf_vs_kr.png`:

This shows the identifiable constraint: $k_f + k_r = \text{constant}$

**Key insights**:
- The diagonal line shows identifiable combinations
- Movement along the line maintains model predictions
- Movement perpendicular to the line changes predictions

### Parameter Dependency Graph

View `visualizations/parameter_graph.png`:

This network representation shows:
- **Nodes**: Parameters
- **Edges**: Identifiability relationships
- **Colors**: Identifiability status
- **Clusters**: Related parameter groups

## Step 7: Practical Applications

### Model Calibration Strategy

Based on the analysis:

1. **Focus on identifiable parameters**: Estimate $E_{tot}$, $k_{cat}$, and $k_f + k_r$

2. **Fix unidentifiable combinations**: Choose reasonable values for $k_f$ and $k_r$ individually, ensuring their sum matches the identified value

3. **Use prior knowledge**: If available, use biological knowledge to constrain $k_f/k_r$ ratios

### Experimental Design Recommendations

To improve identifiability:

1. **Add measurements**: Direct observation of DNA concentration would resolve $k_f$ vs $k_r$

2. **Perturbation experiments**: Vary initial enzyme concentration to better identify $E_{tot}$

3. **Time-course design**: Ensure measurements capture both transient and steady-state behavior

### Model Validation

The analysis helps validate model structure:

1. **Parameter constraints**: Check if estimated values satisfy known biological bounds

2. **Prediction intervals**: Use identifiable combinations for robust predictions

3. **Sensitivity analysis**: Focus on identifiable directions for model sensitivity

## Step 8: Advanced Analysis Options

### Parameters-Only Mode

For faster analysis during model development:

```bash
nullstrike calibration_single --parameters-only
```

This skips visualization generation while providing identifiability results.

### Custom Configuration

Modify `custom_options/options_calibration_single.py` to:

- Add prior knowledge via `prev_ident_pars`
- Adjust visualization ranges in `MANIFOLD_PLOTTING`
- Control Lie derivative computation time with `maxLietime`

### Comparison with Other Models

Run similar analyses on related models:

```bash
nullstrike calibration_double  # Two-extract version
nullstrike C2M                 # Two-compartment pharmacokinetic
nullstrike Bolie               # Glucose-insulin dynamics
```

Compare identifiability patterns across different model structures.

## Troubleshooting Common Issues

### Symbolic Computation Warnings

**Issue**: Long computation times or memory warnings

**Solutions**:
- Use `--parameters-only` for initial testing
- Simplify the model for exploratory analysis
- Increase system memory or use a more powerful machine

### Visualization Errors

**Issue**: Empty or malformed plots

**Solutions**:
- Check parameter ranges in `MANIFOLD_PLOTTING`
- Ensure positive parameters have appropriate bounds
- Verify that constraints are not over-constrained

### Interpretation Difficulties

**Issue**: Complex nullspace structure

**Solutions**:
- Start with simpler models to build intuition
- Focus on the human-readable `analysis_report.txt`
- Consult the mathematical theory in [Theory Overview](../theory/overview.md)

## Next Steps

Now that you've completed your first analysis:

1. **[Theory Deep Dive](../theory/overview.md)**: Understand the mathematical foundations
2. **[Model Definition Guide](../guide/models.md)**: Learn to create your own models
3. **[Advanced Examples](../examples/c2m.md)**: Explore more complex systems
4. **[Results Interpretation](../results/interpretation.md)**: Master the analysis of output files
5. **[Python API](../guide/python-api.md)**: Integrate NullStrike into your workflows

## Summary

In this tutorial, you learned:

- How to run a complete NullStrike analysis
- How to interpret identifiability results  
- How to understand nullspace structure
- How to use visualizations for insight
- How to apply results to experimental design
- How to troubleshoot common issues

The calibration model demonstrates that structural identifiability analysis provides actionable insights for both mathematical understanding and practical experimental design. The combination of individual parameter identifiability, parameter combination analysis, and visualization makes NullStrike a powerful tool for understanding complex dynamical systems.

---

!!! success "Congratulations!"
    
    You've successfully completed your first NullStrike analysis! You now understand the workflow from model definition through results interpretation. This foundation will serve you well as you analyze your own models and explore more advanced features.