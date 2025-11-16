# NullStrike Quick Reference

**One-page guide to using NullStrike for structural identifiability analysis**

---

## Installation

```bash
git clone https://github.com/vipulsinghal02/NullStrike.git
cd NullStrike
pip install -e .
```

---

## CLI Commands

### Basic Usage
```bash
# Analyze built-in models
nullstrike C2M                          # Two-compartment model
nullstrike Bolie                        # Bolie glucose-insulin model
nullstrike calibration_single           # Calibration example (default)

# Custom model with custom options
nullstrike my_model options_my_model

# Parameters-only analysis (skip states/inputs)
nullstrike C2M --parameters-only
nullstrike C2M -p
```

### Utility Commands
```bash
nullstrike --help          # Show help
nullstrike --check         # Validate environment
nullstrike --example       # Create example model/options files
nullstrike --demo          # Quick demo with C2M model
```

---

## Python API

### Quick Analysis
```python
from nullstrike.cli.complete_analysis import main

# Run complete analysis
results = main('C2M', 'options_C2M')
```

### Using StrikePy Directly
```python
from nullstrike.core import strike_goldd

# Just run STRIKE-GOLDD
strike_goldd('options_C2M')
```

### Advanced: Integrated Analysis
```python
from nullstrike.analysis import run_integrated_analysis

# Full control over analysis
results = run_integrated_analysis(
    model_name='C2M',
    options_file='options_C2M',
    analysis_scope='parameters'  # or 'full'
)
```

### Analyzing Existing Results
```python
from nullstrike.analysis.enhanced_subspace import analyze_strikepy_results

# Analyze previously generated StrikePy output
results = analyze_strikepy_results('C2M')
```

---

## Model Definition Template

**File: `custom_models/my_model.py`**

```python
import sympy as sym

# === States ===
x1, x2 = sym.symbols('x1 x2')
x = [[x1], [x2]]  # Must be list of lists

# === Parameters ===
p1, p2, p3 = sym.symbols('p1 p2 p3')
p = [[p1], [p2], [p3]]  # Must be list of lists

# === Outputs (measurements) ===
h = [x1]  # What you can measure

# === Known Inputs ===
u1 = sym.Symbol('u1')
u = [u1]

# === Unknown Inputs (optional) ===
w = []  # Empty if none

# === Dynamics: dx/dt = f ===
f = [
    [-p1*x1 + p2*x2 + u1],  # dx1/dt
    [p3*x1]                  # dx2/dt
]

# === Required for StrikePy ===
variables_locales = locals().copy()
```

---

## Options File Template

**File: `custom_options/options_my_model.py`**

```python
from math import inf

# === Model Configuration ===
modelname = 'my_model'  # Must match model filename

# === Analysis Settings ===
checkObser = 1           # Check state observability (1=yes, 0=no)
maxLietime = inf         # Max time per Lie derivative (seconds)
nnzDerU = [inf]         # Known input derivative limits (per input)
nnzDerW = [inf]         # Unknown input derivative limits (per input)
prev_ident_pars = []    # Previously identified parameters

# === Visualization (optional) ===
MANIFOLD_PLOTTING = {
    'enabled': True,
    'max_2d_plots': 10,
    'max_3d_plots': 5,
    'resolution': 50
}
```

---

## Understanding Results

### Output Structure
```
results/
└── {model_name}/
    └── {timestamp}/
        ├── detailed_analysis.txt          # Text report
        ├── graphs/
        │   ├── identifiability_graph_full.png        # All variables
        │   └── identifiability_graph_params.png      # Parameters only
        ├── manifolds_2d/
        │   └── {param1}_vs_{param2}.png   # Pairwise relationships
        └── manifolds_3d/
            └── {param1}_{param2}_{param3}.png  # 3-parameter surfaces
```

### Interpreting Analysis

**Fully Identifiable:**
```
✓ STATUS: Fully identifiable and observable
All parameters and states can be uniquely determined.
```
→ Your model is well-designed. All parameters estimable!

**Partially Identifiable:**
```
⚠ STATUS: Partially identifiable
3 parameter combinations are unidentifiable

Unidentifiable relationships:
1. p1*p2 (only product is identifiable)
2. p3 + p4 (only sum is identifiable)
```
→ Individual params can't be estimated, but combinations can!

### Key Metrics

- **Nullspace Dimension (k)**: Number of unidentifiable directions
- **Identifiable Combinations**: `(total_params - k)` combinations you CAN estimate
- **Matrix Rank**: Higher is better (more information from measurements)

---

## Common Patterns

### Pattern 1: Parameter Products
```
Nullspace: [1, -1, 0, 0]  → p1 = p2
Identifiable: p1*p2, p1+p2
Unidentifiable: Individual p1, p2
```

### Pattern 2: Scaling Ambiguity
```
Nullspace: [1, 0, 0, 1]  → p1 = -p4
Means: p1 and p4 trade off
Identifiable: p1*V (parameter*volume products)
```

### Pattern 3: Sum Relationships
```
Nullspace: [1, 1, -1, 0]  → p1 + p2 = p3
Identifiable: Total rate (p1+p2), but not individual rates
```

---

## Troubleshooting

### Error: "No OIC matrix found"
```bash
# Solution: Run StrikePy first
nullstrike my_model options_my_model
```

### Error: "Model not found"
```bash
# Check model exists
ls custom_models/my_model.py

# Check options file
ls custom_options/options_my_model.py
```

### Slow Analysis
```bash
# Use checkpointing (automatic, but verify)
ls checkpoints/  # Should see .pkl files

# Or reduce maxLietime in options file
maxLietime = 100  # instead of inf
```

### Import Errors
```python
# Ensure working directory is project root
import os
os.chdir('/path/to/NullStrike')
```

---

## File Locations

```
NullStrike/
├── custom_models/          # Your model definitions here
├── custom_options/         # Your options files here
├── src/nullstrike/         # Package source code
│   ├── core/              # StrikePy engine
│   ├── analysis/          # Nullspace analysis
│   ├── visualization/     # Graphs and manifolds
│   └── cli/              # Command-line interface
├── results/               # Generated outputs
├── checkpoints/           # Cached analysis (auto-created)
└── docs/                  # Documentation
```

---

## Mathematical Quick Reference

### STRIKE-GOLDD Algorithm
```
1. Define system: dx/dt = f(x, p, u)
2. Define outputs: y = h(x, p)
3. Compute Lie derivatives:
   L⁰_f(h) = h
   L¹_f(h) = ∂h/∂x · f
   L²_f(h) = ∂(L_f h)/∂x · f
   ...
4. Build observability matrix O from Lie derivatives
5. Check rank → identifiability
```

### NullStrike Enhancement
```
1. Take O matrix from StrikePy
2. Compute nullspace: N = {v : O·v = 0}
3. If N = {0} → Fully identifiable ✓
4. If dim(N) = k:
   → k directions unidentifiable
   → (n - k) combinations identifiable
5. Find identifiable combos: nullspace(N)
```

### Key Equations
```
Observability Matrix:     O = [L⁰_f h, L¹_f h, ..., Lⁿ_f h]ᵀ
Nullspace:                N = nullspace(O)
Identifiable Directions:  I = nullspace(N) = row_space(O)
Dimension Relationship:   dim(N) + dim(I) = total_variables
```

---

## Example Workflow

### 1. Define Model
```python
# custom_models/simple_pk.py
import sympy as sym

# Two-compartment PK model
C1, C2 = sym.symbols('C1 C2')  # Concentrations
x = [[C1], [C2]]

k12, k21, V1, V2 = sym.symbols('k12 k21 V1 V2')
p = [[k12], [k21], [V1], [V2]]

h = [C1]  # Measure central compartment only

u = []
w = []

f = [
    [-(k12 + k21)*C1 + k21*C2],  # dC1/dt
    [k12*C1 - k21*C2]             # dC2/dt
]

variables_locales = locals().copy()
```

### 2. Create Options
```python
# custom_options/options_simple_pk.py
modelname = 'simple_pk'
checkObser = 1
maxLietime = inf
nnzDerU = []
nnzDerW = []
prev_ident_pars = []
```

### 3. Run Analysis
```bash
nullstrike simple_pk
```

### 4. Review Results
```
⚠ STATUS: Partially identifiable
2 parameter combinations are unidentifiable

Unidentifiable:
1. Individual V1, V2 (only products identifiable)
2. Individual k12, k21 (only sum identifiable)

Identifiable:
1. k12*V1 (transfer clearance)
2. k21*V2 (return clearance)
3. (k12+k21)*V1 (total clearance)
```

### 5. Reparameterize (if needed)
```python
# Use identifiable combinations as new parameters
CL1 = k12*V1  # Transfer clearance
CL2 = k21*V2  # Return clearance
Vd = V1       # Fix one volume, estimate the other via CLs
```

---

## Tips & Best Practices

### Model Design
- ✅ More measurements → better identifiability
- ✅ Measure states directly when possible
- ✅ Consider what's practically measurable
- ⚠️ Too few outputs → poor identifiability

### Parameter Estimation
- If nullspace_dim = k, fix k parameters to literature values
- Or use identifiable combinations as new parameters
- Or add k additional measurements

### Performance
- First run is slow (symbolic computation)
- Subsequent runs use checkpoint cache (10-100x faster)
- Use `maxLietime` to limit computation time
- Parameters-only mode (`-p`) is faster than full analysis

### Debugging
- Check `results/` for detailed text reports
- Look at graphs first for quick overview
- Use `--check` to validate environment
- Try built-in models (C2M, Bolie) to verify setup

---

## Common Use Cases

### Use Case 1: Check if Model is Well-Designed
```bash
nullstrike my_model
# → Look for "Fully identifiable" status
```

### Use Case 2: Find Which Parameter Combos to Estimate
```bash
nullstrike my_model
# → Read "Identifiable combinations" section
# → Use these as estimable parameters
```

### Use Case 3: Decide Which Measurements to Add
```bash
nullstrike my_model
# → Check nullspace_dimension
# → Add that many measurements to outputs (h)
```

### Use Case 4: Batch Analysis of Multiple Models
```python
models = ['model1', 'model2', 'model3']
for model in models:
    results = main(model, f'options_{model}')
    print(f"{model}: {results['nullspace_dimension']} unidentifiable")
```

---

## Key Concepts

| Concept | Meaning | Good/Bad |
|---------|---------|----------|
| **Fully Identifiable** | All parameters uniquely determined | ✓ Good |
| **Partially Identifiable** | Some combinations identifiable | ⚠️ Okay |
| **Unidentifiable** | Nothing can be determined | ✗ Bad |
| **Nullspace Dimension = 0** | Perfect identifiability | ✓ Best |
| **Nullspace Dimension = k** | k degrees of freedom lost | ⚠️ Fixable |
| **Rank = n** | Full rank observability matrix | ✓ Good |
| **Rank < n** | Rank deficient | ⚠️ Issues |

---

## Need More Help?

- **Full Documentation**: https://vipulsinghal02.github.io/NullStrike/
- **GitHub Issues**: https://github.com/vipulsinghal02/NullStrike/issues
- **Examples**: See `docs/examples/` or run `nullstrike --example`
- **Theory**: See `docs/theory/` for mathematical background

---

**Quick Reference Version 1.0** | Last Updated: 2025-01-15
