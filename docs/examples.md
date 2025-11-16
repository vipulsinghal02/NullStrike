# Examples & Tutorials

Step-by-step examples showing how to use NullStrike for different types of analysis.

## Example 1: Two-Compartment PK Model (C2M)

This is the classic pharmacokinetic example that demonstrates parameter combination identifiability.

### Model Definition

**File:** `custom_models/C2M.py`

```python
import sympy as sym

# States (concentrations in compartments)
C1, C2 = sym.symbols('C1 C2')
x = [[C1], [C2]]

# Parameters
k12, k21, k10, V1 = sym.symbols('k12 k21 k10 V1')
p = [[k12], [k21], [k10], [V1]]

# Output: Only measure central compartment
h = [C1]

# Inputs
u = []
w = []

# Dynamics
f = [
    [-(k12 + k21 + k10)*C1 + k21*C2],  # dC1/dt
    [k12*C1 - k21*C2]                    # dC2/dt
]

variables_locales = locals().copy()
```

### Options Configuration

**File:** `custom_options/options_C2M.py`

```python
from math import inf

modelname = 'C2M'
checkObser = 1
maxLietime = inf
nnzDerU = []
nnzDerW = []
prev_ident_pars = []

MANIFOLD_PLOTTING = {
    'enabled': True,
    'max_2d_plots': 10,
    'max_3d_plots': 5
}
```

### Run Analysis

```bash
nullstrike C2M
```

### Results Interpretation

**Output:**
```
⚠ STATUS: Partially identifiable
1 parameter combination is unidentifiable

Unidentifiable relationships:
1. k12 and k21 appear only in sum (k12+k21)

Identifiable combinations:
1. k12*V1 (transfer clearance)
2. k21*V1 (return clearance)
3. (k12+k21+k10)*V1 (total clearance)
```

**Key Insight:** Individual rate constants are unidentifiable, but **clearances** (rate × volume) ARE identifiable!

### Practical Application

**Reparameterize for estimation:**
```python
# Don't estimate: [k12, k21, k10, V1]
# Do estimate: [CL12=k12*V1, CL21=k21*V1, CL10=k10*V1, V1]
```

Or fix one volume:
```python
# Fix V1 = 1.0 (normalize)
# Estimate: [k12, k21, k10] as apparent clearances
```

---

## Example 2: Bolie Glucose-Insulin Model

A more complex model showing multiple unidentifiable combinations.

### Model Overview

```python
# States: Glucose (G), Insulin (I)
# Parameters: Multiple rate constants and sensitivities
# Outputs: Measure both glucose and insulin
```

### Run Analysis

```bash
nullstrike Bolie
```

### Key Findings

**Typical results:**
- Some parameter products are identifiable (not individuals)
- Sensitivity ratios may be estimable
- May need additional measurements or constraints

### Workflow

1. Run initial analysis
2. Check nullspace dimension
3. If partially identifiable:
   - Review identifiable combinations
   - Decide: Fix parameters? Add measurements? Reparameterize?
4. Re-analyze with modifications

---

## Example 3: Custom Model - Simple Enzyme Kinetics

Let's build a complete custom model from scratch.

### Step 1: Define the Model

**File:** `custom_models/enzyme_kinetics.py`

```python
import sympy as sym

# Substrate and Product concentrations
S, P = sym.symbols('S P')
x = [[S], [P]]

# Michaelis-Menten parameters
Vmax, Km = sym.symbols('Vmax Km')
p = [[Vmax], [Km]]

# Measure product formation
h = [P]

# No inputs
u = []
w = []

# Michaelis-Menten kinetics
f = [
    [-Vmax*S/(Km + S)],     # dS/dt
    [Vmax*S/(Km + S)]       # dP/dt
]

variables_locales = locals().copy()
```

### Step 2: Create Options

**File:** `custom_options/options_enzyme_kinetics.py`

```python
from math import inf

modelname = 'enzyme_kinetics'
checkObser = 1
maxLietime = 100  # Limit time for this simple model
nnzDerU = []
nnzDerW = []
prev_ident_pars = []
```

### Step 3: Run Analysis

```bash
nullstrike enzyme_kinetics
```

### Step 4: Interpret Results

**Expected:** Vmax and Km are likely identifiable individually if you measure product over time.

**If not:** Consider measuring substrate S as well:
```python
h = [S, P]  # Measure both
```

---

## Example 4: Fixing Unidentifiable Parameters

When you have unidentifiable parameters, you have several options.

### Option A: Fix to Literature Values

```python
# In options file
prev_ident_pars = ['V1']  # Tell NullStrike V1 is known

# Then in your model, treat V1 as constant
V1_value = 1.0  # From literature
```

### Option B: Reparameterize

**Original model (unidentifiable):**
```python
p = [[k1], [k2], [V]]
f = [[k1*x[0]*V], ...]  # k1*V appears together
```

**Reparameterized (identifiable):**
```python
# Define new parameter
CL = sym.Symbol('CL')  # CL = k1*V
p = [[CL], [k2]]
f = [[CL*x[0]], ...]  # Use CL directly
```

### Option C: Add Measurements

**Current (unidentifiable):**
```python
h = [x1]  # Only measure first state
```

**Modified (better identifiability):**
```python
h = [x1, x2]  # Measure both states
```

---

## Example 5: Batch Processing Multiple Models

Analyze several models programmatically.

### Python Script

```python
from nullstrike.cli.complete_analysis import main
import json

models = [
    ('C2M', 'options_C2M'),
    ('Bolie', 'options_Bolie'),
    ('enzyme_kinetics', 'options_enzyme_kinetics')
]

results_summary = {}

for model_name, options_file in models:
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}...")
    print('='*60)

    try:
        results = main(model_name, options_file)

        results_summary[model_name] = {
            'nullspace_dimension': results.get('nullspace_dimension', 0),
            'fully_identifiable': results.get('fully_identifiable', False),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        results_summary[model_name] = {
            'status': 'failed',
            'error': str(e)
        }

# Save summary
with open('batch_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "="*60)
print("BATCH ANALYSIS SUMMARY")
print("="*60)
for model, result in results_summary.items():
    if result['status'] == 'success':
        if result['fully_identifiable']:
            print(f"✓ {model}: Fully identifiable")
        else:
            k = result['nullspace_dimension']
            print(f"⚠ {model}: Nullspace dimension = {k}")
    else:
        print(f"✗ {model}: Failed - {result['error']}")
```

---

## Example 6: Interpreting Visualizations

### Graph Visualizations

**Files:** `results/{model}/graphs/identifiability_graph_*.png`

**Interpretation:**
- **Nodes:** Parameters/states
- **Edges:** Relationships from nullspace
- **Clusters:** Groups of interdependent parameters

**Example:**
- If k12 and V1 are connected → They appear together in identifiable combinations
- Isolated nodes → Individually identifiable

### 2D Manifold Plots

**Files:** `results/{model}/manifolds_2d/*.png`

**Interpretation:**
- **Linear relationship:** Parameters sum or difference
- **Hyperbolic curve:** Parameters multiply
- **No constraint:** Parameters independent (identifiable)

**Example:**
- Straight line in (p1, p2) plot → p1 + p2 = constant (unidentifiable individually)
- Scattered points → Both identifiable

### 3D Manifold Plots

**Files:** `results/{model}/manifolds_3d/*.png`

**Interpretation:**
- **Planar surface:** Linear constraint among 3 parameters
- **Curved surface:** Nonlinear relationship
- **No clear surface:** All 3 identifiable

---

## Example 7: Using Checkpoints for Speed

NullStrike automatically caches results. Here's how it works:

### First Run (Slow)

```bash
time nullstrike C2M
# Takes 30-60 seconds (symbolic computation)
```

**Output:**
```
Running StrikePy analysis...
Computing Lie derivatives...
Analyzing nullspace...
Saving checkpoint...
✓ Analysis complete
```

### Second Run (Fast)

```bash
time nullstrike C2M
# Takes 2-3 seconds (loads from cache)
```

**Output:**
```
Found valid checkpoint - using cached analysis
✓ Analysis complete
```

### Clear Cache

```bash
rm -rf checkpoints/C2M/
nullstrike C2M  # Will recompute
```

### When Cache is Invalidated

Cache automatically clears when you modify:
- Model file (`custom_models/C2M.py`)
- Options file (`custom_options/options_C2M.py`)

---

## Example 8: Troubleshooting Common Issues

### Issue: "No OIC matrix found"

**Cause:** StrikePy hasn't generated results yet

**Solution:**
```bash
# Make sure you run the full analysis
nullstrike my_model
```

### Issue: Analysis Takes Forever

**Cause:** Symbolic computation is slow for large systems

**Solutions:**
```python
# 1. Limit computation time in options
maxLietime = 300  # 5 minutes max per Lie derivative

# 2. Use parameters-only mode
```
```bash
nullstrike my_model -p
```

```python
# 3. Disable visualizations
MANIFOLD_PLOTTING = {'enabled': False}
```

### Issue: Unexpected Nullspace Dimension

**Cause:** Model may have structural issues

**Check:**
1. Model definition is correct
2. Outputs h include enough measurements
3. Dynamics f are properly defined
4. Review detailed_analysis.txt for clues

---

## Example 9: Model Design Best Practices

### Good Model Design

```python
# Multiple independent measurements
h = [x1, x2, x1 + x2]  # Redundancy helps

# Measure states directly when possible
h = [x1, x2]  # Better than just [x1]

# Structured parameters
p = [[k1*V], [k2*V], [V]]  # Groups related params
```

### Poor Model Design

```python
# Single measurement only
h = [x1]  # Limited information

# Unmeasured states
h = [x1]  # But x2, x3 are critical for identifiability

# Too many parameters
p = [[p1], [p2], ..., [p20]]  # Unless many measurements
```

---

## Next Steps

After working through these examples:

1. **Try your own model:** Use patterns from Examples 1-3
2. **Check theory:** See [Theory](theory.md) for mathematical details
3. **Use reference:** See [Reference](reference.md) for complete API
4. **Quick recall:** Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) cheat sheet

## Example Model Library

Pre-built models in `custom_models/`:
- `C2M.py` - Two-compartment pharmacokinetics
- `Bolie.py` - Glucose-insulin dynamics
- `calibration_single.py` - Simple calibration example

Each has corresponding options in `custom_options/`.
