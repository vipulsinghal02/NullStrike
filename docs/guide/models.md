# Model Definition Guide

This guide explains how to define models for NullStrike analysis. Understanding the model definition format is essential for analyzing your own dynamical systems.

## Model Structure Overview

NullStrike analyzes nonlinear dynamical systems of the form:

$$\begin{align}
\dot{x}(t) &= f(x(t), p, u(t), w(t)) \quad &x(0) = x_0(p) \\
y(t) &= h(x(t), p, u(t), w(t))
\end{align}$$

Every model file must define these components using SymPy symbolic variables.

## Required Model Components

### States (`x`)

**Definition**: Internal variables that evolve over time according to differential equations.

```python
import sympy as sym

# Define state variables
x1 = sym.Symbol('x1')  # Concentration of species A
x2 = sym.Symbol('x2')  # Concentration of species B

# State vector (must be list of lists)
x = [[x1], [x2]]
```

**Important notes**:
- States represent quantities that change over time
- Must be defined as a **list of lists**: `[[x1], [x2]]`, not `[x1, x2]`
- Choose meaningful names that reflect the physical quantities

### Parameters (`p`)

**Definition**: Unknown constants that characterize the system but don't change over time.

```python
# Define parameters
k1 = sym.Symbol('k1')    # Reaction rate constant
k2 = sym.Symbol('k2')    # Degradation rate
V1 = sym.Symbol('V1')    # Volume of compartment 1

# Parameter vector (must be list of lists)
p = [[k1], [k2], [V1]]
```

**Guidelines**:
- Parameters are what you want to determine from data
- Include only parameters that appear in the dynamics or outputs
- Use descriptive names (e.g., `k_forward` rather than `p1`)

### Dynamics (`f`)

**Definition**: Right-hand side of the differential equations $\dot{x} = f(x, p, u, w)$.

```python
# System dynamics
f = [
    [-k1*x1 + k2*x2],        # dx1/dt
    [k1*x1 - k2*x2]          # dx2/dt
]
```

**Requirements**:
- Must be a list with one entry per state variable
- Each entry is a list containing the symbolic expression
- Expressions can involve states, parameters, inputs, and unknown inputs

### Outputs (`h`)

**Definition**: Observable quantities that can be measured experimentally.

```python
# Output equations
h = [x1 + x2]  # Total concentration (observable)
```

**Important**:
- Outputs are what you can actually measure in experiments
- Can be functions of states, parameters, and inputs
- The identifiability analysis determines what can be learned from these measurements

### Inputs (`u`)

**Definition**: Known external signals or control inputs.

```python
# Known inputs (empty if none)
u1 = sym.Symbol('u1')
u = [u1]  # or u = [] if no inputs
```

**Examples**:
- Drug administration rates in pharmacokinetics
- Temperature changes in chemical reactions
- Light intensity in biological systems

### Unknown Inputs (`w`)

**Definition**: Unknown disturbances or unmeasured inputs.

```python
# Unknown inputs (usually empty)
w = []  # or define symbols if present
```

**Use cases**:
- Environmental disturbances
- Measurement noise (if modeled)
- Unknown external influences

### Required Boilerplate

Every model file must end with:

```python
# Required for NullStrike to access variables
variables_locales = locals().copy()
```

This allows NullStrike to import all defined variables from your model file.

## Complete Model Template

```python
import sympy as sym

# ============================================================================
# MODEL: [Your Model Name]
# Description: [Brief description of what the model represents]
# ============================================================================

# State variables
x1 = sym.Symbol('x1')  # [Description of x1]
x2 = sym.Symbol('x2')  # [Description of x2]
# ... add more states as needed

# Parameter variables  
p1 = sym.Symbol('p1')  # [Description of p1]
p2 = sym.Symbol('p2')  # [Description of p2]
# ... add more parameters as needed

# Input variables (if any)
u1 = sym.Symbol('u1')  # [Description of u1, if present]

# ============================================================================
# SYSTEM DEFINITION
# ============================================================================

# State vector
x = [[x1], [x2]]  # List of lists format

# Parameter vector
p = [[p1], [p2]]  # List of lists format

# Output vector (what you can measure)
h = [x1]  # Example: only x1 is observable

# Input vector (empty if no inputs)
u = [u1]  # or u = [] if no inputs

# Unknown input vector (usually empty)
w = []

# System dynamics: dx/dt = f(x, p, u, w)
f = [
    [p1*x1 - p2*x1*x2],     # dx1/dt
    [p2*x1*x2 - p1*x2]      # dx2/dt
]

# Required boilerplate
variables_locales = locals().copy()
```

## Model Examples by Domain

### Pharmacokinetic Models

=== "One-Compartment Model"

    ```python
    import sympy as sym
    
    # States
    A = sym.Symbol('A')  # Amount in central compartment
    x = [[A]]
    
    # Parameters
    k10 = sym.Symbol('k10')  # Elimination rate
    V = sym.Symbol('V')      # Volume of distribution
    p = [[k10], [V]]
    
    # Inputs
    u1 = sym.Symbol('u1')    # Drug input rate
    u = [u1]
    
    # Outputs
    h = [A/V]  # Concentration = Amount/Volume
    
    # Dynamics
    f = [[-k10*A + u1]]  # dA/dt
    
    w = []
    variables_locales = locals().copy()
    ```

=== "Two-Compartment Model"

    ```python
    import sympy as sym
    
    # States
    A1 = sym.Symbol('A1')  # Central compartment
    A2 = sym.Symbol('A2')  # Peripheral compartment
    x = [[A1], [A2]]
    
    # Parameters
    k10 = sym.Symbol('k10')  # Elimination
    k12 = sym.Symbol('k12')  # Central to peripheral
    k21 = sym.Symbol('k21')  # Peripheral to central
    V1 = sym.Symbol('V1')    # Central volume
    p = [[k10], [k12], [k21], [V1]]
    
    # Inputs
    u1 = sym.Symbol('u1')
    u = [u1]
    
    # Outputs
    h = [A1/V1]  # Central concentration
    
    # Dynamics
    f = [
        [-(k10 + k12)*A1 + k21*A2 + u1],  # dA1/dt
        [k12*A1 - k21*A2]                 # dA2/dt
    ]
    
    w = []
    variables_locales = locals().copy()
    ```

### Chemical Reaction Systems

=== "Michaelis-Menten Kinetics"

    ```python
    import sympy as sym
    
    # States
    S = sym.Symbol('S')  # Substrate concentration
    P = sym.Symbol('P')  # Product concentration
    x = [[S], [P]]
    
    # Parameters
    Vmax = sym.Symbol('Vmax')  # Maximum velocity
    Km = sym.Symbol('Km')      # Michaelis constant
    E0 = sym.Symbol('E0')      # Initial enzyme concentration
    p = [[Vmax], [Km], [E0]]
    
    # No inputs
    u = []
    
    # Outputs
    h = [P]  # Product is observable
    
    # Dynamics (Michaelis-Menten kinetics)
    f = [
        [-Vmax*S/(Km + S)],     # dS/dt
        [Vmax*S/(Km + S)]       # dP/dt
    ]
    
    w = []
    variables_locales = locals().copy()
    ```

=== "Enzyme Binding Model"

    ```python
    import sympy as sym
    
    # States
    E = sym.Symbol('E')   # Free enzyme
    S = sym.Symbol('S')   # Free substrate
    ES = sym.Symbol('ES') # Enzyme-substrate complex
    P = sym.Symbol('P')   # Product
    x = [[E], [S], [ES], [P]]
    
    # Parameters
    k1 = sym.Symbol('k1')     # Forward binding rate
    k_1 = sym.Symbol('k_1')   # Reverse binding rate
    k2 = sym.Symbol('k2')     # Catalytic rate
    p = [[k1], [k_1], [k2]]
    
    # No inputs
    u = []
    
    # Outputs
    h = [P]  # Only product is measurable
    
    # Dynamics
    f = [
        [-k1*E*S + (k_1 + k2)*ES],   # dE/dt
        [-k1*E*S + k_1*ES],          # dS/dt
        [k1*E*S - (k_1 + k2)*ES],    # dES/dt
        [k2*ES]                      # dP/dt
    ]
    
    w = []
    variables_locales = locals().copy()
    ```

### Biological Systems

=== "Gene Expression Model"

    ```python
    import sympy as sym
    
    # States
    mRNA = sym.Symbol('mRNA')     # mRNA concentration
    protein = sym.Symbol('protein') # Protein concentration
    x = [[mRNA], [protein]]
    
    # Parameters
    k_transcr = sym.Symbol('k_transcr')   # Transcription rate
    k_transl = sym.Symbol('k_transl')     # Translation rate
    k_deg_m = sym.Symbol('k_deg_m')       # mRNA degradation
    k_deg_p = sym.Symbol('k_deg_p')       # Protein degradation
    p = [[k_transcr], [k_transl], [k_deg_m], [k_deg_p]]
    
    # No inputs
    u = []
    
    # Outputs
    h = [protein]  # Protein fluorescence
    
    # Dynamics
    f = [
        [k_transcr - k_deg_m*mRNA],              # dmRNA/dt
        [k_transl*mRNA - k_deg_p*protein]       # dprotein/dt
    ]
    
    w = []
    variables_locales = locals().copy()
    ```

=== "Predator-Prey Model"

    ```python
    import sympy as sym
    
    # States
    prey = sym.Symbol('prey')        # Prey population
    predator = sym.Symbol('predator') # Predator population
    x = [[prey], [predator]]
    
    # Parameters
    r = sym.Symbol('r')      # Prey growth rate
    a = sym.Symbol('a')      # Predation rate
    b = sym.Symbol('b')      # Conversion efficiency
    m = sym.Symbol('m')      # Predator mortality
    p = [[r], [a], [b], [m]]
    
    # No inputs
    u = []
    
    # Outputs (both populations observable)
    h = [prey, predator]
    
    # Dynamics (Lotka-Volterra)
    f = [
        [r*prey - a*prey*predator],      # dprey/dt
        [b*a*prey*predator - m*predator] # dpredator/dt
    ]
    
    w = []
    variables_locales = locals().copy()
    ```

## Advanced Model Features

### Models with Inputs

```python
import sympy as sym

# States
x1 = sym.Symbol('x1')
x = [[x1]]

# Parameters
k = sym.Symbol('k')
p = [[k]]

# Time-varying input
u1 = sym.Symbol('u1')
u = [u1]

# Output
h = [x1]

# Dynamics with input
f = [[-k*x1 + u1]]  # Input appears in dynamics

w = []
variables_locales = locals().copy()
```

### Multiple Outputs

```python
import sympy as sym

# States
x1, x2 = sym.symbols('x1 x2')
x = [[x1], [x2]]

# Parameters
p1, p2 = sym.symbols('p1 p2')
p = [[p1], [p2]]

# Multiple outputs
h = [x1, x2, x1 + x2]  # Three different measurements

# Dynamics
f = [[-p1*x1], [p2*x1 - p1*x2]]

u = []
w = []
variables_locales = locals().copy()
```

### Initial Conditions as Parameters

If initial conditions are unknown, treat them as parameters:

```python
import sympy as sym

# States
x1, x2 = sym.symbols('x1 x2')
x = [[x1], [x2]]

# Parameters including initial conditions
k1, k2 = sym.symbols('k1 k2')
x1_0, x2_0 = sym.symbols('x1_0 x2_0')  # Initial conditions
p = [[k1], [k2], [x1_0], [x2_0]]

# Outputs
h = [x1]

# Dynamics
f = [[-k1*x1 + k2*x2], [k1*x1 - k2*x2]]

u = []
w = []
variables_locales = locals().copy()
```

## Common Mistakes and Solutions

### Mistake 1: Wrong Vector Format

**Incorrect**:
```python
x = [x1, x2]  # Wrong: not list of lists
p = [p1, p2]  # Wrong: not list of lists
```

**Correct**:
```python
x = [[x1], [x2]]  # Correct: list of lists
p = [[p1], [p2]]  # Correct: list of lists
```

### Mistake 2: Inconsistent Dimensions

**Incorrect**:
```python
x = [[x1], [x2]]     # 2 states
f = [[-k*x1]]        # Only 1 equation
```

**Correct**:
```python
x = [[x1], [x2]]           # 2 states
f = [[-k*x1], [k*x1]]      # 2 equations
```

### Mistake 3: Missing Variables in Expressions

**Incorrect**:
```python
# k2 is used in dynamics but not defined in parameters
p = [[k1]]
f = [[-k1*x1 + k2*x2]]  # k2 undefined
```

**Correct**:
```python
p = [[k1], [k2]]        # Both parameters defined
f = [[-k1*x1 + k2*x2]]  # All variables defined
```

### Mistake 4: Forgetting Required Boilerplate

**Incorrect**:
```python
# ... model definition ...
# Missing: variables_locales = locals().copy()
```

**Correct**:
```python
# ... model definition ...
variables_locales = locals().copy()  # Required!
```

## Model Validation

### Syntax Checking

Before running analysis, verify your model:

```python
# Check that all required variables are defined
required_vars = ['x', 'p', 'h', 'f', 'u', 'w', 'variables_locales']
for var in required_vars:
    if var not in locals():
        print(f"Error: {var} not defined")

# Check dimensions
if len(x) != len(f):
    print(f"Error: {len(x)} states but {len(f)} equations")

# Check that all symbols in f are defined
used_symbols = set()
for eq in f:
    used_symbols.update(eq[0].free_symbols)

defined_symbols = set()
for state in x:
    defined_symbols.update(state[0].free_symbols)
for param in p:
    defined_symbols.update(param[0].free_symbols)

undefined = used_symbols - defined_symbols
if undefined:
    print(f"Warning: Undefined symbols: {undefined}")
```

### Physical Reasonableness

Verify that your model makes physical sense:

- **Mass balance**: For chemical systems, check conservation laws
- **Positivity**: Ensure concentrations/populations stay positive
- **Units**: Verify dimensional consistency
- **Parameter bounds**: Check that parameters have reasonable ranges

## Model Organization

### File Naming Convention

Place model files in `custom_models/` directory:

```
custom_models/
├── my_pharmacokinetic_model.py
├── enzyme_kinetics_v2.py
├── population_dynamics.py
└── ...
```

### Documentation Within Models

Include clear documentation:

```python
"""
Two-Compartment Pharmacokinetic Model

This model describes drug distribution between central and peripheral
compartments with first-order elimination.

States:
    A1: Amount in central compartment [mg]
    A2: Amount in peripheral compartment [mg]

Parameters:
    k10: Elimination rate constant [1/h]
    k12: Distribution rate to peripheral [1/h] 
    k21: Distribution rate from peripheral [1/h]
    V1: Central compartment volume [L]

Outputs:
    C1: Central compartment concentration [mg/L]

Reference: Gabrielsson & Weiner, Pharmacokinetic/Pharmacodynamic 
           Data Analysis, 5th ed., Chapter 3.
"""

import sympy as sym
# ... rest of model definition
```

## Next Steps

After defining your model:

1. **Create options file**: See [Configuration Guide](configuration.md)
2. **Test the model**: Run with `--parameters-only` flag first
3. **Run full analysis**: Generate complete results with visualizations
4. **Interpret results**: See [Results Interpretation](../results/interpretation.md)

---

## Further Reading

- **[Configuration Guide](configuration.md)**: Set up analysis options
- **[CLI Usage](cli-usage.md)**: Command-line interface details
- **[Examples](../examples/simple.md)**: More model examples
- **[Theory Overview](../theory/overview.md)**: Mathematical background

!!! tip "Model Development Strategy"
    
    Start with simple models to verify your approach, then gradually add complexity. Use the `--parameters-only` flag for quick testing during model development.