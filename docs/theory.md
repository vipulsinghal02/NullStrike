# Mathematical Theory

This page explains the mathematical foundations of NullStrike's structural identifiability analysis.

## What is Structural Identifiability?

Structural identifiability analysis determines whether model parameters can be uniquely determined from measurements, independent of actual data values or noise. It's a theoretical property of the model structure itself.

### Key Questions

1. Can parameters be uniquely determined from the outputs?
2. Which parameter combinations are identifiable?
3. How many measurements are needed for full identifiability?

## STRIKE-GOLDD Algorithm

### The Observability-Identifiability Matrix

STRIKE-GOLDD (STRuctural Identifiability taken as Extended-Generalized Observability with Lie Derivatives and Decomposition) uses Lie derivatives to build an observability matrix.

**Given a dynamical system:**
```
dx/dt = f(x, p, u)    # State dynamics
y = h(x, p)           # Measurements
```

Where:
- `x` = states
- `p` = parameters
- `u` = known inputs
- `y` = outputs (measurements)

### Lie Derivatives

The **Lie derivative** measures how the output changes along the system dynamics:

```
L⁰_f(h) = h                    # Output itself
L¹_f(h) = ∂h/∂x · f            # First derivative
L²_f(h) = ∂(L_f h)/∂x · f      # Second derivative
...
Lⁿ_f(h) = continues until rank saturates
```

### Building the OIC Matrix

The **Observability-Identifiability-Controllability (OIC) Matrix** is:

```
O = [L⁰_f(h), L¹_f(h), L²_f(h), ..., Lⁿ_f(h)]ᵀ
```

Each row is a Lie derivative of the output.

**Columns represent:** [states, parameters, unknown inputs]

**Example for 2 states, 2 parameters:**
```
        [x1, x2, p1, p2]
O = [
      [h₁, h₂, 0,  0 ],   ← L⁰_f(h)
      [..., ..., ..., ...], ← L¹_f(h)
      [..., ..., ..., ...]  ← L²_f(h)
    ]
```

### Jacobian and Identifiability

Compute the Jacobian:
```
J = ∂O/∂[x, p, w]
```

**Rank analysis:**
- If rank(J) = n_states + n_params → Fully identifiable
- If rank(J) < n_states + n_params → Some unidentifiable

## Nullspace Analysis (NullStrike Enhancement)

Traditional STRIKE-GOLDD identifies which individual parameters are unidentifiable. **NullStrike goes further** by finding which parameter combinations ARE identifiable.

### The Nullspace

The **nullspace** of the OIC matrix O contains all directions that don't affect the output:

```
N = nullspace(O) = {v : O·v = 0}
```

**Interpretation:**
- Each nullspace vector represents an unidentifiable direction
- If v ∈ N, then changing variables along v doesn't change outputs
- **Dimension of N = k** means k degrees of freedom are lost

### Finding Identifiable Combinations

The **row space** of O (orthogonal complement of nullspace) contains identifiable directions:

```
I = nullspace(N) = row_space(O)
```

**Key relationship:**
```
dim(N) + dim(I) = total_variables
```

If nullspace dimension = k, then **(n - k) independent combinations** are identifiable.

### Example: Two-Compartment Model

**Model:**
```
dC1/dt = -(k12 + k21)*C1 + k21*C2
dC2/dt = k12*C1 - k21*C2
y = C1  # Only measure central compartment
```

**Parameters:** [k12, k21, V1, V2]

**Nullspace analysis finds:**
```
Nullspace dimension = 1

Unidentifiable direction:
v = [0, 0, 1, -1]  → V1 and V2 scale together

Identifiable combinations (3 total):
1. k12*V1 (transfer clearance)
2. k21*V2 (return clearance)
3. (k12+k21)*V1 (total clearance)
```

**Conclusion:** Individual volumes unidentifiable, but clearances ARE identifiable!

### Mathematical Procedure

**Step 1:** Compute nullspace of O
```python
import sympy as sym
N = O.nullspace()  # Returns list of basis vectors
```

**Step 2:** Build nullspace matrix (rows = nullspace vectors)
```python
N_matrix = sym.Matrix(N)  # Each row is a nullspace vector
```

**Step 3:** Compute nullspace of N_matrix
```python
I = N_matrix.nullspace()  # Identifiable directions
```

**Step 4:** Interpret results
```python
# Parse nullspace vectors to find relationships
for v in N:
    # v tells you which variables trade off
    # e.g., v = [1, -1, 0, 0] means p1 = p2 (unidentifiable)
```

## Common Identifiability Patterns

### Pattern 1: Scaling Ambiguity

**Nullspace vector:** `[1, 0, 0, -1]`

**Meaning:** `p1 = p4` (scale together)

**Identifiable:** `p1*p4`, `p1+p4`, `p1/p4`

**Example:** Volume and concentration scale together

### Pattern 2: Sum Relationships

**Nullspace vector:** `[1, 1, -1, 0]`

**Meaning:** `p1 + p2 = p3`

**Identifiable:** Total (p1+p2), but not individual rates

**Example:** Multiple pathways with same net effect

### Pattern 3: Product Relationships

**Nullspace vector:** `[1, 1, 0, 0]`  (in log space)

**Meaning:** `p1*p2` constant

**Identifiable:** Product, but not individual factors

**Example:** Rate constant × enzyme concentration

## Practical Implications

### Reparameterization

If nullspace shows `p1*p2` is identifiable but not p1, p2:

**Original parameters:**
```python
p = [p1, p2, p3]
```

**Reparameterize:**
```python
p_new = [p1*p2, p2, p3]  # Now first param is identifiable
```

### Measurement Design

If nullspace dimension = k:

**Option 1:** Add k independent measurements
```python
# Before: h = [x1]
# After:  h = [x1, x2, x1+x2]  # k=2 additional measurements
```

**Option 2:** Fix k parameters to known values
```python
# Fix V1 = 1.0 (normalize)
# Now estimate [k12, k21, V2] instead of [k12, k21, V1, V2]
```

### Parameter Estimation

Focus on **identifiable combinations** rather than individual parameters:

```python
# Don't estimate: [k12, k21, V1, V2]
# Do estimate: [k12*V1, k21*V2, V1]  # Identifiable!
```

## Symbolic vs Numerical Analysis

### Why Symbolic?

NullStrike uses **symbolic computation** (SymPy) rather than numerical:

**Advantages:**
- Exact mathematical relationships
- No numerical precision errors
- Works for any parameter values
- Reveals structural properties

**Disadvantages:**
- Slower for large systems
- Can produce complex expressions

### Simplification Strategies

Symbolic expressions can be complex. NullStrike:

1. **Rationalizes** expressions to common denominators
2. **Simplifies** using SymPy's simplify functions
3. **Eliminates** identified variables to reduce complexity
4. **Caches** results to avoid recomputation

## Limitations and Assumptions

### Assumptions

1. **Model is correct:** Structure represents true system
2. **Measurements are perfect:** No noise considerations
3. **Infinite data:** Asymptotic identifiability
4. **Local analysis:** May not hold globally

### When Identifiability Fails

**Insufficient measurements:**
- Need more outputs h
- Or measure at more time points

**Model structure:**
- Some models are inherently unidentifiable
- Consider simplifying or adding constraints

**Nonlinear effects:**
- Global vs local identifiability may differ
- Nullspace shows local properties

## Advanced Topics

### Observable but Unidentifiable

A state can be **observable** (reconstructible from outputs) but a parameter affecting it can be **unidentifiable**:

```python
dx/dt = -p*x,  y = x
# x is observable (directly measured)
# p is unidentifiable from single trajectory (need IC or multiple experiments)
```

### Practical Identifiability

**Structural identifiability** (what NullStrike analyzes) ≠ **Practical identifiability**

Practical also considers:
- Measurement noise
- Limited data
- Experimental constraints

Use structural analysis first, then check practical identifiability with simulations.

### Extended Observability

For **unknown inputs** w, extend the nullspace analysis to include them:

```
Variables = [states, parameters, unknown_inputs]
```

Nullspace shows which combinations of parameters AND inputs are identifiable.

## Mathematical References

### Key Equations Summary

```
Lie Derivative:
Lⁿ_f(h) = ∂(Lⁿ⁻¹_f h)/∂x · f

OIC Matrix:
O = [L⁰_f(h), L¹_f(h), ..., Lⁿ_f(h)]ᵀ

Nullspace:
N = {v : O·v = 0}

Identifiable Space:
I = {v : N·v = 0} = row_space(O)

Dimension Relation:
dim(N) + rank(O) = n_variables
```

### Further Reading

- **Original STRIKE-GOLDD:** Villaverde et al. (2016) "STRIKE-GOLDD: Structural Identifiability"
- **Lie Derivatives:** Hermann & Krener (1977) "Nonlinear Controllability and Observability"
- **Identifiability Theory:** Walter & Pronzato (1997) "Identification of Parametric Models"

## Quick Reference

| Concept | Meaning | Formula |
|---------|---------|---------|
| Lie Derivative | Rate of change along dynamics | `L_f(h) = ∂h/∂x · f` |
| OIC Matrix | Observability matrix | `O = [h, L_f h, L²_f h, ...]ᵀ` |
| Nullspace | Unidentifiable directions | `N = {v : O·v = 0}` |
| Nullspace Dim | # unidentifiable combinations | `k = dim(N)` |
| Identifiable Combos | # estimable combinations | `n - k` |
| Full Rank | Fully identifiable | `rank(O) = n` |

**Next:** See [Examples](examples.md) for practical applications of this theory.
