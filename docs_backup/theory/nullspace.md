# Nullspace Analysis

Nullspace analysis is NullStrike's key innovation that extends traditional identifiability analysis. While classical methods tell you which parameters are unidentifiable, nullspace analysis reveals **which parameter combinations are identifiable** and provides the geometric structure of parameter constraints.

## The Fundamental Insight

Consider the identifiability matrix $\mathcal{J} \in \mathbb{R}^{k \times m}$ where:
- $k$ = number of observability conditions (rows of extended observability matrix)
- $m$ = number of parameters

If $\text{rank}(\mathcal{J}) < m$, then some parameters are unidentifiable. Traditional analysis stops here, but nullspace analysis asks: **What can we learn from the structure of unidentifiable directions?**

## Mathematical Framework

### The Nullspace

The **nullspace** (kernel) of $\mathcal{J}$ contains all parameter perturbations that don't affect the observable outputs:

$$\mathcal{N}(\mathcal{J}) = \{v \in \mathbb{R}^m : \mathcal{J}v = 0\}$$

**Physical interpretation**: If $v \in \mathcal{N}(\mathcal{J})$, then changing parameters in direction $v$ produces no change in the model's input-output behavior.

### Nullspace Dimension

The dimension of the nullspace is:

$$\dim(\mathcal{N}) = m - \text{rank}(\mathcal{J})$$

This tells us:
- How many parameters are "truly redundant"
- How many constraints the data provides on parameters
- The geometric dimension of unidentifiable directions

### Nullspace Basis

A **basis** $\{v_1, v_2, \ldots, v_d\}$ for $\mathcal{N}(\mathcal{J})$ provides the fundamental unidentifiable directions. Each basis vector $v_i = [v_{i1}, v_{i2}, \ldots, v_{im}]^T$ defines a **parameter constraint**:

$$v_{i1} p_1 + v_{i2} p_2 + \cdots + v_{im} p_m = \text{constant}$$

## Identifiable Parameter Combinations

### The Orthogonal Complement

The **identifiable subspace** is the orthogonal complement of the nullspace:

$$\mathcal{I} = \mathcal{N}^{\perp} = \{w \in \mathbb{R}^m : w^T v = 0 \text{ for all } v \in \mathcal{N}\}$$

**Key insight**: While individual parameters in $\mathcal{N}$ are unidentifiable, their **projections onto $\mathcal{I}$** are identifiable!

### Computing Identifiable Combinations

Given a nullspace basis $V = [v_1, v_2, \ldots, v_d]$, the identifiable subspace has basis:

$$W = \text{null}(V^T)$$

Each row $w_i$ of $W$ defines an **identifiable parameter combination**:

$$w_{i1} p_1 + w_{i2} p_2 + \cdots + w_{im} p_m = \text{identifiable quantity}$$

## Geometric Interpretation

### Parameter Space Decomposition

Any parameter vector $p$ can be uniquely decomposed as:

$$p = p_{\parallel} + p_{\perp}$$

where:
- $p_{\parallel} \in \mathcal{N}$: Unidentifiable component (parallel to nullspace)
- $p_{\perp} \in \mathcal{I}$: Identifiable component (perpendicular to nullspace)

### Constraint Manifolds

The nullspace defines **constraint manifolds** in parameter space:

- **Points on manifold**: Parameter combinations producing identical outputs
- **Manifold dimension**: $\dim(\mathcal{N})$
- **Normal directions**: Identifiable parameter combinations
- **Tangent directions**: Unidentifiable parameter combinations

## Computational Algorithm

### Step 1: Compute Nullspace Basis

Using SymPy's symbolic nullspace computation:

```python
def compute_nullspace_basis(J):
    """Compute symbolic nullspace basis."""
    # Use exact symbolic computation
    nullspace_vectors = J.nullspace()
    
    # Convert to matrix form
    if nullspace_vectors:
        N = sym.Matrix.hstack(*nullspace_vectors)
    else:
        N = sym.Matrix.zeros(J.cols, 0)  # Empty nullspace
    
    return N
```

### Step 2: Find Identifiable Combinations

```python
def find_identifiable_combinations(nullspace_basis):
    """Find basis for identifiable parameter combinations."""
    if nullspace_basis.cols == 0:
        # All parameters identifiable
        return sym.eye(nullspace_basis.rows)
    
    # Identifiable space is orthogonal complement
    identifiable_basis = nullspace_basis.T.nullspace()
    
    if identifiable_basis:
        I = sym.Matrix.hstack(*identifiable_basis)
    else:
        I = sym.Matrix.zeros(nullspace_basis.rows, 0)
    
    return I
```

### Step 3: Analyze Constraint Structure

```python
def analyze_parameter_constraints(nullspace_basis, parameter_names):
    """Analyze parameter constraints from nullspace structure."""
    constraints = []
    
    for i in range(nullspace_basis.cols):
        # Extract nullspace vector
        v = nullspace_basis[:, i]
        
        # Build constraint equation
        constraint_terms = []
        for j, coeff in enumerate(v):
            if coeff != 0:
                if coeff == 1:
                    constraint_terms.append(parameter_names[j])
                elif coeff == -1:
                    constraint_terms.append(f"-{parameter_names[j]}")
                else:
                    constraint_terms.append(f"{coeff}*{parameter_names[j]}")
        
        constraint = " + ".join(constraint_terms) + " = constant"
        constraints.append(constraint)
    
    return constraints
```

## Practical Examples

### Example 1: Simple Two-Parameter System

Consider a system with identifiability matrix:

$$\mathcal{J} = \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix}$$

**Nullspace analysis**:
- $\text{rank}(\mathcal{J}) = 1$
- $\dim(\mathcal{N}) = 2 - 1 = 1$
- Nullspace basis: $v = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

**Interpretation**:
- Constraint: $p_1 - p_2 = \text{constant}$
- Identifiable combination: $p_1 + p_2$

### Example 2: Three-Parameter Enzyme System

For the calibration model with parameters $[E_{tot}, k_f, k_r, k_{cat}]$:

**Possible nullspace structure**:
```
Nullspace basis vectors:
v1 = [0, 1, -1, 0]    # kf - kr = constant
v2 = [1, 0, 0, -1]    # Etot - kcat = constant
```

**Parameter constraints**:
- $k_f - k_r = c_1$ (binding equilibrium constraint)
- $E_{tot} - k_{cat} = c_2$ (enzyme-catalysis relationship)

**Identifiable combinations**:
- $k_f + k_r$ (total binding rate)
- $E_{tot} + k_{cat}$ (effective reaction capacity)

## Advanced Nullspace Properties

### Nullspace Intersection and Union

For multiple experimental conditions with identifiability matrices $\mathcal{J}_1, \mathcal{J}_2, \ldots$:

- **Intersection**: $\mathcal{N}_{\text{total}} = \bigcap_i \mathcal{N}(\mathcal{J}_i)$
- **Combined identifiability**: More experiments generally reduce nullspace dimension

### Rational Nullspace Computation

For systems with rational expressions, special handling ensures accuracy:

```python
def rational_nullspace(J):
    """Compute nullspace for matrices with rational entries."""
    # Clear denominators to work with polynomials
    J_cleared = clear_denominators(J)
    
    # Compute nullspace of cleared matrix
    nullspace_cleared = J_cleared.nullspace()
    
    # Verify results satisfy original matrix
    for v in nullspace_cleared:
        assert (J * v).simplify() == 0
    
    return nullspace_cleared
```

### Symbolic Simplification

Nullspace vectors often contain complex expressions requiring simplification:

```python
def simplify_nullspace_basis(nullspace_vectors, parameter_symbols):
    """Simplify nullspace basis vectors."""
    simplified = []
    
    for v in nullspace_vectors:
        # Factor out common terms
        v_factored = sym.factor(v)
        
        # Rationalize denominators
        v_rationalized = rationalize_vector(v_factored)
        
        # Choose canonical form (e.g., first nonzero entry positive)
        v_canonical = canonicalize_vector(v_rationalized)
        
        simplified.append(v_canonical)
    
    return simplified
```

## Nullspace Visualization

### 2D Constraint Lines

For two-parameter subspaces, nullspace constraints appear as lines:

```python
def plot_2d_constraints(nullspace_basis, param_names, param_ranges):
    """Plot constraint lines in 2D parameter space."""
    fig, ax = plt.subplots()
    
    for i, constraint_vector in enumerate(nullspace_basis.T):
        # Extract coefficients
        a, b = constraint_vector[0], constraint_vector[1]
        
        if b != 0:
            # Plot line ax + by = constant
            x_vals = np.linspace(*param_ranges[0], 100)
            y_vals = -(a/b) * x_vals  # Assuming constant = 0
            ax.plot(x_vals, y_vals, label=f'Constraint {i+1}')
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.legend()
    return fig
```

### 3D Constraint Surfaces

For three-parameter subspaces, constraints define surfaces:

```python
def plot_3d_constraint_surface(nullspace_vector, param_names, param_ranges):
    """Plot constraint surface in 3D parameter space."""
    # Create parameter grid
    p1_grid, p2_grid = np.meshgrid(
        np.linspace(*param_ranges[0], 50),
        np.linspace(*param_ranges[1], 50)
    )
    
    # Solve constraint for third parameter
    a, b, c = nullspace_vector[:3]
    if c != 0:
        p3_grid = -(a * p1_grid + b * p2_grid) / c
        
        # Plot surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(p1_grid, p2_grid, p3_grid, alpha=0.7)
        
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_zlabel(param_names[2])
    
    return fig
```

## Integration with Model Analysis

### Parameter Estimation Strategy

Nullspace analysis informs parameter estimation:

1. **Constrain unidentifiable directions**: Fix parameters along nullspace
2. **Estimate identifiable combinations**: Focus on orthogonal complement
3. **Use prior information**: Break ties in unidentifiable directions

```python
def design_parameter_estimation(nullspace_basis, identifiable_basis):
    """Design parameter estimation strategy."""
    strategy = {
        'fixed_parameters': [],
        'estimated_combinations': [],
        'prior_constraints': []
    }
    
    # Fix one parameter per nullspace direction
    for i, null_vector in enumerate(nullspace_basis.T):
        # Find parameter with largest coefficient
        max_idx = np.argmax(np.abs(null_vector))
        strategy['fixed_parameters'].append(max_idx)
    
    # Estimate identifiable combinations
    for ident_vector in identifiable_basis.T:
        combination = format_parameter_combination(ident_vector)
        strategy['estimated_combinations'].append(combination)
    
    return strategy
```

### Experimental Design

Nullspace structure guides experimental design:

```python
def suggest_additional_experiments(nullspace_basis, current_outputs):
    """Suggest experiments to improve identifiability."""
    suggestions = []
    
    # Analyze which parameters appear in nullspace
    problematic_params = find_problematic_parameters(nullspace_basis)
    
    for param in problematic_params:
        # Suggest measurements that would make this parameter identifiable
        suggested_outputs = find_outputs_sensitive_to_parameter(param)
        suggestions.append({
            'parameter': param,
            'suggested_measurements': suggested_outputs,
            'rationale': 'Would break current parameter correlation'
        })
    
    return suggestions
```

## Error Analysis and Robustness

### Numerical Stability

Nullspace computation can be sensitive to numerical errors:

```python
def robust_nullspace_computation(J, tolerance=1e-12):
    """Compute nullspace with numerical robustness checks."""
    # Try symbolic computation first
    try:
        null_vectors = J.nullspace()
        return null_vectors
    except:
        # Fall back to numerical computation
        J_numeric = np.array(J.evalf())
        _, _, V = np.linalg.svd(J_numeric)
        
        # Extract null vectors from SVD
        rank = np.sum(np.diag(s) > tolerance)
        null_vectors_numeric = V[rank:, :].T
        
        # Convert back to symbolic form
        return [sym.Matrix(v) for v in null_vectors_numeric.T]
```

### Validation and Cross-Checking

```python
def validate_nullspace_analysis(J, nullspace_basis, identifiable_basis):
    """Validate nullspace analysis results."""
    validation_results = {}
    
    # Check nullspace property: J * v = 0
    for i, v in enumerate(nullspace_basis.T):
        product = (J * v).simplify()
        is_zero = all(elem == 0 for elem in product)
        validation_results[f'nullspace_vector_{i}'] = is_zero
    
    # Check orthogonality: N^T * I = 0
    if nullspace_basis.cols > 0 and identifiable_basis.cols > 0:
        orthogonality = (nullspace_basis.T * identifiable_basis).simplify()
        is_orthogonal = all(elem == 0 for elem in orthogonality)
        validation_results['orthogonality'] = is_orthogonal
    
    # Check dimension consistency
    expected_dim = J.cols - J.rank()
    actual_dim = nullspace_basis.cols
    validation_results['dimension_consistency'] = (expected_dim == actual_dim)
    
    return validation_results
```

## Summary

Nullspace analysis transforms structural identifiability from a binary question ("Is parameter $p_i$ identifiable?") into a rich geometric understanding of parameter space structure. Key benefits include:

- **Parameter combinations**: Identifies what can be learned even when individuals can't
- **Constraint structure**: Reveals mathematical relationships between parameters
- **Experimental design**: Guides what additional measurements would help
- **Parameter estimation**: Informs which directions to constrain vs. estimate
- **Model validation**: Checks if estimated parameters satisfy structural constraints

This analysis is particularly valuable for complex nonlinear systems where traditional identifiability analysis provides limited actionable insights. By understanding the nullspace structure, researchers can make informed decisions about model parameterization, experimental design, and parameter estimation strategies.

---

## Further Reading

- **[Theory Overview](overview.md)**: Mathematical context and motivation
- **[STRIKE-GOLDD Method](strike-goldd.md)**: How nullspace analysis integrates with identifiability computation
- **[Visualization Guide](../results/visualizations.md)**: How to interpret nullspace visualizations

!!! tip "Implementation Notes"
    
    NullStrike's nullspace analysis includes many additional optimizations and special cases not shown here. The complete implementation handles rational functions, trigonometric systems, and various edge cases that arise in practice.