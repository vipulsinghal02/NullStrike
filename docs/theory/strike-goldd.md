# STRIKE-GOLDD Method

The STRIKE-GOLDD (STRuctural Identifiability Taken as Extended-Generalized Observability with Lie Derivatives and Decomposition) algorithm forms the computational backbone of NullStrike's identifiability analysis.

## Background and Motivation

Traditional identifiability analysis methods often struggle with:

- **Computational complexity** for nonlinear systems
- **Symbolic manipulation** of complex expressions  
- **Scalability** to realistic system sizes
- **Generality** across different model structures

STRIKE-GOLDD addresses these challenges through a systematic approach based on differential geometry and computer algebra.

## Theoretical Foundation

### The Identifiability Problem

Consider a nonlinear system:

$$\begin{align}
\dot{x} &= f(x, p, u) \quad &x(0) = x_0(p) \\
y &= h(x, p, u)
\end{align}$$

A parameter $p_i$ is **structurally locally identifiable** if:

$$\exists \, \epsilon > 0 : \forall \, \tilde{p} \text{ with } |\tilde{p}_i - p_i| > \epsilon, \quad y(t; p) \neq y(t; \tilde{p})$$

for some input $u(t)$ and almost all $t > 0$.

### Connection to Observability

The key insight is that **parameter identifiability** is closely related to **state observability**. Consider the extended system:

$$\begin{align}
\dot{x} &= f(x, p, u) \\
\dot{p} &= 0 \\
y &= h(x, p, u)
\end{align}$$

In this extended system, parameters become "states" with zero dynamics. Parameter identifiability becomes equivalent to observability of these extended states.

## The STRIKE-GOLDD Algorithm

### Step 1: Symbolic Model Setup

The algorithm begins with a symbolic representation of the dynamical system:

```python
# States
x = sym.Matrix([x1, x2, ..., xn])

# Parameters  
p = sym.Matrix([p1, p2, ..., pm])

# Dynamics
f = sym.Matrix([f1(x, p, u), f2(x, p, u), ..., fn(x, p, u)])

# Outputs
h = sym.Matrix([h1(x, p, u), h2(x, p, u), ..., hq(x, p, u)])
```

All symbolic computations use exact arithmetic to avoid numerical errors.

### Step 2: Lie Derivative Computation

The heart of the algorithm is the systematic computation of Lie derivatives.

#### Basic Lie Derivative

For a scalar function $g(x, p, u)$ and vector field $f$:

$$\mathcal{L}_f g = \frac{\partial g}{\partial x} f$$

#### Higher-Order Lie Derivatives

Computed recursively:

$$\mathcal{L}_f^{k+1} g = \mathcal{L}_f(\mathcal{L}_f^k g) = \frac{\partial}{\partial x}(\mathcal{L}_f^k g) \cdot f$$

#### Implementation Details

```python
def lie_derivative(g, f, x, order=1):
    """Compute the order-th Lie derivative of g with respect to f."""
    if order == 0:
        return g
    
    # First-order Lie derivative
    grad_g = sym.Matrix([g.diff(xi) for xi in x])
    lie_1 = grad_g.dot(f)
    
    if order == 1:
        return lie_1
    
    # Higher-order via recursion
    return lie_derivative(lie_1, f, x, order - 1)
```

The algorithm handles:

- **Input derivatives**: $\frac{d^k u}{dt^k}$ for known inputs
- **Unknown input effects**: Via additional Lie derivatives
- **Multiple outputs**: Each component computed separately

### Step 3: Observability Matrix Construction

The **observability matrix** $\mathcal{O}$ contains successive Lie derivatives:

$$\mathcal{O} = \begin{bmatrix}
h \\
\mathcal{L}_f h \\
\mathcal{L}_f^2 h \\
\vdots \\
\mathcal{L}_f^{r-1} h
\end{bmatrix}$$

where $r$ is chosen to ensure the matrix has constant rank.

#### Rank Determination

The algorithm computes the rank by:

1. **Symbolic rank computation**: Using SymPy's rank facilities
2. **Rational arithmetic**: To maintain exactness
3. **Iterative construction**: Adding rows until rank stabilizes

```python
def build_observability_matrix(h, f, x, max_order=None):
    """Build observability matrix with successive Lie derivatives."""
    if max_order is None:
        max_order = len(x)  # Conservative bound
    
    O = h  # Start with outputs
    prev_rank = O.rows
    
    for k in range(1, max_order + 1):
        # Add k-th Lie derivative
        lie_k = lie_derivative(h, f, x, k)
        O = O.col_join(lie_k)
        
        # Check if rank increased
        curr_rank = O.rank()
        if curr_rank == prev_rank:
            break  # Rank stabilized
        prev_rank = curr_rank
    
    return O
```

### Step 4: Identifiability Analysis

#### Parameter Jacobian

The **identifiability matrix** is the Jacobian with respect to parameters:

$$\mathcal{J} = \frac{\partial \mathcal{O}}{\partial p}$$

#### Individual Parameter Identifiability

A parameter $p_i$ is locally identifiable if:

$$\frac{\partial \mathcal{O}}{\partial p_i} \not\equiv 0$$

The algorithm checks this condition symbolically.

#### Rank Analysis

The number of identifiable parameters equals:

$$\text{rank}(\mathcal{J}) = \text{number of locally identifiable parameters}$$

### Step 5: Computational Optimizations

#### Expression Simplification

STRIKE-GOLDD includes sophisticated simplification:

```python
def simplify_expression(expr):
    """Simplify symbolic expressions with multiple strategies."""
    # Try different simplification approaches
    simplified = sym.simplify(expr)
    simplified = sym.trigsimp(simplified)
    simplified = sym.factor(simplified)
    simplified = sym.cancel(simplified)
    
    return simplified
```

#### Incremental Computation

For efficiency, the algorithm:

1. **Caches intermediate results**: Avoids recomputation
2. **Uses checkpointing**: Saves progress for large problems  
3. **Implements early termination**: Stops when rank stabilizes

#### Memory Management

Large symbolic expressions require careful memory handling:

- **Expression factoring**: Reduces memory footprint
- **Garbage collection**: Frees unused intermediate results
- **Streaming computation**: Processes large matrices in chunks

## NullStrike Extensions to STRIKE-GOLDD

### Enhanced Nullspace Analysis

While traditional STRIKE-GOLDD determines which parameters are unidentifiable, NullStrike extends this by:

1. **Computing nullspace basis**: Finding the structure of unidentifiable directions
2. **Identifying parameter combinations**: Determining what *can* be identified
3. **Geometric visualization**: Making the results interpretable

### Integration with Nullspace Methods

```python
def enhanced_strike_goldd(model, options):
    """STRIKE-GOLDD with nullspace analysis."""
    
    # Standard STRIKE-GOLDD analysis
    O = build_observability_matrix(model.h, model.f, model.x)
    J = O.jacobian(model.p)
    
    # NullStrike extensions
    rank_J = J.rank()
    nullspace_basis = J.nullspace()
    identifiable_combinations = find_identifiable_combinations(J)
    
    return {
        'observability_matrix': O,
        'identifiability_matrix': J,
        'rank': rank_J,
        'nullspace': nullspace_basis,
        'identifiable_combinations': identifiable_combinations
    }
```

## Algorithm Complexity and Scalability

### Computational Complexity

The complexity depends on several factors:

- **Number of states** ($n$): Affects Lie derivative computation
- **Number of parameters** ($m$): Affects Jacobian size  
- **System nonlinearity**: Affects expression complexity
- **Required Lie derivative order**: Usually $\mathcal{O}(n)$

**Typical complexity**: $\mathcal{O}(n^3 m)$ for basic operations, but can be higher for complex nonlinearities.

### Memory Requirements

Symbolic expressions can grow exponentially:

- **Lie derivative order**: Each order can increase expression size
- **Parameter count**: Jacobian matrix size scales as $\mathcal{O}(nm)$
- **Expression complexity**: Nonlinear terms can create very large expressions

### Scalability Strategies

NullStrike implements several approaches for large systems:

=== "Model Decomposition"

    Break large models into smaller, coupled subsystems:
    
    ```python
    def analyze_coupled_system(subsystems):
        """Analyze identifiability of coupled subsystems."""
        results = {}
        
        for subsystem in subsystems:
            # Analyze each subsystem independently
            results[subsystem.name] = enhanced_strike_goldd(subsystem)
        
        # Analyze coupling effects
        coupling_analysis = analyze_subsystem_coupling(subsystems)
        
        return combine_results(results, coupling_analysis)
    ```

=== "Progressive Analysis"

    Start with simplified models and add complexity gradually:
    
    ```python
    def progressive_analysis(model_variants):
        """Analyze increasingly complex model variants."""
        for variant in sorted(model_variants, key=lambda x: x.complexity):
            result = enhanced_strike_goldd(variant)
            
            if result['computational_time'] > threshold:
                return use_approximation_methods(variant)
        
        return result
    ```

=== "Parallel Computation"

    Distribute computations across multiple processes:
    
    ```python
    def parallel_strike_goldd(model, num_processes=4):
        """Parallel implementation of STRIKE-GOLDD."""
        # Split parameter space
        param_chunks = divide_parameters(model.p, num_processes)
        
        # Compute Jacobian blocks in parallel
        jacobian_blocks = parallel_map(
            compute_jacobian_block, 
            param_chunks
        )
        
        # Combine results
        return combine_jacobian_blocks(jacobian_blocks)
    ```

## Practical Implementation Details

### Handling Special Cases

#### Rational Function Systems

For systems with rational functions:

```python
def handle_rational_functions(expr):
    """Special handling for rational expressions."""
    numerator, denominator = sym.fraction(expr)
    
    # Check for singularities
    singularities = sym.solve(denominator, model.x + model.p)
    
    # Simplify while preserving structure
    simplified = sym.apart(expr, model.p)
    
    return simplified, singularities
```

#### Trigonometric Systems

For systems with trigonometric functions:

```python
def handle_trigonometric_systems(expr):
    """Simplify trigonometric expressions."""
    # Use trigonometric identities
    simplified = sym.trigsimp(expr)
    
    # Convert to exponential form if beneficial
    if is_complex_trig(simplified):
        simplified = sym.expand_trig(simplified)
    
    return simplified
```

### Integration with Computer Algebra Systems

NullStrike leverages SymPy's powerful symbolic capabilities:

- **Automatic differentiation**: For Jacobian computation
- **Matrix operations**: For rank and nullspace computation
- **Expression manipulation**: For simplification and factoring
- **Polynomial systems**: For solving algebraic constraints

### Numerical Validation

While the core algorithm is symbolic, NullStrike includes numerical validation:

```python
def validate_symbolic_results(symbolic_result, model, num_tests=100):
    """Validate symbolic identifiability results numerically."""
    
    for test_case in generate_test_cases(model, num_tests):
        # Substitute numerical values
        numerical_jacobian = symbolic_result.subs(test_case.parameter_values)
        
        # Compare ranks
        symbolic_rank = symbolic_result.rank()
        numerical_rank = numerical_jacobian.rank()
        
        if symbolic_rank != numerical_rank:
            warnings.warn(f"Rank mismatch in test case {test_case.id}")
    
    return validation_report
```

## Error Handling and Robustness

### Common Issues and Solutions

=== "Expression Explosion"

    **Problem**: Symbolic expressions become unmanageably large
    
    **Solutions**:
    - Intermediate simplification
    - Expression factoring  
    - Numerical approximation for very large expressions

=== "Rank Computation Failures"

    **Problem**: SymPy cannot determine matrix rank symbolically
    
    **Solutions**:
    - Alternative rank computation methods
    - Numerical rank estimation as fallback
    - Manual inspection of specific cases

=== "Memory Exhaustion"

    **Problem**: System runs out of memory during computation
    
    **Solutions**:
    - Progressive computation with checkpointing
    - Model simplification
    - Distributed computation

### Debugging and Diagnostics

NullStrike provides extensive debugging capabilities:

```python
def debug_strike_goldd(model, options):
    """Debug version with detailed logging."""
    
    logger.info(f"Starting STRIKE-GOLDD analysis for {model.name}")
    logger.info(f"States: {len(model.x)}, Parameters: {len(model.p)}")
    
    # Track computation progress
    with ProgressTracker() as tracker:
        # Lie derivative computation
        tracker.start_phase("lie_derivatives")
        O = build_observability_matrix(model.h, model.f, model.x)
        tracker.end_phase()
        
        # Jacobian computation
        tracker.start_phase("jacobian")
        J = O.jacobian(model.p)
        tracker.end_phase()
        
        # Rank analysis
        tracker.start_phase("rank_analysis")
        rank = J.rank()
        tracker.end_phase()
    
    return create_debug_report(O, J, rank, tracker.timings)
```

## Integration with NullStrike Workflow

### Checkpointing Integration

STRIKE-GOLDD results are automatically saved:

```python
def strike_goldd_with_checkpointing(model, options):
    """STRIKE-GOLDD with automatic result caching."""
    
    # Check for existing results
    checkpoint = load_checkpoint(model, options)
    if checkpoint and checkpoint.is_valid():
        return checkpoint.results
    
    # Compute results
    results = enhanced_strike_goldd(model, options)
    
    # Save checkpoint
    save_checkpoint(model, options, results)
    
    return results
```

### Visualization Pipeline

STRIKE-GOLDD results feed into NullStrike's visualization system:

```python
def create_visualizations_from_strike_goldd(results, options):
    """Generate visualizations from STRIKE-GOLDD results."""
    
    # Extract key information
    nullspace_basis = results['nullspace']
    identifiable_combos = results['identifiable_combinations']
    
    # Generate plots
    manifold_plots = create_manifold_plots(nullspace_basis, options)
    projection_plots = create_projection_plots(identifiable_combos, options)
    graph_plot = create_parameter_graph(results, options)
    
    return {
        'manifolds': manifold_plots,
        'projections': projection_plots,
        'graph': graph_plot
    }
```

## Summary

The STRIKE-GOLDD algorithm provides NullStrike with a robust, scalable foundation for structural identifiability analysis. Key strengths include:

- **Mathematical rigor**: Based on solid differential geometric principles
- **Computational efficiency**: Optimized symbolic computation with caching
- **Generality**: Handles wide range of nonlinear dynamical systems
- **Extensibility**: Integrates seamlessly with nullspace analysis

The algorithm's implementation in NullStrike includes numerous practical enhancements for real-world applicability, from memory management to numerical validation, making sophisticated identifiability analysis accessible to researchers and practitioners.

---

## Further Reading

- **[Nullspace Analysis](nullspace.md)**: How NullStrike extends STRIKE-GOLDD
- **[Lie Derivatives](lie-derivatives.md)**: Mathematical details of Lie derivative computation
- **[Theory Overview](overview.md)**: Broader mathematical context

!!! note "Implementation Details"
    
    The complete STRIKE-GOLDD implementation in NullStrike includes additional optimizations and error handling not shown here. See the [API Reference](../api/core.md) for full implementation details.