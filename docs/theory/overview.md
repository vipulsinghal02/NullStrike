# Mathematical Foundations: Theory Overview

This page provides the mathematical foundations underlying NullStrike's approach to structural identifiability analysis. Understanding these concepts will help you interpret results and apply the tool effectively to your own models.

## Core Problem Statement

Given a nonlinear dynamical system:

$$\begin{align}
\dot{x}(t) &= f(x(t), p, u(t)) \quad &x(0) = x_0(p) \\
y(t) &= h(x(t), p, u(t))
\end{align}$$

where:

- $x(t) \in \mathbb{R}^n$ are the **states** (internal variables)
- $p \in \mathbb{R}^m$ are the **parameters** (unknown constants)
- $u(t) \in \mathbb{R}^r$ are the **inputs** (known functions of time)
- $y(t) \in \mathbb{R}^q$ are the **outputs** (measured quantities)

**The fundamental question**: Which parameters (or parameter combinations) can be uniquely determined from input-output data $\{u(t), y(t)\}_{t \geq 0}$?

## Traditional Identifiability vs. NullStrike Approach

### Traditional Structural Identifiability

Classical methods determine which individual parameters are identifiable:

- **Globally identifiable**: Parameter has a unique value
- **Locally identifiable**: Parameter has finitely many possible values  
- **Unidentifiable**: Parameter cannot be determined from data

**Limitation**: When parameters are unidentifiable, traditional methods provide little guidance on what *can* be determined.

### NullStrike's Nullspace Approach  

NullStrike extends identifiability analysis by finding:

1. **Which individual parameters are identifiable** (traditional analysis)
2. **Which parameter combinations are identifiable** even when individuals aren't
3. **The geometric structure** of parameter constraints
4. **Visualization** of identifiable vs. unidentifiable directions

This provides actionable insights for model calibration and experimental design.

## The STRIKE-GOLDD Algorithm

NullStrike builds upon the STRIKE-GOLDD (STRuctural Identifiability Taken as Extended-Generalized Observability with Lie Derivatives and Decomposition) algorithm.

### Lie Derivatives and Observability

The core insight is that identifiability is closely related to **observability**. For a system to be observable, we must be able to distinguish different states through the output measurements.

#### Lie Derivatives

Given a vector field $f$ and output function $h$, the **Lie derivative** measures how $h$ changes along trajectories of $f$:

$$\mathcal{L}_f h = \frac{\partial h}{\partial x} f$$

**Higher-order Lie derivatives** capture how the output and its time derivatives evolve:

$$\begin{align}
\mathcal{L}_f^0 h &= h \\
\mathcal{L}_f^1 h &= \frac{\partial h}{\partial x} f \\
\mathcal{L}_f^2 h &= \frac{\partial}{\partial x}\left(\frac{\partial h}{\partial x} f\right) f \\
&\vdots
\end{align}$$

Physically, $\mathcal{L}_f^k h$ represents the $k$-th time derivative of the output:

$$\frac{d^k y}{dt^k} = \mathcal{L}_f^k h(x(t), p, u(t))$$

### The Observability-Identifiability Matrix

The **observability matrix** $\mathcal{O}$ is constructed by stacking Lie derivatives:

$$\mathcal{O} = \begin{bmatrix}
\mathcal{L}_f^0 h \\
\mathcal{L}_f^1 h \\
\mathcal{L}_f^2 h \\
\vdots \\
\mathcal{L}_f^{n-1} h
\end{bmatrix}$$

For identifiability analysis, we examine how $\mathcal{O}$ depends on the parameters $p$.

### Parameter Identifiability via Jacobian Analysis

A parameter $p_i$ is **locally identifiable** if small changes in $p_i$ produce detectable changes in the output trajectory. Mathematically, this requires:

$$\frac{\partial \mathcal{O}}{\partial p_i} \neq 0$$

The **identifiability matrix** is:

$$\mathcal{J} = \frac{\partial \mathcal{O}}{\partial p} \in \mathbb{R}^{(nq) \times m}$$

**Key insight**: The rank of $\mathcal{J}$ determines how many parameters are identifiable.

- If $\text{rank}(\mathcal{J}) = m$: All parameters are locally identifiable
- If $\text{rank}(\mathcal{J}) < m$: Some parameters are unidentifiable

## Nullspace Analysis: The Core Innovation

Traditional STRIKE-GOLDD stops at determining which parameters are unidentifiable. NullStrike's innovation is to analyze the **nullspace structure** to find identifiable parameter combinations.

### The Nullspace

The **nullspace** of the identifiability matrix $\mathcal{J}$ contains the unidentifiable directions:

$$\mathcal{N} = \text{null}(\mathcal{J}) = \{v \in \mathbb{R}^m : \mathcal{J} v = 0\}$$

**Interpretation**: If $v \in \mathcal{N}$, then parameter perturbations in direction $v$ don't change the observable output.

### Nullspace Basis and Parameter Combinations

Let $\{v_1, v_2, \ldots, v_k\}$ be a basis for $\mathcal{N}$, where $k = m - \text{rank}(\mathcal{J})$ is the nullspace dimension.

Each basis vector $v_i = [v_{i1}, v_{i2}, \ldots, v_{im}]^T$ defines an **unidentifiable parameter combination**:

$$v_{i1} p_1 + v_{i2} p_2 + \cdots + v_{im} p_m = \text{constant}$$

**Key insight**: Directions orthogonal to the nullspace are identifiable!

### Identifiable Parameter Combinations

The **identifiable subspace** is the orthogonal complement of the nullspace:

$$\mathcal{I} = \mathcal{N}^{\perp} = \{w \in \mathbb{R}^m : w^T v = 0 \text{ for all } v \in \mathcal{N}\}$$

Vectors in $\mathcal{I}$ define **identifiable parameter combinations**.

### Complete Identifiability Decomposition

For any parameter vector $p$, we can decompose:

$$p = p_{\text{identifiable}} + p_{\text{unidentifiable}}$$

where:
- $p_{\text{identifiable}} \in \mathcal{I}$: Can be determined from data
- $p_{\text{unidentifiable}} \in \mathcal{N}$: Cannot be determined from data

## Geometric Interpretation

### Parameter Space Manifolds  

The nullspace analysis reveals the **geometric structure** of identifiable parameters:

- **Identifiable directions**: Form the identifiable subspace $\mathcal{I}$
- **Unidentifiable directions**: Form the nullspace $\mathcal{N}$  
- **Constraint manifolds**: Surfaces in parameter space where outputs are identical

### Visualization and Intuition

NullStrike generates visualizations to make this geometry concrete:

=== "3D Manifolds"

    Show constraint surfaces in parameter space:
    
    - **Surface points**: Parameter combinations producing identical outputs
    - **Normal directions**: Identifiable parameter combinations
    - **Tangent directions**: Unidentifiable parameter combinations

=== "2D Projections"

    Show pairwise parameter relationships:
    
    - **Constraint lines**: Linear relationships between parameters
    - **Identifiable axes**: Directions where parameters can be determined
    - **Correlation patterns**: How parameters covary

=== "Parameter Graphs"

    Network representation of identifiability:
    
    - **Nodes**: Individual parameters
    - **Edges**: Identifiability relationships
    - **Clusters**: Groups of related parameters
    - **Colors**: Identifiability status

## Mathematical Example: Two-Parameter System

Consider a simple system with parameters $p_1, p_2$ and identifiability matrix:

$$\mathcal{J} = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$$

### Nullspace Analysis

$$\text{rank}(\mathcal{J}) = 1 < 2 \Rightarrow \text{some parameters unidentifiable}$$

The nullspace is:
$$\mathcal{N} = \text{span}\left\{\begin{bmatrix} 1 \\ -1 \end{bmatrix}\right\}$$

**Interpretation**: The combination $p_1 - p_2$ is unidentifiable.

### Identifiable Combinations

The identifiable subspace is:
$$\mathcal{I} = \text{span}\left\{\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right\}$$

**Interpretation**: The combination $p_1 + p_2$ is identifiable.

### Complete Picture

- **Unidentifiable**: $p_1 - p_2 = \text{constant}$ (constraint line)
- **Identifiable**: $p_1 + p_2$ can be determined uniquely
- **Individual parameters**: Neither $p_1$ nor $p_2$ alone is identifiable

## Computational Aspects

### Symbolic vs. Numerical Computation

NullStrike performs **symbolic computation** using SymPy:

**Advantages**:
- Exact results (no numerical errors)
- Works for general parameter values  
- Reveals algebraic structure

**Considerations**:
- Can be computationally intensive for large systems
- May require simplification of complex expressions

### Rank Computation and Numerical Stability

Determining matrix rank symbolically can be challenging. NullStrike uses:

1. **Symbolic rank computation**: When expressions are manageable
2. **Rational arithmetic**: To avoid floating-point errors
3. **Expression simplification**: To handle complex symbolic results

### Scalability Considerations

For systems with many parameters:

- **Computational complexity**: Grows with model size and parameter count
- **Memory requirements**: Large symbolic expressions require significant RAM
- **Checkpointing**: Saves intermediate results for efficient reanalysis

## Extensions and Advanced Topics

### Unknown Inputs and Disturbances

NullStrike can handle systems with unknown inputs $w(t)$:

$$\begin{align}
\dot{x}(t) &= f(x(t), p, u(t), w(t)) \\
y(t) &= h(x(t), p, u(t), w(t))
\end{align}$$

The analysis accounts for the effect of unmeasured disturbances on identifiability.

### Initial Conditions as Parameters

State observability analysis treats initial conditions as unknown parameters:

$$x(0) = x_0 = [x_{0,1}, x_{0,2}, \ldots, x_{0,n}]^T$$

This determines which initial conditions can be estimated from output data.

### Multiple Experiments and Input Design

The framework extends to multiple experiments with different inputs:

$$\mathcal{J}_{\text{total}} = \begin{bmatrix} \mathcal{J}_1 \\ \mathcal{J}_2 \\ \vdots \end{bmatrix}$$

This guides **optimal experiment design** for improved identifiability.

## Relationship to Other Methods

### Comparison with Traditional Approaches

| Method | Identifies | Limitations | NullStrike Advantage |
|--------|------------|-------------|---------------------|
| Transfer function analysis | Individual parameters | Linear systems only | Handles nonlinear systems |
| Profile likelihood | Individual parameters | Requires data | Works symbolically |
| Practical identifiability | Statistical properties | Needs specific data | General structural analysis |
| **NullStrike** | **Parameter combinations** | **Computational complexity** | **Complete nullspace analysis** |

### Connection to Differential Algebra

The theoretical foundation connects to **differential algebra** and the theory of **differential fields**. The identifiability analysis is equivalent to studying the transcendence degree of field extensions.

### Relationship to Observability

**State observability** and **parameter identifiability** are closely related:

- Both use Lie derivative techniques
- Both examine matrix rank conditions  
- Observability focuses on initial conditions; identifiability on parameters

## Practical Implications

### Model Development

Understanding the theory helps in:

1. **Model structure selection**: Choose parametrizations with good identifiability properties
2. **Model reduction**: Eliminate unidentifiable parameters or fix them to known values
3. **Model validation**: Check that estimated parameters satisfy identifiability constraints

### Experimental Design

The nullspace structure guides:

1. **Sensor placement**: Add measurements to break parameter correlations
2. **Input design**: Choose inputs that excite identifiable modes
3. **Data collection**: Focus on time periods with high identifiability

### Parameter Estimation

Results inform estimation strategies:

1. **Constraint handling**: Use identifiable combinations as constraints
2. **Regularization**: Penalize movement in unidentifiable directions
3. **Uncertainty quantification**: Account for fundamental limits on parameter precision

## Summary

NullStrike's mathematical foundation combines:

- **STRIKE-GOLDD algorithm**: For computing observability matrices via Lie derivatives
- **Nullspace analysis**: For finding identifiable parameter combinations  
- **Geometric visualization**: For understanding parameter space structure
- **Symbolic computation**: For exact, general results

This approach provides a complete picture of what can and cannot be learned about model parameters from experimental data, going beyond traditional identifiability analysis to find actionable parameter combinations and constraints.

The theory is implemented efficiently with caching, checkpointing, and visualization tools that make these sophisticated mathematical concepts accessible to practitioners working with real-world dynamical systems.

---

## Further Reading

- **[STRIKE-GOLDD Method](strike-goldd.md)**: Detailed explanation of Lie derivative computation
- **[Nullspace Analysis](nullspace.md)**: Deep dive into nullspace computation and interpretation
- **[Observability & Identifiability](observability.md)**: Connections between state observability and parameter identifiability
- **[Lie Derivatives](lie-derivatives.md)**: Mathematical details of Lie derivative computation

!!! tip "Mathematical Prerequisites"
    
    - **Linear algebra**: Vector spaces, nullspaces, orthogonal complements
    - **Differential equations**: Nonlinear dynamical systems
    - **Symbolic computation**: Basic familiarity with computer algebra
    - **Differential geometry**: Lie derivatives and vector fields (helpful but not required)