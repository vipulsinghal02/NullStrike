# NullStrike Documentation

Welcome to **NullStrike** - a powerful tool for structural identifiability analysis of nonlinear dynamical systems with advanced nullspace analysis capabilities.

![NullStrike Logo](assets/nullstrike-banner.png){ .center }

## What is NullStrike?

NullStrike extends the capabilities of StrikePy (a Python implementation of STRIKE-GOLDD) by adding sophisticated nullspace analysis to determine not just which parameters are unidentifiable, but which **parameter combinations are identifiable** even when individual parameters are not.

### The Core Problem

In many dynamical systems, individual parameters may be unidentifiable, but specific combinations of these parameters can still be uniquely determined from experimental data. NullStrike bridges this gap by:

1. **Computing the observability-identifiability matrix** using Lie derivatives
2. **Analyzing the nullspace** to find unidentifiable directions  
3. **Identifying the row space** containing identifiable parameter combinations
4. **Visualizing these relationships** through 3D manifolds and constraint graphs

## Key Features

=== "Mathematical Foundation"
    
    - **STRIKE-GOLDD Algorithm**: Structural identifiability analysis using Lie derivatives
    - **Nullspace Analysis**: $\mathcal{N} = \text{Matrix}(\text{nullspace\_vectors})$
    - **Identifiable Directions**: $\text{identifiable\_directions} = \mathcal{N}.\text{nullspace}()$
    - **Symbolic Computation**: Full symbolic analysis using SymPy

=== "Visualization & Analysis"
    
    - **3D Manifold Plots**: Visualize parameter constraint surfaces
    - **2D Projections**: Understand pairwise parameter relationships  
    - **Graph Analysis**: Network representation of parameter dependencies
    - **Comprehensive Reports**: Mathematical interpretations and results

=== "Performance & Usability"
    
    - **Checkpointing System**: Efficient reanalysis with intelligent caching
    - **CLI Interface**: Simple command-line usage
    - **Python API**: Programmatic access for advanced users
    - **Batch Processing**: Analyze multiple models efficiently

## Quick Start

Get started with NullStrike in just a few commands:

=== "Command Line"

    ```bash
    # Install NullStrike
    pip install -e .
    
    # Run analysis on built-in examples
    nullstrike C2M                    # Two-compartment model
    nullstrike calibration_single     # Calibration example
    nullstrike Bolie                  # Bolie model
    ```

=== "Python API"

    ```python
    from nullstrike.cli.complete_analysis import main
    
    # Run complete analysis
    results = main('C2M', 'options_C2M')
    
    # Results include:
    # - Identifiability analysis
    # - Parameter combinations  
    # - Visualization files
    # - Detailed reports
    ```

=== "Custom Models"

    ```python
    # Define your model in custom_models/my_model.py
    import sympy as sym
    
    # States
    x1, x2 = sym.symbols('x1 x2')
    x = [[x1], [x2]]
    
    # Parameters  
    p1, p2, p3 = sym.symbols('p1 p2 p3')
    p = [[p1], [p2], [p3]]
    
    # Outputs
    h = [x1]
    
    # Dynamics
    f = [[p1*x1 + p2*x2], [-p3*x1]]
    ```

## Example Results

NullStrike generates comprehensive analysis including:

!!! example "Two-Compartment Model Results"
    
    For a pharmacokinetic two-compartment model:
    
    - **Unidentifiable parameters**: `k12`, `k21`, `V1`, `V2` individually
    - **Identifiable combinations**: `k12*V1`, `k21*V2`, `(k12+k21+k10)*V1`  
    - **Visualization**: 3D manifolds showing constraint surfaces
    - **Graph analysis**: Parameter dependency networks

## Mathematical Background

The core mathematical relationship in NullStrike is:

$$\begin{align}
\mathcal{O} &= \begin{bmatrix} \mathcal{L}_f^0 h \\ \mathcal{L}_f^1 h \\ \vdots \\ \mathcal{L}_f^n h \end{bmatrix} \\[0.5em]
\mathcal{N} &= \text{nullspace}(\mathcal{O}) \\[0.5em]
\mathcal{I} &= \text{nullspace}(\mathcal{N})
\end{align}$$

Where:
- $\mathcal{O}$ is the observability-identifiability matrix
- $\mathcal{L}_f^k h$ are the $k$-th Lie derivatives  
- $\mathcal{N}$ contains unidentifiable directions
- $\mathcal{I}$ contains identifiable parameter combinations

## Navigation Guide

- **[Getting Started](installation.md)**: Installation and setup
- **[Mathematical Foundations](theory/overview.md)**: Theory and algorithms  
- **[User Guide](guide/models.md)**: Practical usage instructions
- **[Examples](examples/simple.md)**: Step-by-step tutorials
- **[API Reference](api/core.md)**: Detailed code documentation

---

!!! tip "Need Help?"
    
    - Check out the [Quick Start guide](quickstart.md) for immediate setup
    - Browse [Examples](examples/simple.md) for common use cases  
    - See [API Reference](api/core.md) for programmatic usage
    - Visit the [GitHub repository](https://github.com/vipulsinghal02/NullStrike) for issues and contributions
