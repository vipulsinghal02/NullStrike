# NullStrike Documentation

## What is NullStrike?

NullStrike (NULLspace analysis + STRIKEpy) is a tool for analyzing the nonlinear observability and structural identifiability properties of dynamical systems in a differential geometric framework (CITE HERMAN KRENER and others). It extends the capabilities of `StrikePy` (a Python implementation of the MATLAB-based `STRIKE-GOLDD` structural identifiability analysis toolbox (CITE villaverde)) by adding the capability to determine not only which parameters are identifiable (and states are observable), but which *combinations of parameters and states are identifiable and observable*, even when individual parameters are not. It does so by computing the **nullspace** (kernel) of the observability-identifiability codistribution, as described in (cite calibration paper.). 

## Geometric Intuition

Given an initialized, parameterized nonlinear ODE model, it might not always be possible to uniquely identify the values of all its parameters and state variables from associated input-output behavior data. The inability of identify parameters (respectively, states) is called (structural) non-identifiability (respectively, unobservability). In general, multiple points in the joint parameter-state variable space will lead to the same input-output behavior, simply by virtue of the structure of the equations of the ODE model. These points form an equivalence class with respect to the (fixed) model structure and input output behaviour, and will, in general, lie along a diffentiable manifold in this joint space. The tools of differential geometry---in particular Frobenius' Theorem and Lie derivatives of output functions along system trajectories---may be used to analyze these manifolds and the nature of the non-identifiability and unobservability. 

This manifold of equivalent points encodes the *co-variation* of parameters and states as their value changes, while keeping the input-output behaviour constant. Changing the input-output behavior corresponds to moving to a different manifold. Indeed, the parameter-state space may be *foliated* via Frobenius theorem, with each leaf corresponding to different input-output behavior (corresponding a different equivalence class of parameters-states). 

Given one of these manifolds (*leaves* in the foliation), the tangent space to the manifold at a point gives the subspace of parameter-state variation directions that leave the input-output behaviour unchanged. These directions can be used to determine how the parameters and states covary to maintain the same behavior. 

See the [Theory](theory.md) section for mode details. 


<!--

### The Core Problem

In many dynamical systems, individual parameters may be unidentifiable, but specific combinations of these parameters can still be uniquely determined from experimental data. NullStrike bridges this gap by:

1. **Computing the observability-identifiability matrix** using Lie derivatives
2. **Analyzing the nullspace** to find unidentifiable directions  
3. **Identifying the row space** containing identifiable parameter combinations
4. **Visualizing these relationships** through 3D and 2D manifolds and constraint graphs

## Key Features

=== "Mathematical Foundation"
    
    - **STRIKE-GOLDD Algorithm**: Structural identifiability analysis using Lie derivatives
    - **Nullspace Analysis**: $\mathcal{N} = \text{Matrix}(\text{nullspace_vectors})$
    - **Identifiable Directions**: $\text{identifiable_directions} = \mathcal{N}.\text{nullspace}()$
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
-->

<!--
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

#TODO explain the math part a little more thoroughly (define terms, explain thing. Either here or in the theory section.)

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

- $\mathcal{I}$ contains identifiable directions
-->
## Navigation Guide

- **[Getting Started](installation.md)**: Installation and setup
- **[Mathematical Foundations](theory.md)**: Theory and algorithms  
- **[Examples](examples.md)**: Step-by-step tutorials

---
<!-- 
!!! tip "Need Help?"
    
    - Check out the [Quick Start guide](quickstart.md) for immediate setup
    - Browse [Examples](examples.md) for common use cases  
    - See [API Reference](reference.md) for programmatic usage
    - Visit the [GitHub repository](https://github.com/vipulsinghal02/NullStrike) for issues and contributions -->
