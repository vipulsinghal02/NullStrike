# NullStrike Documentation

Welcome to NullStrike - a powerful tool for structural identifiability analysis of nonlinear dynamical systems with advanced nullspace analysis capabilities.

## What is NullStrike?

NullStrike builds upon the original StrikePy package (a Python implementation of STRIKE-GOLDD) by adding nullspace analysis to determine not just which parameters are unidentifiable (and states/inputs are unobservable), but which **parameter (and state/input) combinations are unidentifiable/unobservable**.

## Key Features

- **Structural Identifiability Analysis**: Determine which parameters can be uniquely estimated
- **Nullspace Analysis**: Find identifiable parameter combinations when individual parameters aren't identifiable
- **Advanced Visualizations**: 3D manifold plots and 2D constraint visualizations
- **Comprehensive Reports**: Detailed analysis with mathematical interpretations
- **Checkpointing**: Efficient reanalysis with caching system

## Quick Example

```python
from complete_analysis import main
results = main('your_model', 'your_options')
