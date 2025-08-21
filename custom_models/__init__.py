"""
Custom Models Directory

This directory contains model definitions for use with NullStrike analysis.
Models were moved from src/nullstrike/models/ to keep them separate from 
the package code and make them easier to customize.

Each model file should define:
- State variables (x): The system states to be analyzed
- Parameters (p): Unknown parameters to be identified  
- Outputs (h): Measured variables
- Dynamics (f): Differential equations describing the system
- Inputs (u): Known control inputs (optional)
- Unknown inputs (w): Unmeasured disturbances (optional)
- variables_locales: Dictionary containing all local variables
"""

# This module is primarily accessed via direct imports of specific model files
# rather than through this __init__.py file