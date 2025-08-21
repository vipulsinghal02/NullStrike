"""
NullStrike: Enhanced Structural Identifiability Analysis

NullStrike extends StrikePy with advanced nullspace analysis to identify
parameter combinations that are structurally identifiable even when 
individual parameters are not.

This package integrates the complete StrikePy functionality by David Rey Rostro
(based on STRIKE-GOLDD by Alejandro Fernandez Villaverde) with new capabilities
for nullspace analysis and advanced visualization.

Key Features:
- Complete StrikePy functionality (observability-identifiability matrix computation)
- Advanced nullspace analysis for parameter combination identification
- 3D manifold visualization of unidentifiable parameter relationships
- Comprehensive reporting with mathematical interpretations
- Efficient checkpointing system for large analyses

Usage Examples:
    
    # Use original StrikePy functionality
    from nullstrike.core import strike_goldd
    strike_goldd('my_options_file')
    
    # Use enhanced integrated analysis  
    import nullstrike
    results = nullstrike.run_integrated_analysis('my_model', 'my_options')
    
    # Use specific analysis components
    from nullstrike.analysis import analyze_identifiable_combinations
    
License: GPL-3.0 (to maintain compatibility with integrated StrikePy code)

Credits:
- Original StrikePy: David Rey Rostro (davidreyrostro@gmail.com)
- Original STRIKE-GOLDD: Alejandro Fernandez Villaverde (afvillaverde@uvigo.gal)
- NullStrike extensions: [Your Name]

See ATTRIBUTION.md for complete attribution details.
"""

__version__ = "0.1.0"
__author__ = "Vipul Singhal"
__email__ = "vs@alumni.caltech.edu"

# Core StrikePy functionality (for backward compatibility)
from .core import strike_goldd

# Enhanced analysis capabilities
from .analysis import (
    run_integrated_analysis,
    analyze_identifiable_combinations,
    build_identifiability_graph,
    visualize_identifiability_graph,
    visualize_nullspace_manifolds
)

# CLI interface
from .cli import main
from .cli.complete_analysis import validate_environment, create_example_model

__all__ = [
    # Core StrikePy functionality
    'strike_goldd',
    
    # Utilities
    'validate_environment',
    'create_example_model',
    
    # Enhanced NullStrike functionality
    'run_integrated_analysis',
    'analyze_identifiable_combinations', 
    'build_identifiability_graph',
    'visualize_identifiability_graph',
    'visualize_nullspace_manifolds',
    
    # CLI
    'main'
]