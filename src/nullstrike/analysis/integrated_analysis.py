"""
Integrated analysis combining StrikePy's observability-identifiability matrix
computation with nullspace analysis for parameter combinations.

This module extends the original StrikePy functionality with advanced nullspace
analysis to identify structurally identifiable parameter combinations.

License: GPL-3.0 (to maintain compatibility with StrikePy)
"""

import os
import sys
import sympy as sym
import numpy as np
from sympy import Matrix

# Add current working directory to Python path to find custom_options and custom_models
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Import utilities
from ..utils import get_results_dir

# Import StrikePy core functionality
from ..core import strike_goldd

from .enhanced_subspace import analyze_identifiable_combinations
from ..visualization import build_identifiability_graph
from ..visualization import visualize_identifiability_graph, visualize_nullspace_manifolds
from .checkpointing import compute_model_hash, save_checkpoint, load_checkpoint


# """
# Integrated analysis combining StrikePy's observability-identifiability matrix
# computation with nullspace analysis for parameter combinations.
# """

# import os
# import sys
# import sympy as sym
# import numpy as np
# from sympy import Matrix
# from pathlib import Path

# # Import StrikePy components
# from strike_goldd import strike_goldd
# from nullstrike.analysis.enhanced_subspace import (
#     analyze_identifiable_combinations,
#     build_identifiability_graph,
#     visualize_identifiability_graph,
#     visualize_nullspace_manifolds
# )

# from nullstrike.analysis.checkpointing import compute_model_hash, save_checkpoint, load_checkpoint

def run_integrated_analysis(model_name=None, options_file=None, analysis_scope='full'):
    """
    Run complete identifiability analysis combining StrikePy with nullspace analysis.
    
    Parameters:
    -----------
    model_name : str, optional
        Name of the model to analyze (e.g., 'C2M')
    options_file : str, optional  
        Custom options file (e.g., 'options_C2M')
    analysis_scope : str
        'full' - analyze all variables (default)
        'parameters' - analyze only parameters
    
    Returns:
    --------
    dict : Complete analysis results
    """
    
    print("="*70)
    print("INTEGRATED IDENTIFIABILITY ANALYSIS")
    print("="*70)
    
    # Step 1: Run StrikePy analysis
    print("\nStep 1: Running StrikePy observability-identifiability analysis...")
    
    if options_file:
        # Import the specific options file to get model info
        options = __import__(f'custom_options.{options_file}', fromlist=[''])
        
        # Check if this is the default options file (empty modelname)
        if hasattr(options, 'modelname') and options.modelname == '':
            # Using default options with specific model
            strike_goldd(options_file, model_name_override=model_name)
        else:
            # Using model-specific options
            strike_goldd(options_file)
    else:
        # Run with default options
        strike_goldd()
        from ..configs import default_options as options

    # Get MANIFOLD_PLOTTING config from the appropriate options module
    manifold_config = getattr(options, 'MANIFOLD_PLOTTING', {})
    
    # Import the model
    import importlib

    try:
        # Try to load from our package first
        model = importlib.import_module(f'nullstrike.models.{options.modelname}')
    except ImportError:
        # Fall back to custom_models directory 
        model = importlib.import_module(f'custom_models.{options.modelname}')    
    
    # model = importlib.import_module(f'models.{options.modelname}')
    
    # Step 1.5: Check for existing checkpoint
    print("\nStep 1.5: Checking for existing checkpoint...")
    model_hash = compute_model_hash(model, options)
    checkpoint = load_checkpoint(options.modelname, options_file, model_hash)
    
    if checkpoint:
        print("Using cached results from checkpoint")
        onx_matrix = checkpoint['oic_matrix']
        nullspace_results = checkpoint['nullspace_results']
        
        # Generate timestamp for consistent file organization
        from datetime import datetime
        shared_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Skip to visualization and reporting
        results = {
            'model_name': options.modelname,
            'matrix_rank': onx_matrix.rank(),
            'matrix_shape': onx_matrix.shape,
            'states': extract_symbols(model.x),
            'parameters': extract_symbols(model.p),
            'nullspace_analysis': nullspace_results
        }
        
        save_detailed_results(results, options.modelname, shared_timestamp)
        
        if not nullspace_results.get('fully_identifiable', False):
            state_symbols = extract_symbols(model.x)
            param_symbols = extract_symbols(model.p)
            input_symbols = extract_symbols(model.w)
            manifold_config = getattr(options, 'MANIFOLD_PLOTTING', {})
            
            create_visualizations(
                nullspace_results, param_symbols, options.modelname, 
                analysis_scope, state_symbols, input_symbols,
                manifold_config, shared_timestamp
            )
        
        return results
    
    print("No valid checkpoint found - running fresh analysis")    
    
    # Step 2: Load the observability-identifiability matrix from saved file
    if not checkpoint:
        print("\nStep 2: Loading observability-identifiability matrix...")
    
    # Find the most recent OIC matrix file
    results_dir = get_results_dir()
    oic_files = list(results_dir.glob(f'obs_ident_matrix_{options.modelname}_*_Lie_deriv.txt'))
    
    if not oic_files:
        raise FileNotFoundError(f"No observability matrix file found for model {options.modelname}")
    
    # Use the most recent file (highest number of derivatives)
    latest_file = max(oic_files, key=lambda f: extract_lie_deriv_number(f.name))
    print(f"Using matrix file: {latest_file}")
    
    # Load the matrix
    onx_matrix = load_oic_matrix(latest_file, options.modelname)
    
    # Step 3: Extract model information for nullspace analysis
    print("\nStep 3: Preparing for nullspace analysis...")
    
    # Get state and parameter symbols
    state_symbols = extract_symbols(model.x)
    param_symbols = extract_symbols(model.p)
    input_symbols = extract_symbols(model.w)
    
    print(f"States: {state_symbols}")
    print(f"Parameters: {param_symbols}")
    print(f"Unknown inputs: {input_symbols}")  
    
    # Step 4: Perform nullspace analysis
    # print(f"\nStep 4: Performing nullspace analysis (scope: {analysis_scope})...")
    
    # nullspace_results = analyze_identifiable_combinations(
    #     onx_matrix, 
    #     param_symbols, 
    #     state_symbols,
    #     input_symbols,  # Add this if not already there
    #     analysis_scope  # Add this parameter
    # )
    if not checkpoint:
        print(f"\nStep 4: Performing nullspace analysis (scope: {analysis_scope})...")
        
        nullspace_results = analyze_identifiable_combinations(
            onx_matrix, 
            param_symbols, 
            state_symbols,
            input_symbols,
            analysis_scope
        )
        
        # Save checkpoint after analysis
        identifiable_results = nullspace_results.get('identifiable_info', {})
        save_checkpoint(options.modelname, options_file, model_hash, 
                    onx_matrix, nullspace_results, identifiable_results)
    # Step 5: Generate comprehensive report
    print("\nStep 5: Generating comprehensive report...")
    
    # Generate timestamp once for all outputs
    from datetime import datetime
    shared_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'model_name': options.modelname,
        'matrix_rank': onx_matrix.rank(),
        'matrix_shape': onx_matrix.shape,
        'states': state_symbols,
        'parameters': param_symbols,
        'nullspace_analysis': nullspace_results
    }
    
    # Save detailed results
    save_detailed_results(results, options.modelname, shared_timestamp)
    
    # Step 6: Visualization (if nullspace exists)
    if not nullspace_results.get('fully_identifiable', False):
        print("\nStep 6: Creating visualizations...")
        create_visualizations(
            nullspace_results, 
            param_symbols, 
            options.modelname, 
            analysis_scope,
            state_symbols,
            input_symbols,
            manifold_config,
            shared_timestamp        # Add this parameter
        )
        
    # # Step 6: Visualization (if nullspace exists)
    # if not nullspace_results.get('fully_identifiable', False):
    #     print("\nStep 6: Creating visualizations...")
    #     # create_visualizations(nullspace_results, param_symbols)
    #     create_visualizations(nullspace_results, param_symbols, options.modelname)
    
    return results

def extract_lie_deriv_number(filename):
    """Extract the number of Lie derivatives from filename."""
    import re
    match = re.search(r'(\d+)_Lie_deriv\.txt$', filename)
    return int(match.group(1)) if match else 0

def load_oic_matrix(filepath, model_name):
    """Load the observability-identifiability matrix from StrikePy output file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the matrix data (it's stored as "onx = [...]")
    matrix_str = content.split('onx = ')[1]
    
    # Import the model to get all symbol definitions
    import importlib
    try:
        model = importlib.import_module(f'custom_models.{model_name}')
        # Get all variables from the model's namespace
        model_vars = model.variables_locales if hasattr(model, 'variables_locales') else vars(model)
    except Exception as e:
        print(f"Warning: Could not import model {model_name}: {e}")
        model_vars = {}
    
    # Add sympy to the namespace
    import sympy as sym
    exec_namespace = {
        **sym.__dict__,  # All sympy functions and classes
        **model_vars     # All model variables
    }
    
    # Use exec to evaluate the symbolic expression
    local_vars = {}
    exec(f"result = {matrix_str}", {"__builtins__": {}, **exec_namespace}, local_vars)
    matrix_data = local_vars['result']
    
    # Convert to sympy Matrix
    return Matrix(matrix_data)

def extract_symbols(symbol_list):
    """Extract sympy symbols from model lists (handles nested lists)."""
    symbols = []
    for item in symbol_list:
        if isinstance(item, list):
            symbols.extend(extract_symbols(item))
        else:
            symbols.append(item)
    return symbols

def save_detailed_results(results, model_name, timestamp=None):
    """Save comprehensive analysis results to file."""
    from datetime import datetime
    
    # Ensure results directory exists
    results_dir = get_results_dir()
    results_dir.mkdir(exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # filename = results_dir / f"detailed_analysis_{model_name}_{timestamp}.txt"
    results_base = results_dir / model_name / timestamp
    results_base.mkdir(parents=True, exist_ok=True)
    filename = results_base / f"detailed_analysis.txt"
    
    with open(filename, 'w') as f:
        f.write("COMPREHENSIVE IDENTIFIABILITY ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {results['model_name']}\n")
        f.write(f"Matrix shape: {results['matrix_shape']}\n")
        f.write(f"Matrix rank: {results['matrix_rank']}\n\n")
        
        f.write(f"States: {results['states']}\n")
        f.write(f"Parameters: {results['parameters']}\n\n")
        
        nullspace = results['nullspace_analysis']
        if nullspace.get('fully_identifiable', False):
            f.write("RESULT: All parameters are individually identifiable!\n")
        else:
            f.write(f"Nullspace dimension: {nullspace['nullspace_dimension']}\n\n")
            
            # Write nullspace vectors
            f.write("NULLSPACE VECTORS:\n")
            for i, vec in enumerate(nullspace.get('nullspace_basis', [])):
                f.write(f"Vector {i+1}: {vec}\n")
            f.write("\n")
            
            # Write parameter relationships
            f.write("UNIDENTIFIABLE PARAMETER RELATIONSHIPS:\n")
            for i, pattern in enumerate(nullspace.get('unidentifiable_patterns', [])):
                f.write(f"{i+1}. {pattern['relationship']}\n")
                f.write(f"   Interpretation: {pattern['interpretation']}\n")
                
                # Add detailed breakdown by variable type
                if pattern['unobservable_states']:
                    state_names = [str(s) for s, _ in pattern['unobservable_states']]
                    f.write(f"   Unobservable states: {', '.join(state_names)}\n")
                
                if pattern['parameter_combination']:
                    param_names = [str(p) for p, _ in pattern['parameter_combination']]
                    f.write(f"   Unidentifiable parameters: {', '.join(param_names)}\n")
                
                if pattern['input_combination']:
                    input_names = [str(inp) for inp, _ in pattern['input_combination']]
                    f.write(f"   Unobservable inputs: {', '.join(input_names)}\n")
                
                f.write("\n")
            f.write("\n")
            
            
            # Write identifiable combinations
            identifiable_info = nullspace.get('identifiable_info', {})
            if identifiable_info.get('error'):
                f.write(f"ERROR in identifiable combination calculation: {identifiable_info['error']}\n")
            elif not identifiable_info.get('all_vars_identifiable', True):
                f.write("IDENTIFIABLE PARAMETER COMBINATIONS:\n")
                combos = identifiable_info.get('identifiable_combinations', [])
                if combos:
                    for i, combo in enumerate(combos):
                        f.write(f"{i+1}. {combo}\n")
                        
                        # Extract variables involved in this combination
                        involved_vars = []
                        for sym in results['states'] + results['parameters']:
                            if str(sym) in combo:
                                involved_vars.append(str(sym))
                        
                        if involved_vars:
                            f.write(f"   Variables involved: {', '.join(involved_vars)}\n")
                        f.write("\n")
                else:
                    f.write("No identifiable combinations found (this may indicate an error)\n")
                f.write(f"Number of identifiable directions: {identifiable_info.get(
                    'n_identifiable_combinations', 0)}\n")
            else:
                f.write("All parameters are individually identifiable.\n")
            
    print(f"Detailed results saved to: {filename}")
    
# def create_visualizations(nullspace_results, param_symbols, model_name, 
#                           analysis_scope='full', state_symbols=None, input_symbols=None):
def create_visualizations(nullspace_results, param_symbols, model_name, 
                          analysis_scope='full', state_symbols=None, input_symbols=None,
                          manifold_config=None, shared_timestamp=None):
    """Create visualizations for parameter relationships."""
    try:
        from datetime import datetime
        if shared_timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = shared_timestamp
        
        # Ensure results directory exists
        results_dir = get_results_dir()
        results_dir.mkdir(exist_ok=True)
        

        # ---- NEW: Build the canonical symbol universe (state -> param -> input) ----
        symbol_universe = []
        if state_symbols:
            symbol_universe.extend(state_symbols)
        # param_symbols is required in your signature; still guard for safety
        if param_symbols:
            symbol_universe.extend(param_symbols)
        if input_symbols:
            symbol_universe.extend(input_symbols)
        # ---------------------------------------------------------------------------
        # Create identifiability graph
        if nullspace_results.get('nullspace_basis'):
            
            if analysis_scope == 'full':
                # Plot all variables: states + parameters + inputs
                print("Creating visualization for all variables (states + parameters + inputs)...")
                
                all_symbols = []
                symbol_types = []  
                
                n_states = len(state_symbols) if state_symbols else 0
                n_params = len(param_symbols)
                n_inputs = len(input_symbols) if input_symbols else 0
                
                if state_symbols:
                    all_symbols.extend(state_symbols) 
                    symbol_types.extend(['state'] * len(state_symbols))
                    
                all_symbols.extend(param_symbols)
                symbol_types.extend(['param'] * len(param_symbols))  
                
                if input_symbols:
                    all_symbols.extend(input_symbols)
                    symbol_types.extend(['input'] * len(input_symbols))
                
                if nullspace_results['nullspace_basis']:
                        G = build_identifiability_graph(
                            all_symbols, 
                            nullspace_results['nullspace_basis'], 
                            symbol_types, 
                            index_range=None  # Full vector
                        )
                        # save_path = f"results/identifiability_graph_full_{model_name}_{timestamp}.png"
                        
                        results_base = get_results_dir() / model_name / timestamp
                        results_base.mkdir(parents=True, exist_ok=True)
                        (results_base / "graphs").mkdir(exist_ok=True)
                        save_path = results_base / "graphs" / "identifiability_graph_full.png"                        
                        
                        
                        visualize_identifiability_graph(G, 
                            title=f"Full Variable Identifiability Structure - {model_name}",
                            save_path=save_path)
            
            elif analysis_scope == 'parameters':
                # Plot only parameters
                print("Creating visualization for parameters only...")
                n_states = len(state_symbols) if state_symbols else 0
                n_params = len(param_symbols)
                param_start = n_states
                param_end = n_states + n_params
                
                symbol_types = ['param'] * len(param_symbols)
    
                if nullspace_results['nullspace_basis']:
                    G = build_identifiability_graph(
                        param_symbols, 
                        nullspace_results['nullspace_basis'], 
                        symbol_types, 
                        index_range=(param_start, param_end)  # Only parameters
                    )

                    results_base = get_results_dir() / model_name / timestamp
                    results_base.mkdir(parents=True, exist_ok=True)
                    (results_base / "graphs").mkdir(exist_ok=True)
                    save_path = results_base / "graphs" / "identifiability_graph_params.png"                    
                    
                    visualize_identifiability_graph(G, 
                        title=f"Parameter Identifiability Structure - {model_name}",
                        save_path=save_path)
        # ADD MANIFOLD VISUALIZATION
        try:
            print("\nCreating manifold visualizations...")

            visualize_nullspace_manifolds(
                nullspace_results,
                state_symbols,
                param_symbols,
                symbol_universe,
                plot_cfg=manifold_config or {},
                input_symbols=input_symbols or [],
                model_name=model_name,
                shared_timestamp=timestamp,
            )
        except Exception as e:
            print(f"Manifold visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        
def analyze_c2m_model():
    """Analyze the two-compartment model specifically."""
    return run_integrated_analysis(options_file='options_C2M')

def analyze_default_model():
    """Analyze using default options."""
    return run_integrated_analysis()

# Example usage and demonstration

def demonstrate_integration():
    """Demonstrate the integrated analysis workflow."""
    
    print("DEMONSTRATION: Integrated StrikePy + Nullspace Analysis")
    print("=" * 60)
    
    # Example 1: Analyze C2M model
    print("\nExample 1: Two-Compartment Model (C2M)")
    print("-" * 40)
    
    try:
        results = analyze_c2m_model()
        
        print(f"\nAnalysis completed for {results['model_name']}")
        print(f"Matrix dimensions: {results['matrix_shape']}")
        print(f"Matrix rank: {results['matrix_rank']}")
        
        if not results['nullspace_analysis']['fully_identifiable']:
            print("\nParameter relationships found:")
            for rel in results['nullspace_analysis'].get('parameter_relationships', []):
                print(f"  {rel['relationship']}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Make sure you have the C2M model and options_C2M.py configured")
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_integration()