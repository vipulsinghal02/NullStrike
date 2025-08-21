import sympy as sym
import numpy as np
from sympy import Matrix, symbols, simplify, solve, zeros, diff, Eq
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import utilities
from ..utils import get_results_dir

# from ..visualization.graphs import build_identifiability_graph, visualize_identifiability_graph
# from ..visualization.manifolds import visualize_nullspace_manifolds
# from . import build_identifiability_graph, visualize_identifiability_graph, visualize_nullspace_manifolds # I dont think any of these are being used, man. 

def analyze_strikepy_matrix(onx_matrix, model, analysis_scope='full'):
    """
    Analyze StrikePy's observability-identifiability matrix for parameter combinations.
    
    Parameters:
    -----------
    onx_matrix : sympy.Matrix
        The observability-identifiability matrix from StrikePy
    model : module
        The model module containing x, p, h, f definitions
    analysis_scope : str
        'full' - analyze all variables (default)
        'parameters' - analyze only parameters
        Returns:
    --------
    dict : Complete analysis results
    """
    
    
    # Extract model components
    state_symbols = extract_flat_symbols(model.x)
    param_symbols = extract_flat_symbols(model.p)
    
    # Handle unknown inputs if they exist
    input_symbols = []
    if hasattr(model, 'w') and model.w:
        input_symbols = extract_flat_symbols(model.w)
    
    print(f"Analyzing matrix of size {onx_matrix.shape}")
    print(f"States ({len(state_symbols)}): {state_symbols}")
    print(f"Parameters ({len(param_symbols)}): {param_symbols}")
    if input_symbols:
        print(f"Unknown inputs ({len(input_symbols)}): {input_symbols}")
    
    # Perform nullspace analysis
    # results = analyze_identifiable_combinations(
    #     onx_matrix, 
    #     param_symbols, 
    #     state_symbols,
    #     input_symbols
    # )
    results = analyze_identifiable_combinations(
        onx_matrix, 
        param_symbols, 
        state_symbols,
        input_symbols,
        analysis_scope)
    
    # Add model-specific interpretations
    results['model_info'] = {
        'name': getattr(model, '__name__', 'unknown'),
        'states': state_symbols,
        'parameters': param_symbols,
        'inputs': input_symbols,
        'outputs': extract_flat_symbols(model.h) if hasattr(model, 'h') else []
    }
    
    return results

def extract_flat_symbols(nested_list):
    """Extract symbols from potentially nested lists (StrikePy format)."""
    symbols = []
    if not nested_list:
        return symbols
        
    for item in nested_list:
        if isinstance(item, list):
            symbols.extend(extract_flat_symbols(item))
        elif hasattr(item, 'is_Symbol') and item.is_Symbol:
            symbols.append(item)
        else:
            # Try to convert to symbol if it's a string
            try:
                symbols.append(sym.Symbol(str(item)))
            except:
                pass
    
    return symbols
def analyze_identifiable_combinations(O_matrix, param_symbols, state_symbols, input_symbols=None, analysis_scope='full'):
    """
    Enhanced version that handles the specific structure of StrikePy's augmented state vector.
    
    Parameters:
    -----------
    analysis_scope : str
        'full' - analyze all variables (states + parameters + inputs) [DEFAULT]
        'parameters' - analyze only parameters
    """
    
    if input_symbols is None:
        input_symbols = []
    
    n_states = len(state_symbols)
    n_params = len(param_symbols) 
    n_inputs = len(input_symbols)
    
    print("\n" + "="*60)
    print("ENHANCED NULLSPACE ANALYSIS")
    print("="*60)
    
    print(f"Augmented state vector structure:")
    print(f"  States (indices 0-{n_states-1}): {state_symbols}")
    print(f"  Parameters (indices {n_states}-{n_states+n_params-1}): {param_symbols}")
    if n_inputs > 0:
        print(f"  Unknown inputs (indices {n_states+n_params}-{n_states+n_params+n_inputs-1}): {input_symbols}")
    
    # Compute nullspace
    print("\nComputing nullspace...")
    nullspace_basis = O_matrix.nullspace()
    
    if len(nullspace_basis) == 0:
        print("✓ The nullspace is empty - system is fully observable and identifiable!")
        return {
            "fully_identifiable": True,
            "matrix_rank": O_matrix.rank(),
            "expected_rank": O_matrix.cols
        }
    
    print(f"Nullspace dimension: {len(nullspace_basis)}")
    print(f"Matrix rank: {O_matrix.rank()}/{O_matrix.cols}")
    print(f"Rank deficiency: {O_matrix.cols - O_matrix.rank()}")
    
    # Analyze each nullspace vector
    unidentifiable_patterns = []
    
    for i, null_vec in enumerate(nullspace_basis):
        print(f"\n--- Nullspace Vector {i+1} ---")
        print(f"Full vector: {null_vec}")
        
        # Extract components
        state_part = null_vec[:n_states] if n_states > 0 else []
        param_part = null_vec[n_states:n_states+n_params] if n_params > 0 else []
        input_part = null_vec[n_states+n_params:] if n_inputs > 0 else []
        
        print(f"State components: {state_part}")
        print(f"Parameter components: {param_part}")
        if input_part:
            print(f"Input components: {input_part}")
        
        # Analyze state observability
        unobs_states = []
        for j, coeff in enumerate(state_part):
            if coeff != 0:
                unobs_states.append((state_symbols[j], coeff))
        
        # Analyze parameter identifiability  
        param_combo = []
        for j, coeff in enumerate(param_part):
            if coeff != 0:
                param_combo.append((param_symbols[j], coeff))
        
        # Analyze input observability
        input_combo = []
        for j, coeff in enumerate(input_part):
            if coeff != 0:
                input_combo.append((input_symbols[j], coeff))
        
        # Create human-readable relationship
        relationship = build_relationship_string(param_combo, input_combo, unobs_states)
        
        pattern = {
            'vector_index': i,
            'nullspace_vector': null_vec,  # Store the original vector!
            'unobservable_states': unobs_states,
            'parameter_combination': param_combo,
            'input_combination': input_combo,
            'relationship': relationship,
            'interpretation': interpret_pattern(param_combo, input_combo, unobs_states)
        }
        
        unidentifiable_patterns.append(pattern)
        
        print(f"Relationship: {relationship}")
        print(f"Interpretation: {pattern['interpretation']}")
    
    identifiable_info = find_identifiable_directions(
        O_matrix, param_symbols, state_symbols, input_symbols, 
        nullspace_basis, unidentifiable_patterns, analysis_scope  # Add this parameter!
    )
    
    return {
        "fully_identifiable": False,
        "matrix_rank": O_matrix.rank(),
        "expected_rank": O_matrix.cols,
        "nullspace_dimension": len(nullspace_basis),
        "nullspace_basis": nullspace_basis,
        "unidentifiable_patterns": unidentifiable_patterns,
        "identifiable_info": identifiable_info
    }
    
def find_identifiable_directions(O_matrix, param_symbols, state_symbols, input_symbols, 
                               nullspace_vectors, patterns, analysis_scope='full'):
    """
    Find which parameter/state/input combinations ARE identifiable.
    
    Parameters:
    -----------
    O_matrix : sympy.Matrix
        The observability-identifiability matrix
    param_symbols : list
        Parameter symbols
    state_symbols : list  
        State symbols
    input_symbols : list
        Input symbols
    nullspace_vectors : list
        The actual nullspace vectors from O_matrix.nullspace()
    patterns : list
        Unidentifiable patterns (for interpretation only)
    analysis_scope : str
        'full' - analyze all variables (states + parameters + inputs)
        'parameters' - analyze only parameters
        
    Returns:
    --------
    dict : Identifiable combinations and analysis results
    """
    
    n_states = len(state_symbols)
    n_params = len(param_symbols)
    n_inputs = len(input_symbols)
    
    if analysis_scope == 'parameters':
        print("Finding identifiable PARAMETER directions...")
        working_symbols = param_symbols
        # Extract only parameter components from nullspace vectors
        param_nullspace_vectors = []
        for vec in nullspace_vectors:
            param_part = vec[n_states:n_states+n_params]
            param_nullspace_vectors.append([coeff for coeff in param_part])
        working_nullspace_vectors = param_nullspace_vectors
        
    else:  # 'full'
        print("Finding identifiable directions for ALL variables...")
        working_symbols = state_symbols + param_symbols + input_symbols
        # Use full nullspace vectors
        working_nullspace_vectors = [[coeff for coeff in vec] for vec in nullspace_vectors]
    
    print(f"Analysis scope: {analysis_scope}")
    print(f"Working with {len(working_symbols)} variables: {[str(s) for s in working_symbols]}")
    
    if not working_nullspace_vectors:
        return {
            "all_vars_identifiable": True,
            "identifiable_combinations": [str(s) for s in working_symbols],
            "analysis_scope": analysis_scope
        }
    
    # Create nullspace matrix (rows = nullspace vectors) - no reconstruction needed!
    N = Matrix(working_nullspace_vectors)
    print(f"Nullspace matrix N (shape {N.shape}):\n{N}")
    print(f"Rank of N: {N.rank()}")
    print(f"Expected identifiable dimension: {len(working_symbols) - N.rank()}")
    
    # Method 1: Use row space of observability matrix directly
    identifiable_combinations = []
    
    try:
        if analysis_scope == 'full':
            # Get row space of the full observability matrix
            row_space_basis = O_matrix.T.columnspace()
            print(f"Method 1: Found {len(row_space_basis)} identifiable directions from O_matrix row space")
        else:
            # For parameters only, extract parameter columns
            param_start = n_states
            param_end = n_states + n_params
            O_param = O_matrix[:, param_start:param_end]
            row_space_basis = O_param.T.columnspace()
            print(f"Method 1: Found {len(row_space_basis)} identifiable parameter directions")
        
        for i, vec in enumerate(row_space_basis):
            combo_terms = []
            for j, coeff in enumerate(vec):
                if coeff != 0:
                    if coeff == 1:
                        combo_terms.append(str(working_symbols[j]))
                    elif coeff == -1:
                        combo_terms.append(f"-{working_symbols[j]}")
                    else:
                        combo_terms.append(f"({coeff})*{working_symbols[j]}")
            
            if combo_terms:
                combo_expr = " + ".join(combo_terms).replace("+ -", "- ")
                identifiable_combinations.append(combo_expr)
                print(f"\nIdentifiable direction {i+1}: {combo_expr}")
    
    except Exception as e:
        print(f"Method 1 (row space) failed: {e}")
        identifiable_combinations = []
    
    # Method 2: Nullspace of nullspace matrix
    if not identifiable_combinations:
        try:
            print("Method 2: Computing nullspace of nullspace matrix...")
            
            # Find vectors orthogonal to all nullspace vectors
            orthogonal_vectors = N.nullspace()
            
            print(f"Method 2: Found {len(orthogonal_vectors)} identifiable directions")
            
            for i, vec in enumerate(orthogonal_vectors):
                combo_terms = []
                for j, coeff in enumerate(vec):
                    if coeff != 0:
                        if coeff == 1:
                            combo_terms.append(str(working_symbols[j]))
                        elif coeff == -1:
                            combo_terms.append(f"-{working_symbols[j]}")
                        else:
                            combo_terms.append(f"({coeff})*{working_symbols[j]}")
                
                if combo_terms:
                    combo_expr = " + ".join(combo_terms).replace("+ -", "- ")
                    identifiable_combinations.append(combo_expr)
                    print(f"\nIdentifiable combination {i+1}: {combo_expr}")
        
        except Exception as e:
            print(f"Method 2 (nullspace) also failed: {e}")
    
    # Build results
    results = {
        "all_vars_identifiable": False,
        "n_identifiable_combinations": len(identifiable_combinations),
        "identifiable_combinations": identifiable_combinations,
        "nullspace_dimension": len(working_nullspace_vectors),
        "expected_identifiable_dimension": len(working_symbols) - len(working_nullspace_vectors),
        "analysis_scope": analysis_scope
    }
    
    # Classify combinations by variable type (only for full analysis)
    if analysis_scope == 'full':
        param_combinations = []
        state_combinations = []
        input_combinations = []
        mixed_combinations = []
        
        for combo in identifiable_combinations:
            has_param = any(str(p) in combo for p in param_symbols)
            has_state = any(str(s) in combo for s in state_symbols)
            has_input = any(str(i) in combo for i in input_symbols)
            
            if has_param and not has_state and not has_input:
                param_combinations.append(combo)
            elif has_state and not has_param and not has_input:
                state_combinations.append(combo)
            elif has_input and not has_param and not has_state:
                input_combinations.append(combo)
            else:
                mixed_combinations.append(combo)
        
        results.update({
            "parameter_combinations": param_combinations,
            "state_combinations": state_combinations,
            "input_combinations": input_combinations,
            "mixed_combinations": mixed_combinations
        })
    
    return results


def build_relationship_string(param_combo, input_combo, state_combo):
    """Build a human-readable string describing the unidentifiable combination."""
    
    terms = []
    
    # Add parameter terms
    for param, coeff in param_combo:
        if coeff == 1:
            terms.append(str(param))
        elif coeff == -1:
            terms.append(f"-{param}")
        else:
            terms.append(f"{coeff}*{param}")
    
    # Add input terms  
    for inp, coeff in input_combo:
        if coeff == 1:
            terms.append(str(inp))
        elif coeff == -1:
            terms.append(f"-{inp}")
        else:
            terms.append(f"{coeff}*{inp}")
    
    # Add state terms
    for state, coeff in state_combo:
        if coeff == 1:
            terms.append(str(state))
        elif coeff == -1:
            terms.append(f"-{state}")
        else:
            terms.append(f"{coeff}*{state}")
    
    if not terms:
        return "No relationship found"
    
    relationship = " + ".join(terms).replace("+ -", "- ")
    return f"({relationship}) is unidentifiable"

def interpret_pattern(param_combo, input_combo, state_combo):
    """Provide physical interpretation of unidentifiable patterns."""
    
    interpretations = []
    
    # Common patterns in compartmental models
    if len(param_combo) == 2:
        p1, c1 = param_combo[0]
        p2, c2 = param_combo[1]
        
        if c1 == 1 and c2 == -1:
            interpretations.append(f"Only the difference ({p1} - {p2}) matters")
        elif c1 == -1 and c2 == 1:
            interpretations.append(f"Only the difference ({p2} - {p1}) matters")
        elif c1 == c2:
            interpretations.append(f"Parameters {p1} and {p2} are perfectly correlated")
    
    if len(param_combo) > 2 and all(c == param_combo[0][1] for _, c in param_combo):
        params = [str(p) for p, _ in param_combo]
        interpretations.append(f"Parameters {params} can only be scaled together (time-scale ambiguity)")
    
    if param_combo and input_combo:
        interpretations.append("Input-parameter tradeoff detected")
    
    if param_combo and state_combo:
        interpretations.append("Parameter-initial condition tradeoff detected")
    
    if not interpretations:
        interpretations.append("Complex parameter relationship - see mathematical expression")
    
    return " | ".join(interpretations)


def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE IDENTIFIABILITY SUMMARY")
    print("="*70)
    
    if results['fully_identifiable']:
        print("\n✓ EXCELLENT: System is fully observable and identifiable!")
        print("  All parameters can be uniquely determined from the available measurements.")
        return
    
    print(f"\nMatrix Analysis:")
    print(f"  Rank: {results['matrix_rank']}/{results['expected_rank']}")
    print(f"  Rank deficiency: {results['expected_rank'] - results['matrix_rank']}")
    print(f"  Nullspace dimension: {results['nullspace_dimension']}")
    
    print(f"\nUnidentifiable Patterns ({len(results['unidentifiable_patterns'])}):")
    for i, pattern in enumerate(results['unidentifiable_patterns']):
        print(f"  {i+1}. {pattern['relationship']}")
        print(f"     → {pattern['interpretation']}")
    
    identifiable_info = results['identifiable_info']
    if identifiable_info['all_vars_identifiable']:
        print(f"\n✓ All parameters are individually identifiable!")
    else:
        n_id = identifiable_info['n_identifiable_combinations']
        print(f"\nIdentifiable Information ({n_id} combinations):")
        for i, combo in enumerate(identifiable_info['identifiable_combinations']):
            print(f"  {i+1}. {combo}")
    
    print(f"\nRecommendations:")
    if results['nullspace_dimension'] > 0:
        print(f"  • Fix {results['nullspace_dimension']} parameters to known values, OR")
        print(f"  • Add {results['nullspace_dimension']} additional independent measurements, OR")
        print(f"  • Reparameterize using the identifiable combinations shown above")


def load_oic_matrix_with_symbols(filepath, model_name):
    """Load matrix with proper symbol context."""
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Extract the matrix data (it's stored as "onx = [...]")
    matrix_str = content.split('onx = ')[1]
    
    # Import the model to get symbols
    import importlib
    import sympy as sym
    
    try:
        model = importlib.import_module(f'custom_models.{model_name}')
        model_vars = model.variables_locales if hasattr(model, 'variables_locales') else vars(model)
    except Exception as e:
        print(f"Warning: Could not import model {model_name}: {e}")
        model_vars = {}
    
    # Create execution namespace with all symbols
    exec_namespace = {**sym.__dict__, **model_vars}
    
    local_vars = {}
    exec(f"result = {matrix_str}", {"__builtins__": {}, **exec_namespace}, local_vars)
    
    return Matrix(local_vars['result'])

# Usage example specifically for StrikePy integration
def analyze_strikepy_results(model_name='C2M', analysis_scope='full'):
    """
    Complete analysis of StrikePy results for a given model.
    
    Parameters:
    -----------
    analysis_scope : str
        'full' - analyze all variables (default)
        'parameters' - analyze only parameters
    """
    # Import model
    import importlib
    # model = importlib.import_module(f'models.{model_name}') #NOTE: what is this models. thing?  
    try:
        model = importlib.import_module(f'custom_models.{model_name}')
    except ImportError:
        # Try without the models prefix
        model = importlib.import_module(model_name)
        
    # Load most recent OIC matrix
    from pathlib import Path
    import ast
    
    results_dir = get_results_dir()
    oic_files = list(results_dir.glob(f'obs_ident_matrix_{model_name}_*_Lie_deriv.txt'))
    
    if not oic_files:
        raise FileNotFoundError(f"No OIC matrix found for {model_name}. Run StrikePy first!")
    
    latest_file = max(oic_files) #PATTERN -- cool file loading pattern. Memorize. 
    
    onx_matrix = load_oic_matrix_with_symbols(latest_file, model_name)
    
    print(f"Loaded OIC matrix from: {latest_file}")
    
    # Perform analysis
    results = analyze_strikepy_matrix(onx_matrix, model, analysis_scope)
    
    # Generate report
    generate_summary_report(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    results = analyze_strikepy_results('C2M')