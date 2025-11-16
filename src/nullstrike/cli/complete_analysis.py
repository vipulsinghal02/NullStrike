#!/usr/bin/env python3
"""
Complete usage script for integrating StrikePy with nullspace analysis.

This script demonstrates how to:
1. Run StrikePy analysis 
2. Load the observability-identifiability matrix
3. Perform nullspace analysis to find parameter combinations
4. Generate comprehensive reports

Usage:
    python complete_analysis.py [model_name] [options_file]

Examples:
    python complete_analysis.py                    # Use default options
    python complete_analysis.py C2M               # Analyze C2M model with default options  
    python complete_analysis.py C2M options_C2M   # Use custom options
"""

import sys
import os
from pathlib import Path
from ..core import strike_goldd

from ..analysis import load_checkpoint, compute_model_hash

def main():
    """Main function to run complete analysis."""
    
    # Parse command line arguments - ADD SCOPE PARSING
    model_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith(
        '--') else 'calibration_single'
    
    # Determine options file: specific options file > default options file
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        # User specified options file explicitly
        options_file = sys.argv[2]
    else:
        # Check if model-specific options file exists
        model_specific_options = f'options_{model_name}'
        try:
            # Try to import the model-specific options to see if it exists
            __import__(f'custom_options.{model_specific_options}')
            options_file = model_specific_options
        except ImportError:
            # Fall back to default options file
            options_file = 'options_default'
    
    # Check for scope option
    analysis_scope = 'full'  # default
    if '--parameters-only' in sys.argv or '-p' in sys.argv:
        analysis_scope = 'parameters'
    
    print(f"Analysis scope: {analysis_scope}")
    
    print("="*80)
    print("COMPLETE IDENTIFIABILITY ANALYSIS")
    print("StrikePy + Nullspace Analysis")
    print("="*80)

    try:
        

        print("\nMETHOD 1: Integrated Analysis")
        
        print("-" * 40)
        from nullstrike.analysis.integrated_analysis import run_integrated_analysis
        results = run_integrated_analysis(model_name, options_file, analysis_scope)
                
        print(f"\n✓ Analysis completed successfully for model: {results['model_name']}")
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("\nTrying Method 2: Step-by-step analysis...")
        
        try:
            # Method 2: Step-by-step (fallback)
            print("\nMETHOD 2: Step-by-Step Analysis") 
            print("-" * 40)
            
            results = run_step_by_step_analysis(model_name, options_file)
            
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
            print("\nTrying Method 3: Analyze existing results...")
            
            # Method 3: Just analyze existing StrikePy results
            if model_name:
                from nullstrike.analysis.enhanced_subspace import analyze_strikepy_results
                results = analyze_strikepy_results(model_name)
            else:
                print("Please provide a model name or run StrikePy first.")
                return
    
    # Display final summary
    display_final_summary(results)

def run_step_by_step_analysis(model_name=None, options_file=None):
    """Run analysis step by step as fallback method."""
    # Check for checkpoint first
    if model_name and options_file:
        try:
            import importlib
            # model = importlib.import_module(f'nullstrike.models.{model_name}')
            try:
                # Try to load from your package first
                model = importlib.import_module(f'nullstrike.models.{model_name}')
            except ImportError:
                # Fall back to custom_models directory 
                model = importlib.import_module(f'custom_models.{model_name}')            
            options = __import__(f'custom_options.{options_file}', fromlist=[''])
            model_hash = compute_model_hash(model, options)
            checkpoint = load_checkpoint(model_name, options_file, model_hash)
            if checkpoint:
                print("Found valid checkpoint - using cached analysis")
                return {
                    'model_name': model_name,
                    'nullspace_analysis': checkpoint['nullspace_results'],
                    'matrix_rank': checkpoint['oic_matrix'].rank(),
                    'matrix_shape': checkpoint['oic_matrix'].shape,
                    'fully_identifiable': checkpoint['nullspace_results'].get(
                        'fully_identifiable', False),
                    'nullspace_dimension': checkpoint['nullspace_results'].get(
                        'nullspace_dimension', 0),
                    'unidentifiable_patterns': checkpoint['nullspace_results'].get(
                        'unidentifiable_patterns', []),
                    'identifiable_info': checkpoint['nullspace_results'].get(
                        'identifiable_info', {})
                }
        except Exception:
            pass  # Fall back to normal analysis    
    
    # Step 1: Run StrikePy
    print("Step 1: Running StrikePy analysis...")
    from strike_goldd import strike_goldd
    
    if options_file:
        strike_goldd(options_file)
        options = __import__(f'custom_options.{options_file}', fromlist=[''])
    else:
        strike_goldd()
        from nullstrike.configs import default_options as options

    # For fallback methods, we'd need to also handle MANIFOLD_PLOTTING here
    # if they call visualization functions directly
    
    actual_model_name = model_name or options.modelname
    
    # Step 2: Load and analyze results
    print("Step 2: Performing nullspace analysis...")
    from nullstrike.analysis.enhanced_subspace import analyze_strikepy_results
    
    results = analyze_strikepy_results(actual_model_name)
    
    return results

def display_final_summary(results):
    """Display a final summary of all results."""
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    model_info = results.get('model_info', {})
    model_name = model_info.get('name', results.get('model_name', 'Unknown'))
    
    print(f"\nModel: {model_name}")
    
    if results.get('fully_identifiable', False):
        print("✓ STATUS: Fully identifiable and observable")
        print("  All parameters and states can be uniquely determined.")
    else:
        nullspace_dim = results.get('nullspace_dimension', 0)
        if nullspace_dim > 0:
            print(f"⚠ STATUS: Partially identifiable")
            print(f"  {nullspace_dim} parameter combinations are unidentifiable")
            
            # Show key relationships
            patterns = results.get('unidentifiable_patterns', [])
            if patterns:
                print(f"\nKey unidentifiable relationships:")
                for i, pattern in enumerate(patterns[:3]):  # Show first 3
                    print(f"  {i+1}. {pattern['relationship']}")
                if len(patterns) > 3:
                    print(f"     ... and {len(patterns)-3} more")
            
            # Show what IS identifiable
            identifiable_info = results.get('identifiable_info', {})
            if identifiable_info and not identifiable_info.get(
                'all_params_identifiable', True):
                combos = identifiable_info.get('identifiable_combinations', [])
                if combos:
                    print(f"\nIdentifiable combinations:")
                    for i, combo in enumerate(combos[:3]):
                        print(f"  {i+1}. {combo}")
                    if len(combos) > 3:
                        print(f"     ... and {len(combos)-3} more")
    
    # Practical recommendations
    print(f"\nPRACTICAL RECOMMENDATIONS:")
    
    if results.get('fully_identifiable', False):
        print("  • Your model is well-designed for parameter estimation")
        print("  • All parameters should be estimable from your measurements")
    else:
        nullspace_dim = results.get('nullspace_dimension', 0)
        if nullspace_dim > 0:
            print(f"  • Consider fixing {nullspace_dim} parameters to literature values")
            print(f"  • Or add {nullspace_dim} additional measurements")
            print(f"  • Or reparameterize using identifiable combinations")
    
    print(f"\nDetailed results saved in: results/ directory")
    print(f"Check the files for complete mathematical details.")

def quick_demo():
    """Quick demonstration with the C2M model."""
    
    print("\n" + "="*60)
    print("QUICK DEMO: Two-Compartment Model Analysis")
    print("="*60)
    
    try:
        # Analyze C2M model
        from nullstrike.analysis.enhanced_subspace import analyze_strikepy_results
        results = analyze_strikepy_results('C2M')
        
        print("\n✓ Demo completed successfully!")
        
        # Show key insights for C2M
        print("\nC2M Model Insights:")
        print("This two-compartment model typically shows:")
        print("• Parameter combinations rather than individual parameters are identifiable")
        print("• Common patterns include sum/difference relationships between rate constants")
        print("• Input scaling may trade off with initial conditions")
        
        return results
        
    except FileNotFoundError:
        print("\n⚠ No existing C2M results found.")
        print("Run StrikePy first with: python RunModels.py")
        print("Make sure options.py has modelname = 'C2M'")
        return None
    except Exception as e:
        print(f"\nDemo failed: {e}")
        return None


def help_usage():
    """Display detailed usage instructions."""

    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    

    print("\n1. BASIC USAGE:")
    print("   python complete_analysis.py                    # Default: full analysis")
    print("   python complete_analysis.py --parameters-only  # Parameters only")
    print("   python complete_analysis.py Bolie              # Specific model, full analysis")
    print("   python complete_analysis.py Bolie -p           # Specific model, parameters only")
    print("   python complete_analysis.py Bolie options_Bolie --parameters-only")
    
    print("\n2. STEP-BY-STEP WORKFLOW:")
    print("   a) First, ensure your model is defined in models/YourModel.py")
    print("   b) Create options file (optional): custom_options/options_YourModel.py")
    print("   c) Run: python complete_analysis.py YourModel options_YourModel")
    
    print("\n3. FILE STRUCTURE EXPECTED:")
    print("   models/")
    print("   ├── C2M.py                    # Model definition")
    print("   └── YourModel.py              # Your model")
    print("   custom_options/")
    print("   ├── options_C2M.py            # Custom options")
    print("   └── options_YourModel.py      # Your options")
    print("   results/                      # Auto-generated")
    print("   ├── obs_ident_matrix_*.txt    # StrikePy output")
    print("   └── detailed_analysis_*.txt   # Our analysis")
    
    print("\n4. TROUBLESHOOTING:")
    print("   • If 'No OIC matrix found': Run StrikePy first")
    print("   • If import errors: Check your model file format")
    print("   • If analysis fails: Check nullspace computation")
    
    print("\n5. INTERPRETING RESULTS:")
    print("   • 'Fully identifiable': All parameters can be estimated")
    print("   • 'Nullspace dimension N': N parameter combinations unidentifiable")
    print("   • 'Identifiable combinations': What you CAN estimate")

# Additional utility functions

def validate_environment():
    """Check if the environment is set up correctly."""
    
    print("Checking environment...")
    
    required_modules = [
        'sympy', 'numpy', 'matplotlib', 'networkx', 
        'symbtools', 'pathlib'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"✗ {module} (missing)")
    
    if missing:
        print(f"\nMissing modules: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    # Check required directories
    dirs = ['models', 'results', 'custom_options', 'functions']
    for d in dirs:
        if Path(d).exists():
            print(f"✓ {d}/ directory")
        else:
            print(f"✗ {d}/ directory (missing)")
            if d == 'results':
                Path(d).mkdir(exist_ok=True)
                print(f"  → Created {d}/ directory")
    
    return True

def create_example_model():
    """Create a simple example model for testing."""
    
    example_code = '''import sympy as sym

# Simple two-state model
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2') 
x = [[x1], [x2]]

# One output (measure first state)
h = [x1]

# One input
u1 = sym.Symbol('u1')
u = [u1]

# No unknown inputs
w = []

# Two parameters
k1 = sym.Symbol('k1')
k2 = sym.Symbol('k2')
p = [[k1], [k2]]

# Simple dynamics: x1' = -k1*x1 + u1, x2' = k2*x1
f = [[-k1*x1 + u1], [k2*x1]]

variables_locales = locals().copy()
'''
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    example_file = models_dir / 'SimpleExample.py'
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    print(f"Created example model: {example_file}")
    
    # Create corresponding options
    options_code = '''import sympy as sym
from math import inf

modelname = 'SimpleExample'
checkObser = 1
maxLietime = inf
nnzDerU = [inf]
nnzDerW = [inf]
prev_ident_pars = []
'''
    
    options_dir = Path('custom_options')
    options_dir.mkdir(exist_ok=True)
    
    options_file = options_dir / 'options_SimpleExample.py'
    with open(options_file, 'w') as f:
        f.write(options_code)
    
    print(f"Created example options: {options_file}")
    print("\nRun with: python complete_analysis.py SimpleExample options_SimpleExample")

def cli_main():
    """Main CLI entry point that handles special commands."""
    import sys
    
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            help_usage()
            sys.exit(0)
        elif sys.argv[1] == '--demo':
            quick_demo()
            sys.exit(0)
        elif sys.argv[1] == '--check':
            validate_environment()
            sys.exit(0)
        elif sys.argv[1] == '--example':
            create_example_model()
            sys.exit(0)
    
    # Run main analysis
    main()
    
if __name__ == "__main__":
    cli_main()