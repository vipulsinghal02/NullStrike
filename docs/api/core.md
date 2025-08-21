# Core API Reference

This page documents the core NullStrike modules and functions for programmatic use. The API provides fine-grained control over the analysis process and enables integration into custom workflows.

## Main Analysis Function

### `main()` - Complete Analysis

::: nullstrike.cli.complete_analysis.main

The primary entry point for running complete NullStrike analysis programmatically.

#### Basic Usage

```python
from nullstrike.cli.complete_analysis import main

# Run analysis with default options
result = main('C2M')

# Run with custom options file
result = main('my_model', 'options_my_model')

# Run parameters-only analysis
result = main('my_model', 'options_my_model', parameters_only=True)
```

#### Return Value

The function returns a dictionary containing:

- `'success'`: Boolean indicating if analysis completed successfully
- `'model_name'`: Name of the analyzed model
- `'options_file'`: Options file used
- `'results_path'`: Path to results directory
- `'computation_time'`: Total analysis time in seconds
- `'identifiable_parameters'`: List of identifiable parameter names
- `'unidentifiable_parameters'`: List of unidentifiable parameter names
- `'nullspace_dimension'`: Dimension of the nullspace
- `'rank'`: Rank of the identifiability matrix

#### Example with Error Handling

```python
try:
    result = main('my_model')
    if result['success']:
        print(f"Analysis complete! Results in: {result['results_path']}")
        print(f"Identifiable parameters: {result['identifiable_parameters']}")
    else:
        print("Analysis failed")
except Exception as e:
    print(f"Error during analysis: {e}")
```

## Core Analysis Modules

### STRIKE-GOLDD Implementation

::: nullstrike.core.strike_goldd

The core STRIKE-GOLDD algorithm implementation for structural identifiability analysis.

#### Key Functions

##### `strike_goldd_analysis(model, options)`

Performs the complete STRIKE-GOLDD analysis.

**Parameters:**
- `model`: Model object containing symbolic definitions
- `options`: Configuration object with analysis parameters

**Returns:**
- Dictionary with observability matrix, rank, and identifiability results

##### `compute_lie_derivatives(h, f, x, max_order)`

Computes Lie derivatives up to specified order.

**Parameters:**
- `h`: Output function (SymPy expression)
- `f`: System dynamics (SymPy Matrix)
- `x`: State variables (SymPy Matrix)
- `max_order`: Maximum derivative order

**Returns:**
- List of Lie derivatives

##### `build_observability_matrix(lie_derivatives)`

Constructs the observability matrix from Lie derivatives.

**Parameters:**
- `lie_derivatives`: List of SymPy expressions

**Returns:**
- SymPy Matrix representing observability matrix

#### Usage Example

```python
from nullstrike.core.strike_goldd import strike_goldd_analysis
import sympy as sym

# Define simple model
x1, x2 = sym.symbols('x1 x2')
p1, p2 = sym.symbols('p1 p2')

model = {
    'x': sym.Matrix([[x1], [x2]]),
    'p': sym.Matrix([[p1], [p2]]),
    'f': sym.Matrix([[-p1*x1], [p1*x1 - p2*x2]]),
    'h': sym.Matrix([x1])
}

# Run STRIKE-GOLDD analysis
results = strike_goldd_analysis(model, options)
print(f"Matrix rank: {results['rank']}")
```

### Nullspace Analysis

::: nullstrike.analysis.enhanced_subspace

Enhanced nullspace analysis for finding identifiable parameter combinations.

#### Key Functions

##### `compute_nullspace_analysis(identifiability_matrix)`

Computes complete nullspace analysis.

**Parameters:**
- `identifiability_matrix`: SymPy Matrix (Jacobian of observability matrix)

**Returns:**
- Dictionary containing:
  - `'nullspace_basis'`: Basis vectors for nullspace
  - `'identifiable_basis'`: Basis vectors for identifiable subspace
  - `'nullspace_dimension'`: Dimension of nullspace
  - `'constraints'`: List of parameter constraint equations

##### `find_identifiable_combinations(nullspace_basis, parameter_names)`

Finds identifiable parameter combinations.

**Parameters:**
- `nullspace_basis`: SymPy Matrix with nullspace basis vectors
- `parameter_names`: List of parameter names

**Returns:**
- List of identifiable parameter combination strings

#### Usage Example

```python
from nullstrike.analysis.enhanced_subspace import compute_nullspace_analysis
import sympy as sym

# Compute Jacobian (identifiability matrix)
J = observability_matrix.jacobian(parameters)

# Perform nullspace analysis
nullspace_results = compute_nullspace_analysis(J)

print(f"Nullspace dimension: {nullspace_results['nullspace_dimension']}")
print(f"Parameter constraints: {nullspace_results['constraints']}")
```

### Visualization

::: nullstrike.visualization.manifolds

3D manifold visualization for parameter constraint surfaces.

::: nullstrike.visualization.graphs

Parameter dependency graph visualization.

#### Key Functions

##### `create_manifold_plots(nullspace_basis, options)`

Creates 3D manifold plots showing parameter constraints.

**Parameters:**
- `nullspace_basis`: Nullspace basis vectors
- `options`: Plotting configuration

**Returns:**
- Dictionary of matplotlib Figure objects

##### `create_parameter_graph(identifiability_results)`

Creates network graph of parameter dependencies.

**Parameters:**
- `identifiability_results`: Results from identifiability analysis

**Returns:**
- NetworkX graph object and matplotlib Figure

#### Usage Example

```python
from nullstrike.visualization.manifolds import create_manifold_plots
from nullstrike.visualization.graphs import create_parameter_graph

# Create 3D manifold visualizations
manifold_plots = create_manifold_plots(nullspace_basis, plot_options)

# Create parameter dependency graph
graph, fig = create_parameter_graph(identifiability_results)

# Save plots
for name, plot in manifold_plots.items():
    plot.savefig(f'manifold_{name}.png')
```

## Configuration and Model Loading

### Model Utilities

The `nullstrike.models` module provides utilities for loading and validating model definitions from the `custom_models/` directory.

#### Key Functions

##### `load_model(model_name)`

Loads a model from the custom_models directory.

**Parameters:**
- `model_name`: String name of model file (without .py extension)

**Returns:**
- Model object with symbolic definitions

**Available Models:**
- `Bolie` - Bolie's model for glucose-insulin dynamics
- `C2M` - Two-compartment pharmacokinetic model
- `calibration_single` - Single-parameter calibration model
- `1A_integral`, `1B_prop_integral`, `1C_nonlinear` - Control system examples

#### Usage Example

```python
from nullstrike.models import load_model

# Load model
model = load_model('C2M')

# Access model components
print(f"State variables: {model.x}")
print(f"Parameters: {model.p}")
print(f"Outputs: {model.h}")
print(f"Dynamics: {model.f}")
```

#### Dynamic Model Access

Models can also be accessed as attributes:

```python
from nullstrike.models import Bolie, C2M

# Direct access to models
bolie_model = Bolie
c2m_model = C2M
```

### Configuration Management

::: nullstrike.configs.default_options

Default configuration options and utilities.

#### Key Functions

##### `load_options(options_file)`

Loads configuration from options file.

**Parameters:**
- `options_file`: String name of options file

**Returns:**
- Configuration object

##### `get_default_options()`

Returns default configuration options.

**Returns:**
- Default configuration dictionary

## Checkpointing System

::: nullstrike.analysis.checkpointing

Intelligent caching system for expensive computations.

#### Key Functions

##### `load_checkpoint(model_name, options_hash)`

Loads cached results if available and valid.

**Parameters:**
- `model_name`: String name of model
- `options_hash`: Hash of options configuration

**Returns:**
- Cached results dictionary or None if not available

##### `save_checkpoint(model_name, options_hash, results)`

Saves analysis results for future use.

**Parameters:**
- `model_name`: String name of model
- `options_hash`: Hash of options configuration
- `results`: Results dictionary to cache

#### Usage Example

```python
from nullstrike.analysis.checkpointing import load_checkpoint, save_checkpoint

# Try to load cached results
cached_results = load_checkpoint('my_model', options_hash)

if cached_results is None:
    # Run analysis
    results = run_analysis(model, options)
    
    # Save for future use
    save_checkpoint('my_model', options_hash, results)
else:
    print("Using cached results")
    results = cached_results
```

## Utility Functions

### Mathematical Utilities

::: nullstrike.utils

General utility functions for symbolic computation and analysis.

#### Key Functions

##### `compute_model_hash(model, options)`

Computes hash for model and options combination.

**Parameters:**
- `model`: Model object
- `options`: Options object

**Returns:**
- String hash for caching purposes

##### `simplify_expressions(expr_list)`

Simplifies list of symbolic expressions.

**Parameters:**
- `expr_list`: List of SymPy expressions

**Returns:**
- List of simplified expressions

## Error Handling

### Exception Classes

```python
class NullStrikeError(Exception):
    """Base exception for NullStrike errors."""
    pass

class ModelLoadError(NullStrikeError):
    """Raised when model loading fails."""
    pass

class AnalysisError(NullStrikeError):
    """Raised when analysis computation fails."""
    pass

class VisualizationError(NullStrikeError):
    """Raised when visualization generation fails."""
    pass
```

### Exception Handling Example

```python
from nullstrike.cli.complete_analysis import main
from nullstrike.core.exceptions import ModelLoadError, AnalysisError

try:
    result = main('my_model')
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage Patterns

### Custom Analysis Pipeline

```python
from nullstrike.core.strike_goldd import strike_goldd_analysis
from nullstrike.analysis.enhanced_subspace import compute_nullspace_analysis
from nullstrike.visualization.manifolds import create_manifold_plots

def custom_analysis_pipeline(model_name, custom_options):
    """Custom analysis with specific modifications."""
    
    # Load model with custom preprocessing
    model = load_and_preprocess_model(model_name)
    
    # Run core analysis
    strike_results = strike_goldd_analysis(model, custom_options)
    
    # Enhanced nullspace analysis
    nullspace_results = compute_nullspace_analysis(
        strike_results['identifiability_matrix']
    )
    
    # Custom visualization
    plots = create_manifold_plots(
        nullspace_results['nullspace_basis'], 
        custom_options
    )
    
    return {
        'strike_goldd': strike_results,
        'nullspace': nullspace_results,
        'visualizations': plots
    }
```

### Batch Processing

```python
def batch_analysis(model_list, base_options):
    """Analyze multiple models with consistent options."""
    
    results = {}
    
    for model_name in model_list:
        try:
            print(f"Analyzing {model_name}...")
            result = main(model_name, base_options)
            results[model_name] = result
        except Exception as e:
            print(f"Failed to analyze {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

# Usage
models = ['C2M', 'Bolie', 'calibration_single']
batch_results = batch_analysis(models, 'options_default')
```

### Integration with Parameter Estimation

```python
def integrate_with_fitting(model_name, data, initial_guess):
    """Use identifiability results to guide parameter estimation."""
    
    # Run identifiability analysis
    id_results = main(model_name, parameters_only=True)
    
    # Extract identifiable combinations
    identifiable_params = id_results['identifiable_parameters']
    
    # Set up constrained optimization
    def objective(params):
        # Only fit identifiable combinations
        return compute_fit_error(params, data, identifiable_params)
    
    # Run optimization
    fitted_params = optimize(objective, initial_guess)
    
    return fitted_params, id_results
```

## Development and Testing

### Model Development Utilities

```python
def debug_model(model_name):
    """Debug model definition issues."""
    
    try:
        model = load_model(model_name)
        is_valid, messages = validate_model(model)
        
        if is_valid:
            print("Model definition is valid")
        else:
            print("ERROR: Model validation failed:")
            for msg in messages:
                print(f"  - {msg}")
                
        # Test basic analysis
        result = main(model_name, parameters_only=True)
        print(f"Basic analysis successful: {result['rank']} identifiable combinations")
        
    except Exception as e:
        print(f"ERROR: {e}")
```

### Performance Profiling

```python
import time
from nullstrike.cli.complete_analysis import main

def profile_analysis(model_name):
    """Profile analysis performance."""
    
    start_time = time.time()
    
    # Parameters-only timing
    params_start = time.time()
    result = main(model_name, parameters_only=True)
    params_time = time.time() - params_start
    
    # Full analysis timing
    full_start = time.time()
    result = main(model_name)
    full_time = time.time() - full_start
    
    total_time = time.time() - start_time
    
    print(f"Performance profile for {model_name}:")
    print(f"  Parameters-only: {params_time:.2f}s")
    print(f"  Full analysis: {full_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    
    return {
        'parameters_time': params_time,
        'full_time': full_time,
        'total_time': total_time
    }
```

---

## Further Reading

- **[User Guide](../guide/cli-usage.md)**: Command-line interface usage
- **[Examples](../examples/simple.md)**: Practical examples and tutorials
- **[Architecture](../dev/architecture.md)**: System design and components

!!! note "API Stability"
    
    The NullStrike API is designed for stability, but some advanced features may change in future versions. Pin to specific versions for production use.