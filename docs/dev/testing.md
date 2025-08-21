# Testing Guidelines

This guide covers comprehensive testing strategies for NullStrike, including unit tests, integration tests, performance tests, and continuous integration setup.

## Testing Philosophy

NullStrike testing follows a multi-layered approach:

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test component interactions and workflows
3. **End-to-End Tests**: Test complete analysis pipelines
4. **Performance Tests**: Verify scalability and efficiency
5. **Regression Tests**: Prevent reintroduction of bugs

## Test Organization

### Directory Structure

```
tests/
├── unit/                           # Unit tests for individual modules
│   ├── test_core_strike_goldd.py      # Core STRIKE-GOLDD functionality
│   ├── test_analysis_nullspace.py     # Nullspace analysis
│   ├── test_visualization_manifolds.py # 3D visualization
│   ├── test_visualization_graphs.py   # Graph visualization
│   ├── test_checkpointing.py          # State management
│   └── test_utils.py                  # Utility functions
├── integration/                    # Integration and workflow tests
│   ├── test_complete_workflows.py     # End-to-end analysis
│   ├── test_cli_interface.py          # Command-line interface
│   ├── test_model_loading.py          # Model import/validation
│   └── test_configuration.py          # Configuration management
├── performance/                    # Performance and scalability tests
│   ├── test_large_models.py           # Large model handling
│   ├── test_memory_usage.py           # Memory efficiency
│   ├── benchmark_suite.py             # Performance benchmarks
│   └── stress_tests.py                # System stress testing
├── fixtures/                       # Test data and utilities
│   ├── models/                        # Test model definitions
│   │   ├── simple_linear.py           # Basic linear models
│   │   ├── nonlinear_systems.py       # Nonlinear test cases
│   │   └── pathological_cases.py      # Edge cases and error conditions
│   ├── expected_results/              # Known correct results
│   │   ├── c2m_analysis.json          # Two-compartment model results
│   │   └── bolie_analysis.json        # Bolie model results
│   └── test_helpers.py                # Shared testing utilities
└── conftest.py                     # Pytest configuration and fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage reporting
pytest --cov=nullstrike --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_analysis_nullspace.py

# Run specific test function
pytest tests/unit/test_analysis_nullspace.py::test_nullspace_computation

# Run tests matching pattern
pytest -k "nullspace"

# Run with verbose output
pytest -v

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

### Test Categories

```bash
# Run only fast tests (exclude slow/performance tests)
pytest -m "not slow"

# Run only integration tests
pytest tests/integration/

# Run only performance tests
pytest tests/performance/

# Run tests for specific component
pytest tests/unit/test_core_strike_goldd.py

# Run with specific log level
pytest --log-level=DEBUG
```

### Continuous Integration

```bash
# Full CI test suite (what runs on GitHub Actions)
./scripts/run_ci_tests.sh

# This includes:
# - Linting (black, flake8, mypy)
# - Unit tests with coverage
# - Integration tests
# - Documentation build
# - Example verification
```

## Unit Testing

### Test Structure and Patterns

#### Basic Test Class Structure

```python
# tests/unit/test_analysis_nullspace.py
import pytest
import numpy as np
import sympy as sym
from unittest.mock import Mock, patch

from nullstrike.analysis.enhanced_subspace import NullspaceAnalyzer
from nullstrike.core.strike_goldd import STRIKEGOLDDAnalyzer
from tests.fixtures.test_helpers import create_test_model, assert_matrices_equal

class TestNullspaceAnalyzer:
    """Test suite for nullspace analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simple_model = create_test_model('simple_linear')
        self.complex_model = create_test_model('two_compartment')
        self.tolerance = 1e-10
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up any temporary files or state
        pass
    
    def test_nullspace_computation_identifiable_model(self):
        """Test nullspace computation on fully identifiable model."""
        # Arrange
        analyzer = NullspaceAnalyzer(self.simple_model)
        
        # Act
        result = analyzer.compute_nullspace()
        
        # Assert
        assert result.nullspace_basis is not None
        assert result.nullspace_basis.shape[1] == 0  # No nullspace for identifiable model
        assert result.observability_rank == self.simple_model.n_parameters
    
    def test_nullspace_computation_unidentifiable_model(self):
        """Test nullspace computation on model with unidentifiable parameters."""
        # Arrange
        analyzer = NullspaceAnalyzer(self.complex_model)
        
        # Act
        result = analyzer.compute_nullspace()
        
        # Assert
        assert result.nullspace_basis is not None
        assert result.nullspace_basis.shape[1] > 0  # Non-trivial nullspace
        assert result.observability_rank < self.complex_model.n_parameters
        
        # Verify nullspace property: O * N = 0
        product = result.observability_matrix @ result.nullspace_basis
        assert_matrices_equal(product, np.zeros_like(product), self.tolerance)
```

#### Testing Symbolic Computation

```python
def test_lie_derivative_computation(self):
    """Test Lie derivative computation with symbolic expressions."""
    # Arrange
    x1, x2 = sym.symbols('x1 x2')
    p1, p2 = sym.symbols('p1 p2')
    
    # Simple dynamics: dx/dt = p1*x1, dy/dt = p2*x2  
    f = [p1*x1, p2*x2]
    h = [x1 + x2]  # Output: sum of states
    
    analyzer = STRIKEGOLDDAnalyzer(f, h, [x1, x2], [p1, p2])
    
    # Act
    lie_derivatives = analyzer.compute_lie_derivatives(max_order=2)
    
    # Assert
    expected_0th = x1 + x2
    expected_1st = p1*x1 + p2*x2
    expected_2nd = p1**2*x1 + p2**2*x2
    
    assert lie_derivatives[0] == expected_0th
    assert lie_derivatives[1] == expected_1st  
    assert lie_derivatives[2] == expected_2nd

def test_observability_matrix_construction(self):
    """Test observability matrix construction from Lie derivatives."""
    # Arrange
    model = create_test_model('simple_2param')
    analyzer = STRIKEGOLDDAnalyzer(model)
    
    # Act
    obs_matrix = analyzer.compute_observability_matrix()
    
    # Assert
    expected_shape = (3, 2)  # 3 Lie derivatives, 2 parameters
    assert obs_matrix.shape == expected_shape
    
    # Check that matrix entries are correct symbolic expressions
    # (This requires knowing the expected analytical form)
    assert obs_matrix[0, 0] == 1  # ∂h/∂p1
    assert obs_matrix[0, 1] == 0  # ∂h/∂p2
```

#### Error Handling Tests

```python
def test_invalid_model_definition(self):
    """Test error handling for invalid model definitions."""
    with pytest.raises(ModelDefinitionError, match="States cannot be empty"):
        NullspaceAnalyzer(states=[], parameters=['p1'], dynamics=['p1'], outputs=['x1'])

def test_computation_timeout(self):
    """Test behavior when computation exceeds time limit."""
    # Arrange
    large_model = create_test_model('large_symbolic')
    analyzer = NullspaceAnalyzer(large_model)
    analyzer.options.max_lie_time = 0.1  # Very short timeout
    
    # Act & Assert
    with pytest.raises(ComputationError, match="Computation timed out"):
        analyzer.compute_observability_matrix()

def test_numerical_instability_detection(self):
    """Test detection of numerically unstable computations."""
    # Arrange
    model = create_test_model('ill_conditioned')
    analyzer = NullspaceAnalyzer(model)
    
    # Act
    with pytest.warns(UserWarning, match="Numerical instability detected"):
        result = analyzer.compute_nullspace()
    
    # Assert
    assert result.numerical_issues is True
    assert 'ill_conditioned' in result.warnings
```

### Parameterized Tests

```python
@pytest.mark.parametrize("model_name,expected_nullspace_dim", [
    ('simple_linear', 0),
    ('two_compartment', 2), 
    ('bolie_model', 1),
    ('calibration_double', 3)
])
def test_nullspace_dimensions(model_name, expected_nullspace_dim):
    """Test nullspace dimensions for various models."""
    # Arrange
    model = create_test_model(model_name)
    analyzer = NullspaceAnalyzer(model)
    
    # Act
    result = analyzer.compute_nullspace()
    
    # Assert
    actual_dim = result.nullspace_basis.shape[1]
    assert actual_dim == expected_nullspace_dim, \
        f"Model {model_name}: expected dim {expected_nullspace_dim}, got {actual_dim}"

@pytest.mark.parametrize("matrix_size,expected_performance", [
    (10, 1.0),    # Small matrix: < 1 second
    (50, 5.0),    # Medium matrix: < 5 seconds  
    (100, 30.0),  # Large matrix: < 30 seconds
])
def test_performance_scaling(matrix_size, expected_performance):
    """Test performance scaling with matrix size."""
    # Arrange
    model = create_large_test_model(n_params=matrix_size)
    analyzer = NullspaceAnalyzer(model)
    
    # Act
    import time
    start_time = time.time()
    result = analyzer.compute_nullspace()
    elapsed_time = time.time() - start_time
    
    # Assert
    assert elapsed_time < expected_performance, \
        f"Computation took {elapsed_time:.2f}s, expected < {expected_performance}s"
    assert result is not None
```

## Integration Testing

### End-to-End Workflow Tests

```python
# tests/integration/test_complete_workflows.py
import tempfile
import shutil
from pathlib import Path

from nullstrike.cli.complete_analysis import main as cli_main
from nullstrike.analysis.integrated_analysis import AnalysisWorkflow

class TestCompleteWorkflows:
    """Test complete analysis workflows from start to finish."""
    
    def setup_method(self):
        """Set up temporary directories for test outputs."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.temp_dir / "results"
        self.results_dir.mkdir()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
    
    def test_cli_complete_analysis_c2m(self):
        """Test complete CLI analysis workflow with C2M model."""
        # Act
        result = cli_main('C2M', parameters_only=False)
        
        # Assert
        assert result.status == 'success'
        assert result.strike_goldd_results is not None
        assert result.nullspace_results is not None
        assert result.visualizations is not None
        
        # Verify expected outputs exist
        results_path = Path('results/C2M')
        assert (results_path / 'analysis_report.txt').exists()
        assert (results_path / 'nullspace_analysis.txt').exists()
        assert (results_path / 'visualizations').exists()
    
    def test_python_api_workflow(self):
        """Test complete workflow using Python API."""
        # Arrange
        from nullstrike import load_model, AnalysisOptions
        model = load_model('Bolie')
        options = AnalysisOptions(max_lie_time=120, generate_visualizations=True)
        
        # Act
        workflow = AnalysisWorkflow(model, options)
        result = workflow.run_complete_analysis()
        
        # Assert
        assert result.success is True
        assert len(result.identifiable_combinations) > 0
        assert result.visualizations['manifold_3d'] is not None
        assert result.computation_time > 0
    
    def test_checkpointing_recovery(self):
        """Test analysis recovery from checkpoints."""
        # Arrange
        model = load_model('large_test_model')
        options = AnalysisOptions(enable_checkpointing=True)
        workflow = AnalysisWorkflow(model, options)
        
        # Start analysis and simulate interruption
        workflow.start_analysis()
        workflow.complete_strike_goldd_phase()
        
        # Simulate restart
        new_workflow = AnalysisWorkflow(model, options)
        
        # Act
        result = new_workflow.resume_analysis()
        
        # Assert
        assert result.resumed_from_checkpoint is True
        assert result.resumed_stage == 'nullspace_analysis'
        assert result.success is True
```

### CLI Interface Tests

```python
# tests/integration/test_cli_interface.py
import subprocess
import sys
from pathlib import Path

class TestCLIInterface:
    """Test command-line interface functionality."""
    
    def test_cli_help_message(self):
        """Test CLI help message display."""
        # Act
        result = subprocess.run([sys.executable, '-m', 'nullstrike', '--help'], 
                              capture_output=True, text=True)
        
        # Assert
        assert result.returncode == 0
        assert 'NullStrike' in result.stdout
        assert '--parameters-only' in result.stdout
        assert 'model_name' in result.stdout
    
    def test_cli_version_display(self):
        """Test CLI version information."""
        # Act
        result = subprocess.run([sys.executable, '-m', 'nullstrike', '--version'],
                              capture_output=True, text=True)
        
        # Assert
        assert result.returncode == 0
        assert 'NullStrike' in result.stdout
        # Should match version in pyproject.toml
    
    def test_cli_parameters_only_mode(self):
        """Test parameters-only CLI mode."""
        # Act
        result = subprocess.run([
            sys.executable, '-m', 'nullstrike', 'C2M', '--parameters-only'
        ], capture_output=True, text=True)
        
        # Assert
        assert result.returncode == 0
        assert 'Analysis complete' in result.stdout
        assert 'Visualization' not in result.stdout  # No viz in parameters-only
    
    def test_cli_invalid_model_error(self):
        """Test CLI error handling for invalid model."""
        # Act
        result = subprocess.run([
            sys.executable, '-m', 'nullstrike', 'nonexistent_model'
        ], capture_output=True, text=True)
        
        # Assert
        assert result.returncode != 0
        assert 'Error' in result.stderr
        assert 'nonexistent_model' in result.stderr
```

## Performance Testing

### Benchmark Suite

```python
# tests/performance/benchmark_suite.py
import time
import psutil
import pytest
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    execution_time: float
    memory_peak: float
    memory_final: float
    success: bool
    details: Dict

class PerformanceBenchmark:
    """Performance benchmark suite for NullStrike."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @pytest.mark.slow
    def test_small_model_performance(self):
        """Benchmark analysis performance on small models."""
        models = ['simple_linear', 'two_compartment', 'enzyme_kinetics']
        
        for model_name in models:
            result = self._benchmark_model_analysis(model_name)
            
            # Performance expectations for small models
            assert result.execution_time < 30.0, \
                f"{model_name} took {result.execution_time:.2f}s (expected < 30s)"
            assert result.memory_peak < 500 * 1024 * 1024, \  # 500 MB
                f"{model_name} used {result.memory_peak / 1024**2:.1f}MB (expected < 500MB)"
    
    @pytest.mark.slow  
    def test_medium_model_performance(self):
        """Benchmark analysis performance on medium-sized models."""
        models = ['metabolic_network', 'signaling_pathway', 'pharmacokinetic_pbpk']
        
        for model_name in models:
            result = self._benchmark_model_analysis(model_name)
            
            # Performance expectations for medium models
            assert result.execution_time < 300.0, \  # 5 minutes
                f"{model_name} took {result.execution_time:.2f}s (expected < 300s)"
            assert result.memory_peak < 2 * 1024 * 1024 * 1024, \  # 2 GB
                f"{model_name} used {result.memory_peak / 1024**3:.1f}GB (expected < 2GB)"
    
    def _benchmark_model_analysis(self, model_name: str) -> BenchmarkResult:
        """Benchmark a single model analysis."""
        # Setup
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Execute
        start_time = time.time()
        try:
            result = cli_main(model_name, parameters_only=True)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        
        end_time = time.time()
        memory_after = process.memory_info().rss
        
        # Collect results
        benchmark_result = BenchmarkResult(
            test_name=f"analysis_{model_name}",
            execution_time=end_time - start_time,
            memory_peak=memory_after,
            memory_final=memory_after,
            success=success,
            details={'model': model_name, 'result': str(result)[:100]}
        )
        
        self.results.append(benchmark_result)
        return benchmark_result

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated analyses."""
        model_name = 'simple_linear'
        n_iterations = 10
        memory_samples = []
        
        for i in range(n_iterations):
            # Run analysis
            cli_main(model_name, parameters_only=True)
            
            # Sample memory usage
            process = psutil.Process()
            memory_samples.append(process.memory_info().rss)
            
            # Clean up any caches
            import gc
            gc.collect()
        
        # Check for memory growth trend
        memory_growth = memory_samples[-1] - memory_samples[0]
        growth_per_iteration = memory_growth / n_iterations
        
        # Allow some growth but flag significant leaks
        max_growth_per_iteration = 10 * 1024 * 1024  # 10 MB per iteration
        assert growth_per_iteration < max_growth_per_iteration, \
            f"Potential memory leak: {growth_per_iteration / 1024**2:.1f}MB growth per iteration"
```

### Stress Testing

```python
# tests/performance/stress_tests.py
import threading
import concurrent.futures
from pathlib import Path

class StressTests:
    """Stress tests for NullStrike system stability."""
    
    def test_concurrent_analyses(self):
        """Test multiple concurrent analyses."""
        models = ['C2M', 'Bolie', 'simple_linear', 'calibration_single']
        n_workers = 4
        
        def run_analysis(model_name):
            """Run single analysis in thread."""
            try:
                result = cli_main(model_name, parameters_only=True)
                return (model_name, True, result)
            except Exception as e:
                return (model_name, False, str(e))
        
        # Execute concurrent analyses
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(run_analysis, model) for model in models]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Check results
        successes = [r for r in results if r[1]]
        failures = [r for r in results if not r[1]]
        
        assert len(successes) >= len(models) * 0.8, \
            f"Too many failures in concurrent test: {len(failures)} failures"
    
    def test_large_model_handling(self):
        """Test handling of artificially large models."""
        # Create progressively larger test models
        for n_params in [20, 50, 100]:
            model = self._create_large_test_model(n_params=n_params)
            
            # Test with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Analysis of {n_params}-parameter model exceeded timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minute timeout
            
            try:
                result = NullspaceAnalyzer(model).compute_nullspace()
                assert result is not None
                print(f"✓ Handled {n_params}-parameter model successfully")
            except TimeoutError:
                print(f"⚠ {n_params}-parameter model exceeded timeout (expected for large models)")
            finally:
                signal.alarm(0)  # Cancel timeout
    
    def _create_large_test_model(self, n_params: int):
        """Create large test model with specified number of parameters."""
        import sympy as sym
        
        # Create many parameters and states
        params = [sym.Symbol(f'p{i}') for i in range(n_params)]
        states = [sym.Symbol(f'x{i}') for i in range(min(n_params, 10))]
        
        # Create complex dynamics (polynomial in parameters)
        dynamics = []
        for i, x in enumerate(states):
            # Each state depends on multiple parameters
            expr = sum(params[j] * x * sym.sin(j*x) for j in range(min(5, n_params)))
            dynamics.append(expr)
        
        # Simple output
        outputs = [sum(states)]
        
        return ModelDefinition(states, params, dynamics, outputs)
```

## Test Fixtures and Utilities

### Shared Test Fixtures

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_results_dir():
    """Temporary directory for test results."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def c2m_model():
    """Two-compartment pharmacokinetic model for testing."""
    from tests.fixtures.models.simple_linear import create_c2m_model
    return create_c2m_model()

@pytest.fixture
def analysis_options():
    """Standard analysis options for testing."""
    from nullstrike.configs.default_options import AnalysisOptions
    return AnalysisOptions(
        max_lie_time=60,
        generate_visualizations=False,  # Faster tests
        enable_checkpointing=False,
        numerical_tolerance=1e-10
    )

@pytest.fixture(scope="session")
def expected_results():
    """Load expected results for regression testing."""
    import json
    expected_file = Path(__file__).parent / "fixtures" / "expected_results" / "all_models.json"
    with open(expected_file) as f:
        return json.load(f)
```

### Test Helper Functions

```python
# tests/fixtures/test_helpers.py
import numpy as np
import sympy as sym
from typing import List, Dict, Any, Optional

def create_test_model(model_type: str) -> 'ModelDefinition':
    """Factory function to create test models."""
    if model_type == 'simple_linear':
        return _create_simple_linear_model()
    elif model_type == 'two_compartment':
        return _create_two_compartment_model()
    elif model_type == 'unidentifiable':
        return _create_unidentifiable_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def assert_matrices_equal(actual: np.ndarray, expected: np.ndarray, 
                         tolerance: float = 1e-10, 
                         msg: Optional[str] = None):
    """Assert two matrices are equal within tolerance."""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    
    if max_diff > tolerance:
        error_msg = f"Matrices differ by {max_diff:.2e} (tolerance: {tolerance:.2e})"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)

def assert_symbolic_equal(expr1: sym.Expr, expr2: sym.Expr, 
                         simplify: bool = True):
    """Assert two symbolic expressions are mathematically equal."""
    if simplify:
        diff = sym.simplify(expr1 - expr2)
    else:
        diff = expr1 - expr2
    
    if diff != 0:
        raise AssertionError(f"Expressions not equal:\n  {expr1}\n  {expr2}\n  Difference: {diff}")

def compare_analysis_results(actual: 'AnalysisResults', 
                           expected: 'AnalysisResults',
                           tolerance: float = 1e-8) -> Dict[str, bool]:
    """Compare two analysis results for regression testing."""
    comparison = {}
    
    # Compare nullspace dimensions
    comparison['nullspace_dimension'] = (
        actual.nullspace_basis.shape[1] == expected.nullspace_basis.shape[1]
    )
    
    # Compare identifiable parameter counts
    comparison['identifiable_count'] = (
        len(actual.identifiable_parameters) == len(expected.identifiable_parameters)
    )
    
    # Compare parameter combinations (symbolic)
    comparison['parameter_combinations'] = _compare_parameter_combinations(
        actual.identifiable_combinations, 
        expected.identifiable_combinations
    )
    
    return comparison

def _create_simple_linear_model():
    """Create simple 2-parameter linear model."""
    # States: x1, x2
    x1, x2 = sym.symbols('x1 x2')
    states = [x1, x2]
    
    # Parameters: p1, p2  
    p1, p2 = sym.symbols('p1 p2')
    parameters = [p1, p2]
    
    # Dynamics: dx1/dt = -p1*x1, dx2/dt = -p2*x2
    dynamics = [-p1*x1, -p2*x2]
    
    # Output: y = x1 + x2
    outputs = [x1 + x2]
    
    return ModelDefinition(states, parameters, dynamics, outputs)
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/
    
    - name: Check formatting with black
      run: |
        black --check src/ tests/
    
    - name: Type check with mypy
      run: |
        mypy src/
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=nullstrike --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Test CLI interface
      run: |
        nullstrike C2M --parameters-only
        nullstrike Bolie --parameters-only
    
    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  performance:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m "not stress"
    
    - name: Performance regression check
      run: |
        python scripts/check_performance_regression.py
```

### Test Scripts

```bash
#!/bin/bash
# scripts/run_ci_tests.sh
set -e

echo "=== Running NullStrike Test Suite ==="

echo "1. Code formatting and linting..."
black --check src/ tests/
flake8 src/ tests/
mypy src/

echo "2. Unit tests with coverage..."
pytest tests/unit/ -v --cov=nullstrike --cov-report=term --cov-report=html

echo "3. Integration tests..."
pytest tests/integration/ -v

echo "4. CLI interface verification..."
nullstrike C2M --parameters-only
nullstrike Bolie --parameters-only

echo "5. Documentation build test..."
mkdocs build --strict

echo "6. Example verification..."
python scripts/verify_examples.py

echo "All tests passed! ✓"
```

## Test Data Management

### Expected Results

```python
# scripts/generate_expected_results.py
"""Generate expected results for regression testing."""

import json
from pathlib import Path
from nullstrike.cli.complete_analysis import main as cli_main

def generate_expected_results():
    """Generate expected results for all test models."""
    test_models = ['C2M', 'Bolie', 'calibration_single', 'simple_linear']
    expected_results = {}
    
    for model_name in test_models:
        print(f"Generating expected results for {model_name}...")
        
        result = cli_main(model_name, parameters_only=True)
        
        expected_results[model_name] = {
            'nullspace_dimension': result.nullspace_results.nullspace_basis.shape[1],
            'observability_rank': result.nullspace_results.observability_rank,
            'identifiable_count': len(result.strike_goldd_results.identifiable_parameters),
            'identifiable_combinations': [
                str(combo) for combo in result.nullspace_results.identifiable_combinations
            ],
            'computation_time': result.computation_time
        }
    
    # Save to file
    output_file = Path('tests/fixtures/expected_results/all_models.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(expected_results, f, indent=2)
    
    print(f"Expected results saved to {output_file}")

if __name__ == '__main__':
    generate_expected_results()
```

### Regression Testing

```python
# tests/integration/test_regression.py
"""Regression tests to prevent breaking changes."""

import pytest
import json
from pathlib import Path
from nullstrike.cli.complete_analysis import main as cli_main

class TestRegression:
    """Regression tests against known good results."""
    
    @pytest.fixture(scope="class")
    def expected_results(self):
        """Load expected results from file."""
        expected_file = Path(__file__).parent.parent / "fixtures" / "expected_results" / "all_models.json"
        with open(expected_file) as f:
            return json.load(f)
    
    @pytest.mark.parametrize("model_name", ['C2M', 'Bolie', 'calibration_single'])
    def test_analysis_regression(self, model_name, expected_results):
        """Test that analysis results match expected values."""
        # Run analysis
        result = cli_main(model_name, parameters_only=True)
        
        # Get expected values
        expected = expected_results[model_name]
        
        # Compare key metrics
        assert result.nullspace_results.nullspace_basis.shape[1] == expected['nullspace_dimension']
        assert result.nullspace_results.observability_rank == expected['observability_rank']
        assert len(result.strike_goldd_results.identifiable_parameters) == expected['identifiable_count']
        
        # Compare identifiable combinations (allowing for different ordering)
        actual_combinations = set(str(combo) for combo in result.nullspace_results.identifiable_combinations)
        expected_combinations = set(expected['identifiable_combinations'])
        assert actual_combinations == expected_combinations
    
    def test_performance_regression(self, expected_results):
        """Test that performance hasn't significantly degraded."""
        model_name = 'C2M'  # Use C2M as performance benchmark
        
        # Run analysis with timing
        import time
        start_time = time.time()
        result = cli_main(model_name, parameters_only=True)
        actual_time = time.time() - start_time
        
        # Compare with expected performance (allow 50% degradation)
        expected_time = expected_results[model_name]['computation_time']
        max_allowed_time = expected_time * 1.5
        
        assert actual_time < max_allowed_time, \
            f"Performance regression: {actual_time:.2f}s > {max_allowed_time:.2f}s (expected: {expected_time:.2f}s)"
```

This comprehensive testing framework ensures NullStrike remains reliable, performant, and correct across all supported use cases. The multi-layered approach catches issues at every level, from individual function bugs to system-wide performance problems.

---

## Next Steps

1. **Set up the testing environment** following the installation instructions
2. **Run the existing tests** to understand the current test coverage
3. **Contribute new tests** for any areas lacking coverage
4. **Study [Performance Optimization](performance.md)** for optimization strategies
5. **Review [API Development](api-development.md)** for extension patterns