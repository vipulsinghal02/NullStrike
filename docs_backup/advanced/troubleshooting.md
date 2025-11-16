# Advanced Troubleshooting

This guide covers troubleshooting complex issues in NullStrike, from debugging symbolic computation problems to resolving performance bottlenecks and addressing edge cases in mathematical analysis.

## Diagnostic Tools and Workflows

### 1. Comprehensive Diagnostic System

```python
# advanced_diagnostics.py
"""Advanced diagnostic tools for NullStrike troubleshooting."""

import sys
import time
import traceback
import logging
import psutil
import sympy as sym
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class DiagnosticResult:
    """Results from diagnostic checks."""
    category: str
    test_name: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

class NullStrikeDiagnostics:
    """Comprehensive diagnostic system for NullStrike."""
    
    def __init__(self, model_name: str = None, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup diagnostic logging."""
        logger = logging.getLogger('nullstrike_diagnostics')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def run_full_diagnostics(self) -> List[DiagnosticResult]:
        """Run complete diagnostic suite."""
        
        self.logger.info("Starting NullStrike diagnostics...")
        
        # System diagnostics
        self._check_system_requirements()
        self._check_python_environment()
        self._check_dependencies()
        
        # NullStrike-specific diagnostics
        self._check_installation()
        self._check_model_files()
        
        # Model-specific diagnostics (if model specified)
        if self.model_name:
            self._check_model_definition()
            self._check_model_analysis()
        
        # Performance diagnostics
        self._check_performance_indicators()
        
        # Generate summary
        self._generate_diagnostic_summary()
        
        return self.results
    
    def _check_system_requirements(self):
        """Check system requirements and capabilities."""
        
        # Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self._add_result(
                'system', 'python_version', 'fail',
                f"Python {python_version.major}.{python_version.minor} is not supported",
                recommendations=["Upgrade to Python 3.8 or higher"]
            )
        else:
            self._add_result(
                'system', 'python_version', 'pass',
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb < 4:
            self._add_result(
                'system', 'memory', 'warning',
                f"Low system memory: {memory_gb:.1f} GB",
                recommendations=["Consider using memory optimization options"]
            )
        elif memory_gb < 2:
            self._add_result(
                'system', 'memory', 'fail',
                f"Insufficient memory: {memory_gb:.1f} GB",
                recommendations=["Upgrade system memory to at least 4 GB"]
            )
        else:
            self._add_result(
                'system', 'memory', 'pass',
                f"System memory: {memory_gb:.1f} GB"
            )
        
        # CPU cores
        cpu_count = psutil.cpu_count()
        self._add_result(
            'system', 'cpu_cores', 'pass',
            f"CPU cores: {cpu_count}",
            details={'logical_cores': psutil.cpu_count(logical=True)}
        )
    
    def _check_python_environment(self):
        """Check Python environment configuration."""
        
        # Virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if in_venv:
            self._add_result(
                'environment', 'virtual_env', 'pass',
                "Running in virtual environment"
            )
        else:
            self._add_result(
                'environment', 'virtual_env', 'warning',
                "Not running in virtual environment",
                recommendations=["Consider using virtual environment for isolation"]
            )
        
        # Path configuration
        import nullstrike
        nullstrike_path = Path(nullstrike.__file__).parent
        
        self._add_result(
            'environment', 'nullstrike_path', 'pass',
            f"NullStrike location: {nullstrike_path}"
        )
    
    def _check_dependencies(self):
        """Check critical dependencies."""
        
        critical_deps = {
            'sympy': '1.9',
            'numpy': '1.21.0',
            'matplotlib': '3.5.0',
            'networkx': '2.6'
        }
        
        for package, min_version in critical_deps.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                
                # Simple version comparison (not fully robust)
                if version != 'unknown' and version >= min_version:
                    self._add_result(
                        'dependencies', package, 'pass',
                        f"{package} {version}"
                    )
                else:
                    self._add_result(
                        'dependencies', package, 'warning',
                        f"{package} {version} (minimum: {min_version})",
                        recommendations=[f"Consider upgrading {package}"]
                    )
                    
            except ImportError:
                self._add_result(
                    'dependencies', package, 'fail',
                    f"{package} not found",
                    recommendations=[f"Install {package}: pip install {package}"]
                )
    
    def _check_installation(self):
        """Check NullStrike installation integrity."""
        
        try:
            from nullstrike.cli.complete_analysis import main
            self._add_result(
                'installation', 'cli_import', 'pass',
                "CLI module imports successfully"
            )
        except ImportError as e:
            self._add_result(
                'installation', 'cli_import', 'fail',
                f"CLI import failed: {e}",
                recommendations=["Reinstall NullStrike: pip install -e ."]
            )
        
        try:
            from nullstrike.core.strike_goldd import STRIKEGOLDDAnalyzer
            self._add_result(
                'installation', 'core_import', 'pass',
                "Core module imports successfully"
            )
        except ImportError as e:
            self._add_result(
                'installation', 'core_import', 'fail',
                f"Core import failed: {e}",
                recommendations=["Check installation integrity"]
            )
    
    def _check_model_files(self):
        """Check model files accessibility."""
        
        model_dirs = ['custom_models', 'src/nullstrike/models']
        
        for model_dir in model_dirs:
            model_path = Path(model_dir)
            
            if model_path.exists():
                model_files = list(model_path.glob('*.py'))
                non_init_files = [f for f in model_files if not f.name.startswith('__')]
                
                self._add_result(
                    'models', f'{model_dir}_access', 'pass',
                    f"Found {len(non_init_files)} model files in {model_dir}"
                )
            else:
                self._add_result(
                    'models', f'{model_dir}_access', 'warning',
                    f"Model directory {model_dir} not found"
                )
    
    def _check_model_definition(self):
        """Check specific model definition."""
        if not self.model_name:
            return
        
        try:
            # Try to load the model
            from nullstrike.core.models import load_model
            model = load_model(self.model_name)
            
            self._add_result(
                'model', 'loading', 'pass',
                f"Model {self.model_name} loads successfully"
            )
            
            # Check model structure
            self._validate_model_structure(model)
            
        except Exception as e:
            self._add_result(
                'model', 'loading', 'fail',
                f"Model {self.model_name} failed to load: {e}",
                recommendations=[
                    "Check model file syntax",
                    "Verify all required variables (x, p, f, h) are defined",
                    "Check for SymPy import issues"
                ]
            )
    
    def _validate_model_structure(self, model):
        """Validate mathematical structure of model."""
        
        # Check dimensions
        n_states = len(model.states)
        n_params = len(model.parameters)
        n_outputs = len(model.outputs)
        n_dynamics = len(model.dynamics)
        
        if n_dynamics != n_states:
            self._add_result(
                'model', 'dimensions', 'fail',
                f"Dimension mismatch: {n_states} states but {n_dynamics} dynamics equations",
                recommendations=["Ensure one dynamics equation per state variable"]
            )
        else:
            self._add_result(
                'model', 'dimensions', 'pass',
                f"Model dimensions: {n_states} states, {n_params} parameters, {n_outputs} outputs"
            )
        
        # Check for symbolic expressions
        try:
            for i, dynamic in enumerate(model.dynamics):
                if not isinstance(dynamic, sym.Expr):
                    self._add_result(
                        'model', 'symbolic_dynamics', 'fail',
                        f"Dynamics equation {i} is not a SymPy expression"
                    )
                    return
            
            for i, output in enumerate(model.outputs):
                if not isinstance(output, sym.Expr):
                    self._add_result(
                        'model', 'symbolic_outputs', 'fail',
                        f"Output equation {i} is not a SymPy expression"
                    )
                    return
            
            self._add_result(
                'model', 'symbolic_structure', 'pass',
                "All dynamics and outputs are valid SymPy expressions"
            )
            
        except Exception as e:
            self._add_result(
                'model', 'symbolic_structure', 'fail',
                f"Error validating symbolic structure: {e}"
            )
    
    def _check_model_analysis(self):
        """Check if model can be analyzed successfully."""
        if not self.model_name:
            return
        
        try:
            # Quick analysis test
            from nullstrike.cli.complete_analysis import main
            
            start_time = time.time()
            result = main(self.model_name, parameters_only=True)
            analysis_time = time.time() - start_time
            
            self._add_result(
                'analysis', 'basic_analysis', 'pass',
                f"Basic analysis completed in {analysis_time:.1f}s",
                details={
                    'nullspace_dimension': result.nullspace_results.nullspace_basis.shape[1],
                    'identifiable_count': len(result.strike_goldd_results.identifiable_parameters)
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            recommendations = []
            
            if "timeout" in error_msg.lower():
                recommendations.extend([
                    "Increase maxLietime in options",
                    "Simplify model equations",
                    "Use parameters-only mode"
                ])
            elif "memory" in error_msg.lower():
                recommendations.extend([
                    "Reduce model complexity",
                    "Enable memory optimization",
                    "Use streaming computation"
                ])
            else:
                recommendations.extend([
                    "Check model definition",
                    "Verify symbolic expressions",
                    "Review error details"
                ])
            
            self._add_result(
                'analysis', 'basic_analysis', 'fail',
                f"Analysis failed: {error_msg}",
                recommendations=recommendations
            )
    
    def _check_performance_indicators(self):
        """Check performance-related indicators."""
        
        # Available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 1:
            self._add_result(
                'performance', 'available_memory', 'warning',
                f"Low available memory: {available_gb:.1f} GB",
                recommendations=["Close other applications", "Consider memory optimization"]
            )
        else:
            self._add_result(
                'performance', 'available_memory', 'pass',
                f"Available memory: {available_gb:.1f} GB"
            )
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            self._add_result(
                'performance', 'cpu_usage', 'warning',
                f"High CPU usage: {cpu_percent:.1f}%",
                recommendations=["Wait for CPU usage to decrease", "Close other applications"]
            )
        else:
            self._add_result(
                'performance', 'cpu_usage', 'pass',
                f"CPU usage: {cpu_percent:.1f}%"
            )
    
    def _add_result(self, category: str, test_name: str, status: str, 
                   message: str, details: Dict[str, Any] = None,
                   recommendations: List[str] = None):
        """Add diagnostic result."""
        result = DiagnosticResult(
            category=category,
            test_name=test_name,
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
        self.results.append(result)
        
        # Log result
        log_level = {
            'pass': logging.INFO,
            'warning': logging.WARNING,
            'fail': logging.ERROR
        }.get(status, logging.INFO)
        
        self.logger.log(log_level, f"[{category}] {test_name}: {message}")
    
    def _generate_diagnostic_summary(self):
        """Generate diagnostic summary."""
        
        pass_count = sum(1 for r in self.results if r.status == 'pass')
        warning_count = sum(1 for r in self.results if r.status == 'warning')
        fail_count = sum(1 for r in self.results if r.status == 'fail')
        
        self.logger.info("=" * 50)
        self.logger.info("DIAGNOSTIC SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total tests: {len(self.results)}")
        self.logger.info(f"Passed: {pass_count}")
        self.logger.info(f"Warnings: {warning_count}")
        self.logger.info(f"Failed: {fail_count}")
        
        if fail_count > 0:
            self.logger.error("CRITICAL ISSUES FOUND - Review failed tests")
        elif warning_count > 0:
            self.logger.warning("Some issues detected - Review warnings")
        else:
            self.logger.info("All diagnostics passed!")
    
    def generate_diagnostic_report(self, output_file: str = "diagnostic_report.md"):
        """Generate detailed diagnostic report."""
        
        report = []
        report.append("# NullStrike Diagnostic Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.model_name:
            report.append(f"Model: {self.model_name}")
        report.append("")
        
        # Summary
        pass_count = sum(1 for r in self.results if r.status == 'pass')
        warning_count = sum(1 for r in self.results if r.status == 'warning')
        fail_count = sum(1 for r in self.results if r.status == 'fail')
        
        report.append("## Summary")
        report.append(f"- **Total tests**: {len(self.results)}")
        report.append(f"- **Passed**: {pass_count}")
        report.append(f"- **Warnings**: {warning_count}")
        report.append(f"- **Failed**: {fail_count}")
        report.append("")
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Generate sections for each category
        for category, results in categories.items():
            report.append(f"## {category.title()} Tests")
            report.append("")
            
            for result in results:
                status_icon = {'pass': 'PASS', 'warning': 'WARN', 'fail': 'FAIL'}[result.status]
                report.append(f"### {result.test_name} {status_icon}")
                report.append(f"**Status**: {result.status}")
                report.append(f"**Message**: {result.message}")
                
                if result.details:
                    report.append("**Details**:")
                    for key, value in result.details.items():
                        report.append(f"- {key}: {value}")
                
                if result.recommendations:
                    report.append("**Recommendations**:")
                    for rec in result.recommendations:
                        report.append(f"- {rec}")
                
                report.append("")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"Diagnostic report saved to: {output_file}")

# CLI interface for diagnostics
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NullStrike diagnostic tool")
    parser.add_argument('--model', '-m', help='Specific model to diagnose')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', '-r', default='diagnostic_report.md',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = NullStrikeDiagnostics(args.model, args.verbose)
    results = diagnostics.run_full_diagnostics()
    
    # Generate report
    diagnostics.generate_diagnostic_report(args.report)
    
    # Exit with error code if any failures
    fail_count = sum(1 for r in results if r.status == 'fail')
    sys.exit(fail_count)

if __name__ == '__main__':
    main()
```

## Common Issues and Solutions

### 1. Symbolic Computation Problems

#### Issue: SymPy Expression Complexity

**Symptoms**:
- Analysis hangs during Lie derivative computation
- Memory usage grows rapidly
- Expressions become unmanageably large

**Diagnostic Steps**:
```python
# Check expression complexity
def analyze_expression_complexity(expr):
    """Analyze complexity of symbolic expression."""
    
    complexity_metrics = {
        'total_ops': sym.count_ops(expr),
        'depth': expr.count(sym.Add) + expr.count(sym.Mul),
        'variables': len(expr.free_symbols),
        'functions': len([atom for atom in expr.atoms() if isinstance(atom, sym.Function)]),
        'size': len(str(expr))
    }
    
    print("Expression Complexity Analysis:")
    for metric, value in complexity_metrics.items():
        print(f"  {metric}: {value}")
    
    # Identify problematic patterns
    if complexity_metrics['total_ops'] > 10000:
        print("WARNING: Very complex expression (>10k operations)")
    
    if complexity_metrics['size'] > 100000:
        print("WARNING: Very large expression string (>100k characters)")
    
    return complexity_metrics

# Usage
from your_model import f, h  # Model dynamics and outputs
expr = sym.diff(h[0], x[0])  # Example expression
analyze_expression_complexity(expr)
```

**Solutions**:

1. **Expression Simplification**:
```python
# Aggressive simplification
simplified_expr = sym.simplify(expr)

# Or step-by-step simplification
expr = sym.expand(expr)
expr = sym.collect(expr, variables)
expr = sym.factor(expr)
```

2. **Model Reformulation**:
```python
# Replace complex functions with auxiliary variables
# Instead of: f = [p1*sin(p2*x1 + p3*x2)**2]
# Use: 
aux = sym.Symbol('aux')
f = [p1*aux**2]
# Add constraint: aux = sin(p2*x1 + p3*x2)
```

3. **Numerical Substitution**:
```python
# Replace symbolic parameters with numerical values for testing
numerical_subs = {p1: 1.0, p2: 2.0, p3: 0.5}
simplified_f = [eq.subs(numerical_subs) for eq in f]
```

#### Issue: Infinite or Undefined Expressions

**Symptoms**:
- `zoo` (complex infinity) in results
- `nan` values
- Division by zero errors

**Diagnostic Steps**:
```python
def check_expression_validity(expressions):
    """Check expressions for mathematical validity."""
    
    issues = []
    
    for i, expr in enumerate(expressions):
        # Check for infinities
        if expr.has(sym.oo) or expr.has(sym.zoo):
            issues.append(f"Expression {i} contains infinity: {expr}")
        
        # Check for undefined (NaN)
        if expr.has(sym.nan):
            issues.append(f"Expression {i} contains NaN: {expr}")
        
        # Check for potential division by zero
        denominators = [atom for atom in expr.atoms() if atom.is_Pow and atom.exp < 0]
        for denom in denominators:
            if denom.base.could_extract_minus_sign():
                issues.append(f"Expression {i} has potential division by zero: {denom}")
    
    return issues

# Check model expressions
issues = check_expression_validity(f + h)
for issue in issues:
    print(f"WARNING: {issue}")
```

**Solutions**:

1. **Add Mathematical Constraints**:
```python
# Add assumptions to symbols
p1 = sym.Symbol('p1', positive=True)  # Ensure positive
p2 = sym.Symbol('p2', real=True)      # Ensure real
```

2. **Use Piecewise Functions**:
```python
# Handle edge cases explicitly
safe_expr = sym.Piecewise(
    (expr, sym.Ne(denominator, 0)),
    (0, True)  # Default value
)
```

### 2. Performance and Memory Issues

#### Issue: Out of Memory Errors

**Symptoms**:
- `MemoryError` exceptions
- System becomes unresponsive
- Swap usage increases dramatically

**Diagnostic Steps**:
```python
import psutil

def monitor_memory_usage():
    """Monitor memory usage during analysis."""
    
    process = psutil.Process()
    
    def memory_callback():
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / 1024**3:.2f} GB")
        
        # System memory
        sys_mem = psutil.virtual_memory()
        print(f"System memory: {sys_mem.percent}% used")
        
        if sys_mem.percent > 90:
            print("WARNING: System memory usage very high!")
    
    return memory_callback

# Use during analysis
memory_monitor = monitor_memory_usage()
# Call memory_monitor() periodically during analysis
```

**Solutions**:

1. **Chunked Processing**:
```python
def chunked_nullspace_computation(matrix, chunk_size=1000):
    """Compute nullspace in chunks to manage memory."""
    
    n_rows = matrix.shape[0]
    
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = matrix[start:end, :]
        
        # Process chunk
        chunk_result = process_matrix_chunk(chunk)
        
        # Yield result and clean up
        yield chunk_result
        del chunk
        gc.collect()
```

2. **Streaming Computation**:
```python
def streaming_lie_derivatives(h, f, x, max_order):
    """Compute Lie derivatives in streaming fashion."""
    
    current_h = h.copy()
    
    for order in range(max_order + 1):
        yield order, current_h
        
        # Compute next order
        next_h = []
        for hi in current_h:
            lie_deriv = sum(sym.diff(hi, xi) * fi for xi, fi in zip(x, f))
            next_h.append(lie_deriv)
        
        current_h = next_h
        
        # Memory cleanup
        if order % 5 == 0:  # Cleanup every 5 iterations
            gc.collect()
```

#### Issue: Slow Analysis Performance

**Symptoms**:
- Analysis takes much longer than expected
- CPU usage is low despite computation
- Progress appears to hang

**Diagnostic Steps**:
```python
import cProfile
import pstats

def profile_analysis(model_name):
    """Profile analysis to identify bottlenecks."""
    
    profiler = cProfile.Profile()
    
    profiler.enable()
    try:
        from nullstrike.cli.complete_analysis import main
        result = main(model_name, parameters_only=True)
    finally:
        profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result

# Profile specific model
result = profile_analysis('slow_model')
```

**Solutions**:

1. **Algorithmic Optimization**:
```python
# Use fast evaluation for numerical expressions
from sympy.utilities.lambdify import lambdify

# Convert to fast numerical functions
fast_funcs = [lambdify([x, p], expr, modules='numpy') for expr in f]
```

2. **Parallel Processing**:
```python
from multiprocessing import Pool

def parallel_lie_computation(expressions, variables, max_workers=4):
    """Compute Lie derivatives in parallel."""
    
    def compute_single_derivative(args):
        expr, var = args
        return sym.diff(expr, var)
    
    with Pool(max_workers) as pool:
        tasks = [(expr, var) for expr in expressions for var in variables]
        results = pool.map(compute_single_derivative, tasks)
    
    return results
```

### 3. Model Definition Issues

#### Issue: Model Loading Failures

**Symptoms**:
- `ModuleNotFoundError` or `ImportError`
- Syntax errors in model files
- Missing variable definitions

**Diagnostic Steps**:
```python
def validate_model_file(model_file_path):
    """Validate model file before loading."""
    
    try:
        # Check file exists
        if not Path(model_file_path).exists():
            return False, "Model file does not exist"
        
        # Check syntax
        with open(model_file_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, model_file_path, 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check required variables
        required_vars = ['x', 'p', 'f', 'h']
        missing_vars = []
        
        for var in required_vars:
            if f"{var} =" not in code and f"{var}=" not in code:
                missing_vars.append(var)
        
        if missing_vars:
            return False, f"Missing required variables: {missing_vars}"
        
        return True, "Model file is valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

# Validate before loading
is_valid, message = validate_model_file('custom_models/my_model.py')
print(message)
```

**Solutions**:

1. **Template-Based Model Creation**:
```python
def create_model_template(model_name, n_states, n_params):
    """Create model template file."""
    
    template = f'''# {model_name} model definition
import sympy as sym

# State variables
{', '.join(f'x{i+1}' for i in range(n_states))} = sym.symbols('{' '.join(f'x{i+1}' for i in range(n_states))}')
x = [[{'], ['.join(f'x{i+1}' for i in range(n_states))}]]

# Parameters
{', '.join(f'p{i+1}' for i in range(n_params))} = sym.symbols('{' '.join(f'p{i+1}' for i in range(n_params))}')
p = [[{'], ['.join(f'p{i+1}' for i in range(n_params))}]]

# Dynamics (define your equations here)
f = [
{chr(10).join(f'    [p{i+1}*x{i+1}],  # dx{i+1}/dt' for i in range(n_states))}
]

# Outputs (define your observations here)
h = [
    x1,  # Example: observe first state
]
'''
    
    with open(f'custom_models/{model_name}.py', 'w') as f:
        f.write(template)
    
    print(f"Model template created: custom_models/{model_name}.py")

# Create template
create_model_template('my_new_model', n_states=3, n_params=5)
```

### 4. Numerical Stability Issues

#### Issue: Rank Deficiency and Ill-Conditioning

**Symptoms**:
- Inconsistent nullspace results
- Small changes in parameters cause large result changes
- Warning about numerical precision

**Diagnostic Steps**:
```python
def analyze_matrix_conditioning(matrix):
    """Analyze numerical conditioning of matrix."""
    
    # Convert to numerical array
    if hasattr(matrix, 'evalf'):
        numerical_matrix = np.array(matrix.evalf(), dtype=float)
    else:
        numerical_matrix = np.array(matrix, dtype=float)
    
    # Compute condition number
    try:
        condition_number = np.linalg.cond(numerical_matrix)
        
        print(f"Matrix shape: {numerical_matrix.shape}")
        print(f"Condition number: {condition_number:.2e}")
        
        if condition_number > 1e12:
            print("WARNING: Matrix is ill-conditioned")
        elif condition_number > 1e8:
            print("WARNING: Matrix is poorly conditioned")
        else:
            print("Matrix conditioning is acceptable")
        
        # Singular values
        singular_values = np.linalg.svd(numerical_matrix, compute_uv=False)
        print(f"Smallest singular value: {np.min(singular_values):.2e}")
        print(f"Largest singular value: {np.max(singular_values):.2e}")
        
        return condition_number, singular_values
        
    except np.linalg.LinAlgError as e:
        print(f"Matrix analysis failed: {e}")
        return None, None

# Analyze observability matrix
cond_num, sing_vals = analyze_matrix_conditioning(observability_matrix)
```

**Solutions**:

1. **Regularization**:
```python
def regularized_nullspace(matrix, regularization=1e-12):
    """Compute nullspace with regularization."""
    
    # Add regularization to diagonal
    regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
    
    # Compute nullspace
    U, s, Vt = np.linalg.svd(regularized_matrix)
    
    # Select threshold based on regularization
    threshold = max(regularization * 10, 1e-10)
    null_space = Vt[s < threshold, :].T
    
    return null_space
```

2. **Robust Numerical Methods**:
```python
from scipy.linalg import null_space
from scipy.sparse.linalg import svds

def robust_nullspace_computation(matrix, method='scipy'):
    """Robust nullspace computation with multiple methods."""
    
    if method == 'scipy':
        # Use SciPy's robust null space computation
        null_space_basis = null_space(matrix)
        
    elif method == 'svd_truncated':
        # Use truncated SVD for large matrices
        k = min(matrix.shape) - 1
        U, s, Vt = svds(matrix, k=k, which='SM')
        
        # Find small singular values
        threshold = 1e-10
        null_indices = s < threshold
        null_space_basis = Vt[null_indices, :].T
        
    else:
        # Fallback to standard method
        U, s, Vt = np.linalg.svd(matrix)
        threshold = 1e-10
        null_space_basis = Vt[s < threshold, :].T
    
    return null_space_basis
```

This comprehensive troubleshooting guide provides diagnostic tools and solutions for the most common issues encountered when using NullStrike with complex models and challenging computational scenarios.

---

## Quick Reference

For immediate help with common issues:

1. **Analysis hangs**: Check expression complexity, reduce maxLietime
2. **Memory errors**: Use chunked processing, enable streaming mode  
3. **Model loading fails**: Validate model syntax, check required variables
4. **Inconsistent results**: Check numerical conditioning, use regularization
5. **Performance issues**: Profile analysis, enable parallel processing

Use the diagnostic tool: `python advanced_diagnostics.py --model your_model --verbose`