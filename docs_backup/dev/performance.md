# Performance Optimization and Profiling

This guide covers performance optimization techniques, profiling tools, and best practices for making NullStrike analysis efficient and scalable.

## Performance Overview

NullStrike performance depends on several key factors:

1. **Symbolic Computation**: SymPy expression complexity and simplification
2. **Matrix Operations**: Nullspace computation and linear algebra
3. **Memory Management**: Large expression storage and manipulation
4. **I/O Operations**: Model loading, checkpointing, and result saving
5. **Visualization**: Plot generation and rendering

## Profiling Tools and Techniques

### Built-in Performance Monitoring

```python
# Enable performance monitoring in NullStrike
from nullstrike.utils.profiling import PerformanceProfiler, ProfiledAnalysis

class ProfiledAnalysis:
    """Analysis wrapper with built-in performance monitoring."""
    
    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.profiler = PerformanceProfiler()
    
    def run_with_profiling(self):
        """Run analysis with detailed performance tracking."""
        with self.profiler.profile_session("complete_analysis"):
            
            # Phase 1: Model loading and validation
            with self.profiler.profile_section("model_validation"):
                validated_model = self._validate_model()
            
            # Phase 2: STRIKE-GOLDD analysis
            with self.profiler.profile_section("strike_goldd"):
                strike_results = self._run_strike_goldd(validated_model)
            
            # Phase 3: Nullspace analysis
            with self.profiler.profile_section("nullspace_analysis"):
                nullspace_results = self._run_nullspace_analysis(strike_results)
            
            # Phase 4: Visualization
            with self.profiler.profile_section("visualization"):
                visualizations = self._generate_visualizations(nullspace_results)
        
        # Generate performance report
        return self.profiler.generate_report()

# Usage example
profiled_analysis = ProfiledAnalysis(model, options)
performance_report = profiled_analysis.run_with_profiling()

print("Performance Report:")
print("==================")
for section, timing in performance_report.timings.items():
    print(f"{section}: {timing:.2f}s")
```

### Memory Profiling

```python
# Memory usage profiling
import psutil
import gc
from memory_profiler import profile

class MemoryProfiler:
    """Monitor memory usage during analysis."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.peak_memory = 0
    
    def sample_memory(self, label: str = ""):
        """Take a memory usage sample."""
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss
        
        self.memory_samples.append({
            'label': label,
            'timestamp': time.time(),
            'rss': current_memory,
            'vms': memory_info.vms
        })
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        return current_memory
    
    @profile  # Decorator for line-by-line memory profiling
    def analyze_with_memory_tracking(self, model, options):
        """Run analysis with detailed memory tracking."""
        self.sample_memory("start")
        
        # Load model
        validated_model = validate_model(model)
        self.sample_memory("model_loaded")
        
        # STRIKE-GOLDD phase
        observability_matrix = compute_observability_matrix(validated_model)
        self.sample_memory("observability_computed")
        
        # Force garbage collection to see actual usage
        gc.collect()
        self.sample_memory("after_gc")
        
        # Nullspace computation
        nullspace_basis = compute_nullspace(observability_matrix)
        self.sample_memory("nullspace_computed")
        
        return nullspace_basis
    
    def generate_memory_report(self):
        """Generate memory usage report."""
        if not self.memory_samples:
            return "No memory samples collected"
        
        report = ["Memory Usage Report"]
        report.append("===================")
        
        start_memory = self.memory_samples[0]['rss']
        
        for sample in self.memory_samples:
            memory_mb = sample['rss'] / 1024 / 1024
            growth_mb = (sample['rss'] - start_memory) / 1024 / 1024
            report.append(f"{sample['label']}: {memory_mb:.1f} MB (+{growth_mb:.1f} MB)")
        
        report.append(f"Peak memory: {self.peak_memory / 1024 / 1024:.1f} MB")
        
        return "\n".join(report)

# Usage
memory_profiler = MemoryProfiler()
result = memory_profiler.analyze_with_memory_tracking(model, options)
print(memory_profiler.generate_memory_report())
```

### CPU Profiling with cProfile

```python
# CPU profiling for detailed function-level analysis
import cProfile
import pstats
import io
from pstats import SortKey

def profile_analysis(model_name: str, options_file: str = None):
    """Profile NullStrike analysis with cProfile."""
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run analysis under profiler
    profiler.enable()
    try:
        from nullstrike.cli.complete_analysis import main
        result = main(model_name, options_file, parameters_only=True)
    finally:
        profiler.disable()
    
    # Generate detailed report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    
    print("CPU Profiling Report:")
    print("====================")
    print(s.getvalue())
    
    # Identify bottlenecks
    ps.sort_stats(SortKey.TIME)
    print("\nTop Time Consumers:")
    print("===================")
    ps.print_stats(10)
    
    return result

# Run profiling
result = profile_analysis('C2M')
```

### Visualization Performance Profiling

```python
# Profile visualization generation
import matplotlib.pyplot as plt
import time

class VisualizationProfiler:
    """Profile visualization generation performance."""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def profile_visualization_pipeline(self, results, options):
        """Profile complete visualization pipeline."""
        
        visualizations = {}
        
        # 3D Manifold plots
        start_time = time.time()
        try:
            manifold_plots = self._profile_manifold_generation(results, options)
            visualizations['manifolds'] = manifold_plots
            self.timings['manifolds'] = time.time() - start_time
        except Exception as e:
            self.timings['manifolds'] = None
            print(f"Manifold generation failed: {e}")
        
        # 2D Projection plots  
        start_time = time.time()
        try:
            projection_plots = self._profile_projection_generation(results, options)
            visualizations['projections'] = projection_plots
            self.timings['projections'] = time.time() - start_time
        except Exception as e:
            self.timings['projections'] = None
            print(f"Projection generation failed: {e}")
        
        # Graph visualizations
        start_time = time.time()
        try:
            graph_plots = self._profile_graph_generation(results, options)
            visualizations['graphs'] = graph_plots
            self.timings['graphs'] = time.time() - start_time
        except Exception as e:
            self.timings['graphs'] = None
            print(f"Graph generation failed: {e}")
        
        return visualizations
    
    def _profile_manifold_generation(self, results, options):
        """Profile 3D manifold generation."""
        from nullstrike.visualization.manifolds import ManifoldVisualizer
        
        visualizer = ManifoldVisualizer(results, options)
        
        # Profile different aspects
        phases = {}
        
        # Data preparation
        start = time.time()
        data = visualizer.prepare_manifold_data()
        phases['data_prep'] = time.time() - start
        
        # Mesh generation
        start = time.time()
        meshes = visualizer.generate_parameter_meshes(data)
        phases['mesh_generation'] = time.time() - start
        
        # Plot creation
        start = time.time()
        plots = visualizer.create_3d_plots(meshes)
        phases['plot_creation'] = time.time() - start
        
        print(f"Manifold phases: {phases}")
        return plots
    
    def generate_performance_recommendations(self):
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if self.timings.get('manifolds', 0) > 60:
            recommendations.append(
                "Consider reducing manifold resolution or parameter ranges"
            )
        
        if self.timings.get('projections', 0) > 30:
            recommendations.append(
                "Limit number of 2D projection pairs"
            )
        
        if sum(t for t in self.timings.values() if t) > 300:
            recommendations.append(
                "Consider using --parameters-only mode for development"
            )
        
        return recommendations
```

## Optimization Strategies

### Symbolic Expression Optimization

```python
# Optimize symbolic expressions for performance
import sympy as sym
from sympy.utilities.lambdify import lambdify

class ExpressionOptimizer:
    """Optimize symbolic expressions for computational efficiency."""
    
    def __init__(self):
        self.optimization_cache = {}
        self.cse_cache = {}
    
    def optimize_expression(self, expr, variables, optimization_level='standard'):
        """Apply various optimization techniques."""
        
        if optimization_level == 'minimal':
            return expr
        elif optimization_level == 'standard':
            return self._standard_optimization(expr, variables)
        elif optimization_level == 'aggressive':
            return self._aggressive_optimization(expr, variables)
        else:
            raise ValueError(f"Unknown optimization level: {optimization_level}")
    
    def _standard_optimization(self, expr, variables):
        """Apply standard optimization techniques."""
        # 1. Expand and collect terms
        expr = sym.expand(expr)
        expr = sym.collect(expr, variables)
        
        # 2. Apply common subexpression elimination
        expr = self._apply_cse(expr)
        
        # 3. Factor if beneficial
        factored = sym.factor(expr)
        if self._count_operations(factored) < self._count_operations(expr):
            expr = factored
        
        return expr
    
    def _aggressive_optimization(self, expr, variables):
        """Apply aggressive optimization (slower but more thorough)."""
        # Start with standard optimization
        expr = self._standard_optimization(expr, variables)
        
        # Apply trigonometric simplification
        expr = sym.trigsimp(expr)
        
        # Apply full simplification (expensive)
        expr = sym.simplify(expr)
        
        # Try polynomial optimization
        if expr.is_polynomial():
            expr = sym.Poly(expr, variables).as_expr()
        
        return expr
    
    def _apply_cse(self, expr):
        """Apply common subexpression elimination."""
        expr_str = str(expr)
        
        if expr_str in self.cse_cache:
            return self.cse_cache[expr_str]
        
        # Perform CSE
        replacements, simplified = sym.cse(expr)
        
        if simplified:
            result = simplified[0]
            self.cse_cache[expr_str] = result
            return result
        
        return expr
    
    def _count_operations(self, expr):
        """Estimate computational cost of expression."""
        return sym.count_ops(expr)
    
    def create_fast_evaluator(self, expr, variables):
        """Create fast numerical evaluator using lambdify."""
        try:
            # Optimize expression first
            optimized_expr = self.optimize_expression(expr, variables)
            
            # Create fast numerical function
            fast_func = lambdify(variables, optimized_expr, modules=['numpy'])
            
            return fast_func
            
        except Exception as e:
            # Fallback to slower evaluation
            print(f"Fast evaluator creation failed: {e}")
            return lambda *args: float(expr.subs(dict(zip(variables, args))))

# Usage example
optimizer = ExpressionOptimizer()

# Optimize Lie derivatives for faster computation
x, p1, p2 = sym.symbols('x p1 p2')
expr = p1*x**3 + p2*x**2 + p1*p2*x

optimized = optimizer.optimize_expression(expr, [x, p1, p2])
fast_eval = optimizer.create_fast_evaluator(optimized, [x, p1, p2])

# Compare performance
import time
import numpy as np

# Test values
test_x = np.linspace(0, 10, 1000)
test_p1 = 2.0
test_p2 = 3.0

# Symbolic evaluation (slow)
start = time.time()
for x_val in test_x:
    result_symbolic = float(expr.subs([(x, x_val), (p1, test_p1), (p2, test_p2)]))
symbolic_time = time.time() - start

# Fast evaluation
start = time.time()
results_fast = fast_eval(test_x, test_p1, test_p2)
fast_time = time.time() - start

print(f"Symbolic evaluation: {symbolic_time:.3f}s")
print(f"Fast evaluation: {fast_time:.3f}s")
print(f"Speedup: {symbolic_time/fast_time:.1f}x")
```

### Matrix Operation Optimization

```python
# Optimize matrix operations for large systems
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import warnings

class MatrixOptimizer:
    """Optimize matrix operations for large systems."""
    
    def __init__(self):
        self.sparsity_threshold = 0.1  # Use sparse if < 10% non-zero
        self.parallel_threshold = 1000  # Use parallel for matrices > 1000x1000
    
    def optimize_nullspace_computation(self, matrix, tolerance=1e-10):
        """Optimized nullspace computation for large matrices."""
        
        # Convert to appropriate format
        if self._should_use_sparse(matrix):
            return self._sparse_nullspace(matrix, tolerance)
        else:
            return self._dense_nullspace(matrix, tolerance)
    
    def _should_use_sparse(self, matrix):
        """Determine if sparse representation is beneficial."""
        if hasattr(matrix, 'nnz'):  # Already sparse
            return True
        
        # Check sparsity
        if isinstance(matrix, np.ndarray):
            non_zero_ratio = np.count_nonzero(matrix) / matrix.size
            return non_zero_ratio < self.sparsity_threshold
        
        return False
    
    def _sparse_nullspace(self, matrix, tolerance):
        """Compute nullspace using sparse methods."""
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        # Use SVD for sparse matrices
        try:
            # Compute smallest singular values/vectors
            k = min(matrix.shape) - 1
            U, s, Vt = svds(matrix, k=k, which='SM')
            
            # Find null space vectors
            null_mask = s < tolerance
            null_space = Vt[null_mask, :].T
            
            return null_space
            
        except Exception as e:
            print(f"Sparse nullspace computation failed: {e}")
            # Fallback to dense computation
            return self._dense_nullspace(matrix.toarray(), tolerance)
    
    def _dense_nullspace(self, matrix, tolerance):
        """Compute nullspace using dense methods."""
        # Use SVD for numerical stability
        U, s, Vt = np.linalg.svd(matrix, full_matrices=True)
        
        # Find null space
        null_mask = s < tolerance
        null_space = Vt[len(s):, :].T
        
        return null_space
    
    def optimize_matrix_rank(self, matrix, tolerance=1e-10):
        """Efficient matrix rank computation."""
        if self._should_use_sparse(matrix):
            return self._sparse_rank(matrix, tolerance)
        else:
            return self._dense_rank(matrix, tolerance)
    
    def _sparse_rank(self, matrix, tolerance):
        """Compute rank for sparse matrices."""
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        # Use sparse SVD
        try:
            k = min(matrix.shape) - 1
            U, s, Vt = svds(matrix, k=k)
            return np.sum(s > tolerance)
        except:
            # Fallback
            return self._dense_rank(matrix.toarray(), tolerance)
    
    def _dense_rank(self, matrix, tolerance):
        """Compute rank for dense matrices."""
        s = np.linalg.svd(matrix, compute_uv=False)
        return np.sum(s > tolerance)

# Parallel matrix operations
class ParallelMatrixOperations:
    """Parallel matrix operations for large systems."""
    
    def __init__(self, n_workers=None):
        import multiprocessing as mp
        self.n_workers = n_workers or mp.cpu_count()
    
    def parallel_nullspace_computation(self, matrix_blocks):
        """Compute nullspace of block matrices in parallel."""
        from multiprocessing import Pool
        
        with Pool(self.n_workers) as pool:
            results = pool.map(self._compute_block_nullspace, matrix_blocks)
        
        # Combine results
        return self._combine_nullspace_results(results)
    
    def _compute_block_nullspace(self, matrix_block):
        """Compute nullspace of single matrix block."""
        optimizer = MatrixOptimizer()
        return optimizer.optimize_nullspace_computation(matrix_block)
    
    def _combine_nullspace_results(self, results):
        """Combine nullspace results from multiple blocks."""
        # This is a simplified combination - real implementation
        # would need sophisticated block matrix nullspace theory
        combined = np.hstack(results)
        
        # Orthogonalize the combined basis
        Q, R = np.linalg.qr(combined)
        return Q
```

### Memory Optimization

```python
# Memory optimization strategies
import gc
import weakref
from typing import Generator, Any

class MemoryOptimizedAnalysis:
    """Memory-efficient analysis for large models."""
    
    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.memory_limit = options.get('memory_limit_gb', 4) * 1024**3
        
    def compute_observability_streaming(self) -> Generator[np.ndarray, None, None]:
        """Compute observability matrix in streaming fashion."""
        
        for lie_order in range(self.options.max_lie_order + 1):
            # Compute one row at a time
            lie_derivatives = self._compute_lie_derivatives_order(lie_order)
            
            for i, output in enumerate(self.model.outputs):
                for j, param in enumerate(self.model.parameters):
                    # Compute single matrix entry
                    entry = sym.diff(lie_derivatives[i], param)
                    yield (lie_order * len(self.model.outputs) + i, j, entry)
                    
                    # Force garbage collection periodically
                    if (i * len(self.model.parameters) + j) % 100 == 0:
                        gc.collect()
    
    def chunk_based_nullspace(self, matrix, chunk_size=1000):
        """Compute nullspace using chunk-based processing."""
        n_rows, n_cols = matrix.shape
        
        if n_rows <= chunk_size:
            # Small enough to process normally
            return self._standard_nullspace(matrix)
        
        # Process in chunks
        chunk_results = []
        for start_row in range(0, n_rows, chunk_size):
            end_row = min(start_row + chunk_size, n_rows)
            chunk = matrix[start_row:end_row, :]
            
            # Process chunk
            chunk_result = self._process_matrix_chunk(chunk)
            chunk_results.append(chunk_result)
            
            # Clean up
            del chunk
            gc.collect()
        
        # Combine chunk results
        return self._combine_chunk_results(chunk_results)
    
    def _process_matrix_chunk(self, chunk):
        """Process a single matrix chunk."""
        # Simplified processing - real implementation would
        # maintain mathematical correctness across chunks
        U, s, Vt = np.linalg.svd(chunk, full_matrices=False)
        return {'U': U, 's': s, 'Vt': Vt}
    
    def _combine_chunk_results(self, chunk_results):
        """Combine results from multiple chunks."""
        # This requires sophisticated mathematical techniques
        # for maintaining nullspace properties across chunks
        pass
    
    def memory_efficient_visualization(self, results):
        """Generate visualizations with memory constraints."""
        
        # Check available memory
        available_memory = self._get_available_memory()
        
        if available_memory < self.memory_limit * 0.5:
            # Use low-resolution visualizations
            viz_options = {
                'resolution': 'low',
                'max_points': 1000,
                'use_sampling': True
            }
        else:
            # Use standard resolution
            viz_options = {
                'resolution': 'standard',
                'max_points': 10000,
                'use_sampling': False
            }
        
        return self._generate_visualizations(results, viz_options)
    
    def _get_available_memory(self):
        """Get available system memory."""
        import psutil
        return psutil.virtual_memory().available

# Lazy evaluation for large expressions
class LazyExpressionEvaluator:
    """Lazy evaluation system for large symbolic expressions."""
    
    def __init__(self):
        self.expression_cache = weakref.WeakValueDictionary()
        self.evaluation_cache = {}
    
    def lazy_compute(self, expression_generator, cache_key=None):
        """Compute expressions lazily on demand."""
        
        def lazy_wrapper():
            if cache_key and cache_key in self.evaluation_cache:
                return self.evaluation_cache[cache_key]
            
            result = expression_generator()
            
            if cache_key:
                self.evaluation_cache[cache_key] = result
            
            return result
        
        return lazy_wrapper
    
    def stream_large_computation(self, computation_func, chunk_size=100):
        """Stream large computations to avoid memory overflow."""
        
        def stream_generator():
            chunk_count = 0
            for chunk in computation_func():
                yield chunk
                chunk_count += 1
                
                # Periodic cleanup
                if chunk_count % 10 == 0:
                    gc.collect()
        
        return stream_generator()
```

## Performance Monitoring and Alerts

```python
# Real-time performance monitoring
class PerformanceMonitor:
    """Monitor performance and provide real-time feedback."""
    
    def __init__(self, warning_thresholds=None):
        self.warning_thresholds = warning_thresholds or {
            'memory_gb': 8.0,
            'computation_time_minutes': 10.0,
            'expression_complexity': 10000
        }
        self.alerts = []
    
    def monitor_analysis(self, analysis_function):
        """Decorator to monitor analysis performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                # Run analysis with monitoring
                result = self._run_with_monitoring(analysis_function, *args, **kwargs)
                
                # Check final performance metrics
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                self._check_performance_thresholds(
                    computation_time=end_time - start_time,
                    memory_usage=end_memory,
                    memory_growth=end_memory - start_memory
                )
                
                return result
                
            except Exception as e:
                self.alerts.append(f"Analysis failed: {str(e)}")
                raise
        
        return wrapper
    
    def _run_with_monitoring(self, func, *args, **kwargs):
        """Run function with real-time monitoring."""
        
        # Set up monitoring thread
        import threading
        monitor_thread = threading.Thread(
            target=self._background_monitoring,
            daemon=True
        )
        monitor_thread.start()
        
        try:
            return func(*args, **kwargs)
        finally:
            # Stop monitoring
            self.monitoring_active = False
    
    def _background_monitoring(self):
        """Background monitoring thread."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            time.sleep(30)  # Check every 30 seconds
            
            current_memory = self._get_memory_usage()
            if current_memory > self.warning_thresholds['memory_gb']:
                self.alerts.append(f"High memory usage: {current_memory:.1f} GB")
    
    def _check_performance_thresholds(self, computation_time, memory_usage, memory_growth):
        """Check if performance thresholds are exceeded."""
        
        if computation_time > self.warning_thresholds['computation_time_minutes'] * 60:
            self.alerts.append(
                f"Long computation time: {computation_time/60:.1f} minutes"
            )
        
        if memory_usage > self.warning_thresholds['memory_gb']:
            self.alerts.append(
                f"High memory usage: {memory_usage:.1f} GB"
            )
        
        if memory_growth > 2.0:  # 2 GB growth
            self.alerts.append(
                f"Significant memory growth: {memory_growth:.1f} GB"
            )
    
    def _get_memory_usage(self):
        """Get current memory usage in GB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    def get_performance_summary(self):
        """Get summary of performance issues."""
        if not self.alerts:
            return "No performance issues detected"
        
        return "\n".join([
            "Performance Alerts:",
            "==================="
        ] + [f"• {alert}" for alert in self.alerts])

# Usage example
monitor = PerformanceMonitor()

@monitor.monitor_analysis
def run_monitored_analysis(model_name):
    from nullstrike.cli.complete_analysis import main
    return main(model_name)

# Run with monitoring
result = run_monitored_analysis('C2M')
print(monitor.get_performance_summary())
```

## Configuration for Performance

```python
# Performance-oriented configuration options
class PerformanceOptions:
    """Configuration options optimized for performance."""
    
    @staticmethod
    def fast_development_config():
        """Configuration for fast development iterations."""
        return {
            'max_lie_time': 60,  # 1 minute limit
            'max_lie_order': 3,  # Lower order derivatives
            'generate_visualizations': False,
            'enable_checkpointing': False,
            'simplify_expressions': 'minimal',
            'numerical_tolerance': 1e-8,  # Slightly looser tolerance
            'use_sparse_matrices': True,
            'parallel_computation': True
        }
    
    @staticmethod
    def memory_constrained_config():
        """Configuration for memory-constrained systems."""
        return {
            'max_lie_time': 300,
            'chunk_size': 100,
            'streaming_computation': True,
            'lazy_evaluation': True,
            'visualization_resolution': 'low',
            'cleanup_intermediate': True,
            'memory_limit_gb': 2.0
        }
    
    @staticmethod
    def high_accuracy_config():
        """Configuration prioritizing accuracy over speed."""
        return {
            'max_lie_time': 1800,  # 30 minutes
            'max_lie_order': 10,
            'numerical_tolerance': 1e-12,
            'simplify_expressions': 'aggressive',
            'double_check_results': True,
            'multiple_precision': True
        }
    
    @staticmethod
    def production_config():
        """Balanced configuration for production use."""
        return {
            'max_lie_time': 600,  # 10 minutes
            'max_lie_order': 5,
            'generate_visualizations': True,
            'enable_checkpointing': True,
            'simplify_expressions': 'standard',
            'parallel_computation': True,
            'performance_monitoring': True
        }

# Apply performance configuration
def configure_for_performance(config_type='fast_development'):
    """Apply performance-optimized configuration."""
    config_map = {
        'fast_development': PerformanceOptions.fast_development_config(),
        'memory_constrained': PerformanceOptions.memory_constrained_config(),
        'high_accuracy': PerformanceOptions.high_accuracy_config(),
        'production': PerformanceOptions.production_config()
    }
    
    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return config_map[config_type]

# Usage
performance_config = configure_for_performance('fast_development')
print("Performance Configuration:")
for key, value in performance_config.items():
    print(f"  {key}: {value}")
```

This comprehensive performance optimization guide provides tools and techniques for making NullStrike analysis efficient across different computational constraints and requirements. The combination of profiling tools, optimization strategies, and performance monitoring ensures that users can identify bottlenecks and optimize their analyses effectively.

---

## Next Steps

1. **Profile your specific models** using the provided tools
2. **Identify bottlenecks** in your typical analysis workflows  
3. **Apply appropriate optimizations** based on your constraints
4. **Monitor performance** improvements with the tracking tools
5. **Study [Release Procedures](release.md)** for maintaining performance across versions