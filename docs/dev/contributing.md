# Contributing to NullStrike

Thank you for your interest in contributing to NullStrike! This guide will help you get started with development, understand the codebase, and make meaningful contributions.

## Quick Start for Contributors

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vipulsinghal02/NullStrike.git
   cd NullStrike
   ```

2. **Create a development environment**:
   ```bash
   # Using conda (recommended)
   conda create -n nullstrike-dev python=3.9
   conda activate nullstrike-dev
   
   # Or using venv
   python -m venv nullstrike-dev
   source nullstrike-dev/bin/activate  # Linux/macOS
   # nullstrike-dev\Scripts\activate   # Windows
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev,docs]"
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   nullstrike C2M --parameters-only
   pytest tests/
   ```

## Development Workflow

### Branch Strategy

We use a simplified Git workflow:

- **`main`**: Stable release branch
- **Feature branches**: `feature/description` or `fix/issue-description`
- **Documentation**: `docs/description`

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/add-new-algorithm
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Run specific test
   pytest tests/test_analysis.py::test_nullspace_computation
   
   # Test with examples
   nullstrike C2M --parameters-only
   nullstrike Bolie --parameters-only
   ```

4. **Format and lint**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add: new nullspace optimization algorithm"
   git push origin feature/add-new-algorithm
   ```

6. **Create a pull request** with a clear description

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use Black formatting (line length: 88)
black src/ tests/

# Type hints are required for new code
def compute_nullspace(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Compute nullspace of a matrix.
    
    Args:
        matrix: Input matrix for nullspace computation
        tolerance: Numerical tolerance for rank determination
        
    Returns:
        Nullspace basis vectors as columns
        
    Raises:
        ValueError: If matrix is empty or invalid
    """
    pass

# Use descriptive variable names
observability_matrix = compute_observability_matrix(model)
nullspace_basis = compute_nullspace(observability_matrix)

# Prefer explicit over implicit
if result is not None:  # Good
if result:             # Avoid for None checks
```

### Documentation Standards

#### Docstring Format

Use Google style docstrings:

```python
def enhanced_nullspace_analysis(
    model: ModelDefinition, 
    options: AnalysisOptions
) -> NullspaceResults:
    """Perform enhanced nullspace analysis with parameter combinations.
    
    This function extends the basic STRIKE-GOLDD analysis by computing
    the nullspace of the observability matrix and identifying which
    parameter combinations remain identifiable.
    
    Args:
        model: Model definition containing states, parameters, and dynamics
        options: Analysis configuration including tolerances and limits
        
    Returns:
        NullspaceResults containing:
            - nullspace_basis: Basis vectors for unidentifiable directions
            - identifiable_combinations: List of identifiable parameter combinations
            - observability_rank: Rank of the observability matrix
            
    Raises:
        ComputationError: If symbolic computation fails or times out
        ValidationError: If model definition is invalid
        
    Example:
        ```python
        from nullstrike.core import load_model
        from nullstrike.analysis import enhanced_nullspace_analysis
        
        model = load_model('C2M')
        options = AnalysisOptions(max_lie_time=300)
        results = enhanced_nullspace_analysis(model, options)
        
        print(f"Found {len(results.identifiable_combinations)} combinations")
        ```
        
    Note:
        Large models may require increased time limits in options.
        The algorithm complexity is approximately O(n³) where n is
        the number of parameters.
    """
    pass
```

#### Mathematical Documentation

For mathematical algorithms, include LaTeX:

```python
def lie_derivative_matrix(h, f, x, order):
    """Compute Lie derivative matrix for observability analysis.
    
    Computes the matrix of Lie derivatives:
    
    $$\\mathcal{L} = \\begin{bmatrix}
    \\mathcal{L}_f^0 h \\\\
    \\mathcal{L}_f^1 h \\\\
    \\vdots \\\\
    \\mathcal{L}_f^n h
    \\end{bmatrix}$$
    
    Where $\\mathcal{L}_f^k h$ is the k-th Lie derivative of output h
    along vector field f.
    
    The observability matrix rank determines structural identifiability:
    - rank($\\mathcal{L}$) = n_params: All parameters identifiable
    - rank($\\mathcal{L}$) < n_params: Some parameters unidentifiable
    
    Args:
        h: Output functions as SymPy expressions
        f: State dynamics as SymPy expressions  
        x: State variables
        order: Maximum Lie derivative order
        
    Returns:
        Matrix of Lie derivatives up to specified order
    """
    pass
```

### Testing Standards

#### Test Structure

```python
# tests/test_analysis.py
import pytest
import numpy as np
from nullstrike.analysis import enhanced_nullspace_analysis
from nullstrike.core import ModelDefinition

class TestNullspaceAnalysis:
    """Test suite for nullspace analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simple_model = self._create_simple_model()
        self.complex_model = self._create_complex_model()
    
    def test_nullspace_computation_simple_model(self):
        """Test nullspace computation on a simple identifiable model."""
        result = enhanced_nullspace_analysis(self.simple_model)
        
        assert result.nullspace_basis is not None
        assert result.observability_rank > 0
        assert len(result.identifiable_combinations) >= 0
    
    def test_nullspace_computation_unidentifiable_model(self):
        """Test nullspace computation on model with unidentifiable parameters."""
        result = enhanced_nullspace_analysis(self.complex_model)
        
        # Should find non-trivial nullspace
        assert result.nullspace_basis.shape[1] > 0
        assert result.observability_rank < self.complex_model.n_parameters
    
    @pytest.mark.slow
    def test_large_model_performance(self):
        """Test performance on large models (marked as slow test)."""
        large_model = self._create_large_model(n_states=10, n_params=20)
        
        import time
        start_time = time.time()
        result = enhanced_nullspace_analysis(large_model)
        computation_time = time.time() - start_time
        
        assert computation_time < 300  # 5 minutes max
        assert result is not None
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty model
        with pytest.raises(ValidationError):
            enhanced_nullspace_analysis(None)
        
        # Invalid options
        with pytest.raises(ValidationError):
            enhanced_nullspace_analysis(self.simple_model, options="invalid")
    
    def _create_simple_model(self):
        """Create a simple test model."""
        # Implementation details...
        pass
```

#### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Marked with `@pytest.mark.slow`
- **Regression tests**: Prevent previously fixed bugs

## Contribution Areas

### High-Priority Areas

1. **Algorithm Optimization**
   - Symbolic computation performance
   - Numerical stability improvements
   - Memory usage optimization
   - Parallel computation support

2. **New Features**
   - Additional identifiability methods
   - Enhanced visualization options
   - Better model import/export
   - Integration with external tools

3. **Documentation**
   - Mathematical explanations
   - Tutorial improvements
   - API documentation
   - Example models

4. **Testing**
   - Edge case coverage
   - Performance benchmarks
   - Regression test suite
   - Cross-platform testing

### New Contributor Ideas

Good first contributions for new developers:

1. **Add example models** in `custom_models/`:
   ```python
   # custom_models/enzyme_kinetics.py
   # Simple Michaelis-Menten model
   import sympy as sym
   
   # Define states, parameters, dynamics
   # Well-documented biological model
   ```

2. **Improve error messages**:
   ```python
   # Before
   raise Exception("Computation failed")
   
   # After  
   raise ComputationError(
       f"Lie derivative computation failed for model '{model.name}' "
       f"after {elapsed_time:.1f}s. Try reducing maxLietime in options."
   )
   ```

3. **Add visualization options**:
   ```python
   # Add new plot types, color schemes, or output formats
   def plot_parameter_heatmap(results, **kwargs):
       """Create heatmap showing parameter correlation matrix."""
       pass
   ```

4. **Documentation improvements**:
   - Fix typos and unclear explanations
   - Add missing examples
   - Improve mathematical notation
   - Create tutorial notebooks

## Code Architecture

### Module Structure

```
src/nullstrike/
├── cli/                    # Command-line interface
│   ├── complete_analysis.py    # Main CLI entry point
│   └── __init__.py
├── core/                   # Core STRIKE-GOLDD implementation
│   ├── strike_goldd.py         # Main algorithm
│   ├── functions/              # Helper functions
│   └── __init__.py
├── analysis/               # Enhanced analysis methods
│   ├── enhanced_subspace.py    # Nullspace analysis
│   ├── integrated_analysis.py  # Complete workflow
│   ├── checkpointing.py        # State management
│   └── __init__.py
├── visualization/          # Plotting and visualization
│   ├── manifolds.py            # 3D parameter manifolds
│   ├── graphs.py               # Network visualizations
│   └── __init__.py
├── models/                 # Model utilities
│   └── __init__.py
├── configs/                # Configuration management
│   ├── default_options.py      # Default settings
│   └── __init__.py
└── utils.py               # Shared utilities
```

### Key Design Principles

1. **Separation of Concerns**
   - Core algorithms isolated from UI
   - Visualization separate from computation
   - Configuration management centralized

2. **Extensibility**
   - Plugin architecture for new algorithms
   - Configurable visualization pipeline
   - Modular model definitions

3. **Performance**
   - Lazy evaluation where possible
   - Checkpointing for long computations
   - Memory-efficient data structures

4. **Robustness**
   - Comprehensive error handling
   - Input validation
   - Graceful degradation

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nullstrike --cov-report=html

# Run specific test file
pytest tests/test_analysis.py

# Run tests matching pattern
pytest -k "test_nullspace"

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Run with verbose output
pytest -v
```

### Writing Tests

1. **Test naming**: Use descriptive names
   ```python
   def test_nullspace_computation_with_zero_rank_matrix():
   def test_lie_derivative_computation_exceeds_time_limit():
   ```

2. **Test isolation**: Each test should be independent
   ```python
   def setup_method(self):
       """Set up fresh state for each test."""
       self.temp_dir = tempfile.mkdtemp()
       self.model = create_test_model()
   
   def teardown_method(self):
       """Clean up after each test."""
       shutil.rmtree(self.temp_dir)
   ```

3. **Fixtures for common setup**:
   ```python
   @pytest.fixture
   def c2m_model():
       """Load the two-compartment model for testing."""
       return load_model('C2M')
   
   def test_analysis_with_c2m(c2m_model):
       result = enhanced_nullspace_analysis(c2m_model)
       assert result is not None
   ```

### Continuous Integration

Our CI pipeline runs:

1. **Lint checks**: Black, flake8, mypy
2. **Unit tests**: Full test suite with coverage
3. **Integration tests**: End-to-end workflows
4. **Documentation build**: Ensure docs compile
5. **Example verification**: Built-in examples work

## Documentation Contributions

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Documentation Standards

1. **Mathematical notation**: Use LaTeX with MathJax
   ```markdown
   The observability matrix is computed as:
   
   $$\mathcal{O} = \begin{bmatrix}
   \mathcal{L}_f^0 h \\
   \mathcal{L}_f^1 h \\
   \vdots
   \end{bmatrix}$$
   ```

2. **Code examples**: Always test code examples
   ```python
   # This code should actually work
   from nullstrike import load_model
   model = load_model('C2M')
   ```

3. **Cross-references**: Link related sections
   ```markdown
   See [Model Definition](../guide/models.md) for details.
   ```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
   ```bash
   pytest
   black src/ tests/
   flake8 src/ tests/
   ```

2. **Update documentation** if needed

3. **Add tests** for new functionality

4. **Update CHANGELOG.md** with your changes

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Verified examples still work

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks**: Must pass CI
2. **Code review**: At least one maintainer approval
3. **Testing**: Manual verification if needed
4. **Documentation**: Check for clarity and completeness

## Release Process

### Versioning

We use semantic versioning (SemVer):

- **MAJOR**: Breaking changes to API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Create release branch**: `release/v0.2.0`
4. **Test thoroughly**: All examples and tests
5. **Tag release**: `git tag v0.2.0`
6. **Build and publish**: To PyPI if applicable

## Getting Help

### Resources

- **Documentation**: [NullStrike Docs](https://vipulsinghal02.github.io/NullStrike/)
- **Issues**: [GitHub Issues](https://github.com/vipulsinghal02/NullStrike/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vipulsinghal02/NullStrike/discussions)

### Communication Channels

1. **GitHub Issues**: Bug reports and feature requests
2. **GitHub Discussions**: Questions and general discussion
3. **Email**: Direct contact for sensitive issues

### Common Questions

**Q: How do I add a new identifiability algorithm?**

A: Create a new module in `src/nullstrike/analysis/` following the existing patterns. See [API Development](api-development.md) for details.

**Q: How do I add support for a new model format?**

A: Extend the model loading system in `src/nullstrike/models/`. Add parsers for your format and update the CLI.

**Q: How do I optimize performance for large models?**

A: See [Performance Optimization](performance.md) for profiling tools and optimization strategies.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

1. **Be respectful**: Treat all community members with respect
2. **Be constructive**: Provide helpful feedback and suggestions
3. **Be patient**: Remember that people have different experience levels
4. **Be collaborative**: Work together to improve the project

### Reporting Issues

If you experience inappropriate behavior, please report it to the maintainers through:

- Private email to project maintainers
- GitHub's reporting features
- Direct message to project leads

## Acknowledgments

Thank you to all contributors who help make NullStrike better! Contributors are recognized in:

- **AUTHORS.md**: All contributors
- **Release notes**: Major contributions
- **Documentation**: Example and tutorial authors

---

## Next Steps

After reading this guide:

1. **Set up your development environment**
2. **Explore the codebase** using the [Architecture Guide](architecture.md)
3. **Look for issues** labeled "good first issue" or "help wanted"
4. **Join the discussion** on GitHub Discussions
5. **Start contributing** with documentation, tests, or small features

Welcome to the NullStrike community!