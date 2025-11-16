# Contributing to NullStrike

Thank you for your interest in contributing to NullStrike! This guide covers development setup, code organization, testing, and contribution guidelines.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/vipulsinghal02/NullStrike.git
cd NullStrike

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Check CLI works
nullstrike --check

# Run tests
pytest

# Check code quality
black --check src/
flake8 src/
mypy src/
```

---

## Project Structure

```
NullStrike/
├── src/nullstrike/           # Main package
│   ├── core/                 # StrikePy engine
│   │   ├── strike_goldd.py
│   │   └── functions/
│   ├── analysis/             # Nullspace analysis
│   │   ├── integrated_analysis.py
│   │   ├── enhanced_subspace.py
│   │   └── checkpointing.py
│   ├── visualization/        # Plots and graphs
│   │   ├── graphs.py
│   │   └── manifolds.py
│   ├── cli/                  # Command-line interface
│   │   └── complete_analysis.py
│   ├── configs/              # Default configuration
│   │   └── default_options.py
│   └── utils.py              # Utilities
│
├── custom_models/            # User model definitions
├── custom_options/           # User configuration files
├── tests/                    # Test suite
├── docs/                     # Documentation
├── results/                  # Generated outputs
├── checkpoints/              # Cached analysis
│
├── pyproject.toml            # Package configuration
├── README.md                 # Main readme
└── LICENSE                   # GPL-3.0 license
```

---

## Code Architecture

### Core Components

**1. StrikePy Engine (`core/`)**
- Original STRIKE-GOLDD algorithm
- Computes observability-identifiability matrix using Lie derivatives
- Symbolic computation with SymPy

**2. Nullspace Analysis (`analysis/`)**
- Enhanced analysis finding identifiable combinations
- Checkpoint system for performance
- Integration with StrikePy

**3. Visualization (`visualization/`)**
- Network graphs (NetworkX + Matplotlib)
- 2D/3D manifold plots
- Constraint surface visualization

**4. CLI (`cli/`)**
- Command-line interface
- 3-method fallback strategy for robustness
- User-friendly output formatting

### Key Design Patterns

**Fallback Strategy:**
```python
try:
    # Method 1: Integrated analysis (preferred)
    run_integrated_analysis()
except:
    try:
        # Method 2: Step-by-step
        run_step_by_step_analysis()
    except:
        # Method 3: Analyze existing results
        analyze_strikepy_results()
```

**Checkpointing for Performance:**
```python
# Hash model + options
model_hash = compute_model_hash(model, options)

# Try to load cached results
checkpoint = load_checkpoint(model_name, options_file, model_hash)

if checkpoint:
    return checkpoint  # Fast path
else:
    results = expensive_computation()
    save_checkpoint(..., results)  # Cache for next time
```

---

## Development Workflow

### Making Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes following code style:**
```python
# Use type hints
def analyze_model(matrix: Matrix, symbols: List[Symbol]) -> Dict[str, Any]:
    """Analyze structural identifiability.

    Args:
        matrix: Observability-identifiability matrix
        symbols: List of parameter symbols

    Returns:
        Dictionary with analysis results
    """
    pass
```

3. **Write tests:**
```python
# tests/test_nullspace.py
def test_nullspace_dimension():
    """Test nullspace dimension computation."""
    O = create_test_matrix()
    results = analyze_identifiable_combinations(O, ...)
    assert results['nullspace_dimension'] == 2
```

4. **Run quality checks:**
```bash
# Format code
black src/

# Check style
flake8 src/

# Type check
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=nullstrike
```

5. **Commit and push:**
```bash
git add .
git commit -m "Add feature: describe what you did"
git push origin feature/your-feature-name
```

6. **Create pull request on GitHub**

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_nullspace.py

# Specific test
pytest tests/test_nullspace.py::test_nullspace_dimension

# With coverage report
pytest --cov=nullstrike --cov-report=html
open htmlcov/index.html
```

### Writing Tests

**Test structure:**
```python
import pytest
from nullstrike.analysis import analyze_identifiable_combinations

def test_fully_identifiable_model():
    """Test that fully identifiable model returns nullspace_dim=0."""
    # Arrange
    O = create_fully_identifiable_matrix()
    params = [sym.Symbol('p1'), sym.Symbol('p2')]

    # Act
    results = analyze_identifiable_combinations(O, params, [], [])

    # Assert
    assert results['fully_identifiable'] == True
    assert results['nullspace_dimension'] == 0
```

**Test fixtures:**
```python
@pytest.fixture
def sample_matrix():
    """Create a sample OIC matrix for testing."""
    return Matrix([[1, 0], [0, 1]])

def test_with_fixture(sample_matrix):
    """Test using fixture."""
    assert sample_matrix.rank() == 2
```

---

## Code Style Guidelines

### Python Style

**Follow PEP 8 with these specifics:**

- Line length: 100 characters max
- Use type hints
- Docstrings: Google style
- Imports: Organized (stdlib, third-party, local)

**Example:**
```python
from typing import List, Dict, Optional
import sympy as sym
from sympy import Matrix

from ..core import strike_goldd
from ..utils import get_results_dir


def analyze_matrix(
    oic_matrix: Matrix,
    param_symbols: List[sym.Symbol],
    state_symbols: Optional[List[sym.Symbol]] = None
) -> Dict[str, Any]:
    """Analyze observability-identifiability matrix.

    Computes nullspace and identifies parameter combinations.

    Args:
        oic_matrix: The observability-identifiability matrix
        param_symbols: List of parameter symbols
        state_symbols: Optional list of state symbols

    Returns:
        Dictionary containing:
            - nullspace_dimension: int
            - fully_identifiable: bool
            - unidentifiable_patterns: list

    Raises:
        ValueError: If matrix is empty or invalid

    Example:
        >>> O = Matrix([[1, 0], [0, 1]])
        >>> params = [sym.Symbol('p1'), sym.Symbol('p2')]
        >>> results = analyze_matrix(O, params)
        >>> results['fully_identifiable']
        True
    """
    if oic_matrix.shape[0] == 0:
        raise ValueError("Matrix cannot be empty")

    # Implementation here
    pass
```

### Documentation Style

**Docstrings:**
- Use Google style
- Include Args, Returns, Raises, Example
- Keep concise but complete

**Comments:**
```python
# Good: Explain WHY
# Use nullspace of nullspace to find identifiable combinations
I = N_matrix.nullspace()

# Bad: Explain WHAT (code is self-explanatory)
# Call nullspace function
I = N_matrix.nullspace()
```

---

## Adding New Features

### Adding a New Analysis Method

1. **Add function to `analysis/`:**
```python
# analysis/new_method.py
def new_analysis_method(matrix, symbols):
    """Perform new type of analysis."""
    # Implementation
    return results
```

2. **Add tests:**
```python
# tests/test_new_method.py
def test_new_method():
    """Test new analysis method."""
    results = new_analysis_method(...)
    assert ...
```

3. **Integrate with CLI:**
```python
# cli/complete_analysis.py
from ..analysis.new_method import new_analysis_method

# Add to workflow
results['new_analysis'] = new_analysis_method(...)
```

4. **Document:**
```markdown
# docs/reference.md
## new_analysis_method()

Performs [description]...
```

### Adding a New Visualization

1. **Add to `visualization/`:**
```python
# visualization/new_plot.py
def create_new_plot(results, output_path):
    """Create new type of plot."""
    fig, ax = plt.subplots()
    # Plotting code
    plt.savefig(output_path)
```

2. **Call from integrated_analysis:**
```python
# analysis/integrated_analysis.py
from ..visualization.new_plot import create_new_plot

create_new_plot(results, output_dir / 'new_plot.png')
```

---

## Performance Considerations

### Symbolic Computation

Symbolic operations are slow. Optimize by:

1. **Use checkpointing** (already implemented)
2. **Simplify expressions early:**
```python
expr = sym.simplify(expr)
```

3. **Limit computation time:**
```python
with time_limit(maxLietime):
    result = expensive_computation()
```

4. **Cache intermediate results:**
```python
@lru_cache(maxsize=128)
def expensive_function(args):
    pass
```

### Memory Usage

Large matrices can consume memory:

1. **Use sparse matrices** when possible
2. **Free memory explicitly:**
```python
del large_matrix
gc.collect()
```

3. **Limit plot resolution** for 3D surfaces

---

## Documentation

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally
mkdocs serve
# Open http://127.0.0.1:8000

# Build static site
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

### Documentation Files

```
docs/
├── index.md          # Landing page
├── quickstart.md     # Quick start guide
├── installation.md   # Installation instructions
├── theory.md         # Mathematical theory
├── examples.md       # Tutorials and examples
├── reference.md      # Complete API reference
└── contributing.md   # This file
```

---

## Release Process

### Version Numbers

Follow Semantic Versioning (semver):
- **Major** (1.x.x): Breaking changes
- **Minor** (x.1.x): New features, backwards compatible
- **Patch** (x.x.1): Bug fixes

### Creating a Release

1. **Update version:**
```python
# pyproject.toml
version = "1.2.0"
```

2. **Update CHANGELOG.md:**
```markdown
## [1.2.0] - 2025-01-16
### Added
- New feature X
### Fixed
- Bug Y
```

3. **Commit and tag:**
```bash
git commit -m "Bump version to 1.2.0"
git tag v1.2.0
git push origin main --tags
```

4. **Create GitHub release** with release notes

---

## Getting Help

### Asking Questions

- **GitHub Discussions:** For general questions
- **GitHub Issues:** For bugs or feature requests
- **Email:** vipulsinghal02@gmail.com

### Reporting Bugs

Include:
1. NullStrike version
2. Python version
3. Operating system
4. Minimal reproducible example
5. Error message / stack trace

**Template:**
```markdown
**NullStrike version:** 1.0.0
**Python version:** 3.10
**OS:** macOS 13.0

**Description:**
Brief description of bug

**Steps to reproduce:**
1. Step 1
2. Step 2

**Expected behavior:**
What should happen

**Actual behavior:**
What actually happens

**Error message:**
```
Traceback...
```
```

---

## Code of Conduct

Be respectful and constructive in all interactions. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

---

## License

NullStrike is licensed under GPL-3.0. All contributions will be under the same license.

By contributing, you agree that your contributions will be licensed under GPL-3.0.

---

## Attribution

NullStrike builds on:
- **StrikePy** by David Rey Rostro (Python implementation)
- **STRIKE-GOLDD** by Alejandro Fernandez Villaverde (original MATLAB)

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed attribution.

---

## Quick Contribution Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines (black, flake8, mypy pass)
- [ ] Tests added for new features
- [ ] All tests pass (`pytest`)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

---

**Thank you for contributing to NullStrike!**

Questions? Open a GitHub Discussion or Issue.
