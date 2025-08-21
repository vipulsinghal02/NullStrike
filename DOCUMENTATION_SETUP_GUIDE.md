# Professional MkDocs Documentation Setup Guide

This guide provides a complete template for replicating the high-quality documentation system used in NullStrike. The setup combines MkDocs Material theme with mathematical rendering, automatic API documentation, and automated GitHub Pages deployment.

## 1. Core Tools & Dependencies

### Essential Package Versions

Create a `requirements-docs.txt` file with these exact dependencies:

```txt
# Documentation dependencies
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-material-extensions>=1.1.0
pymdown-extensions>=10.0.0
mkdocstrings[python]>=0.23.0

# Scientific computing dependencies (for mathematical projects)
sympy>=1.12
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.0
scipy>=1.10.0
```

### Key Components

1. **MkDocs**: Static site generator with hot reload development server
2. **Material Theme**: Professional design with dark/light modes
3. **PyMdown Extensions**: Advanced markdown extensions for code highlighting
4. **MkDocstrings**: Automatic API documentation from Python docstrings
5. **MathJax**: LaTeX mathematical notation rendering

## 2. File Structure & Organization

### Required Directory Structure

```
your-project/
├── docs/                          # Documentation source files
│   ├── index.md                   # Homepage
│   ├── installation.md            # Installation guide
│   ├── quickstart.md             # Quick start tutorial
│   ├── getting-started/          # Beginner guides
│   │   └── first-analysis.md
│   ├── theory/                   # Mathematical foundations
│   │   ├── overview.md
│   │   └── mathematical-details.md
│   ├── guide/                    # User guides
│   │   ├── configuration.md
│   │   └── cli-usage.md
│   ├── examples/                 # Tutorials & examples
│   │   └── simple.md
│   ├── api/                      # API documentation
│   │   └── core.md
│   ├── dev/                      # Developer documentation
│   │   ├── contributing.md
│   │   ├── architecture.md
│   │   └── testing.md
│   └── javascripts/              # Custom JavaScript
│       └── mathjax.js
├── mkdocs.yml                    # MkDocs configuration
├── requirements-docs.txt         # Documentation dependencies
└── .github/workflows/
    └── docs.yml                  # GitHub Actions workflow
```

## 3. MkDocs Configuration (mkdocs.yml)

### Complete Template Configuration

```yaml
site_name: Your Project Documentation
site_description: Professional documentation for your Python package
site_author: Your Name
site_url: https://yourusername.github.io/your-project/
repo_url: https://github.com/yourusername/your-project
repo_name: yourusername/your-project
edit_uri: edit/main/docs/

theme:
  name: material
  palette:
    # Light mode
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs          # Top-level tabs
    - navigation.sections      # Collapsible sections
    - navigation.expand        # Expand navigation by default
    - navigation.top           # Back-to-top button
    - navigation.tracking      # Track scroll position
    - search.highlight         # Highlight search terms
    - search.suggest          # Search suggestions
    - content.code.copy       # Copy button for code blocks
    - content.code.annotate   # Code annotations
    - content.tooltips        # Enhanced tooltips
    - content.tabs.link       # Link tabbed content

# Navigation structure
nav:
  - Home: index.md
  - Getting Started:
    - Installation: installation.md
    - Quick Start: quickstart.md
  - Mathematical Foundations:
    - Theory Overview: theory/overview.md
    - Technical Details: theory/mathematical-details.md
  - User Guide:
    - Configuration: guide/configuration.md
    - Command Line Usage: guide/cli-usage.md
  - Examples & Tutorials:
    - Simple Example: examples/simple.md
  - API Reference:
    - Core Module: api/core.md
  - Development:
    - Contributing: dev/contributing.md
    - Architecture: dev/architecture.md
    - Testing: dev/testing.md

# Markdown extensions for enhanced features
markdown_extensions:
  # Code highlighting with line numbers
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  # Tabbed content
  - pymdownx.tabbed:
      alternate_style: true
  # Admonitions (notes, warnings, etc.)
  - admonition
  - pymdownx.details
  # Mathematical notation
  - pymdownx.arithmatex:
      generic: true
  # Additional features
  - attr_list
  - md_in_html

# Mathematical rendering with MathJax
extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

# API documentation from docstrings
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]  # Path to your Python source code
          options:
            docstring_style: google    # Google-style docstrings
            show_source: true          # Show source code links
            show_bases: true           # Show base classes
            show_root_heading: true    # Show module headings
            show_root_toc_entry: true  # Show in table of contents
            show_object_full_path: false  # Cleaner display
```

## 4. Mathematical Rendering Setup

### MathJax Configuration (docs/javascripts/mathjax.js)

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],      // Inline math: \(x = 5\)
    displayMath: [["\\[", "\\]"]],      // Display math: \[x = 5\]
    processEscapes: true,               // Process backslash escapes
    processEnvironments: true           // Process LaTeX environments
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"      # Only process designated elements
  }
};

// Re-render math when page content changes (for SPA navigation)
document$.subscribe(() => {
  MathJax.typesetPromise()
})
```

### Mathematical Notation Examples

In your markdown files, use:

```markdown
# Inline Mathematics
The equation \(E = mc^2\) is famous.

# Display Mathematics
The fundamental theorem:

\[
\int_a^b f(x) dx = F(b) - F(a)
\]

# Complex Equations with Alignment
\[
\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial \mathbf{E}}{\partial t}
\end{align}
\]
```

## 5. Professional Theme Features

### Content Organization with Tabs

```markdown
=== "Python"

    ```python
    def hello_world():
        print("Hello, World!")
    ```

=== "JavaScript"

    ```javascript
    function helloWorld() {
        console.log("Hello, World!");
    }
    ```

=== "Command Line"

    ```bash
    echo "Hello, World!"
    ```
```

### Admonitions for Important Information

```markdown
!!! note "Important Note"
    
    This is important information that users should know.

!!! warning "Be Careful"
    
    This action cannot be undone.

!!! tip "Pro Tip"
    
    Use keyboard shortcuts for faster navigation.

!!! abstract "Summary"
    
    Key points from this section.
```

### Code Blocks with Advanced Features

```markdown
```python title="example.py" linenums="1" hl_lines="3 4"
def calculate_something(x, y):
    """Calculate something important."""
    result = x * y  # (1)
    return result + 10  # (2)
```

1. Multiply the inputs
2. Add constant offset
```

## 6. API Documentation with mkdocstrings

### API Documentation Template (docs/api/core.md)

```markdown
# Core API Reference

This page documents the core modules and functions for programmatic use.

## Main Functions

### Primary Analysis Function

::: your_package.main.analyze

The main analysis function with complete documentation pulled from docstrings.

#### Basic Usage

```python
from your_package.main import analyze

# Run basic analysis
result = analyze(data_file="data.csv")

# Run with custom options
result = analyze(
    data_file="data.csv",
    method="advanced",
    verbose=True
)
```

#### Return Value

The function returns a dictionary containing:

- `'success'`: Boolean indicating if analysis completed successfully
- `'results'`: Main analysis results
- `'metadata'`: Analysis metadata and timing
```

### Docstring Requirements

Ensure your Python functions have comprehensive docstrings:

```python
def analyze(data_file, method="standard", verbose=False):
    """Analyze data using specified method.
    
    Args:
        data_file (str): Path to input data file
        method (str, optional): Analysis method. Defaults to "standard".
        verbose (bool, optional): Enable verbose output. Defaults to False.
    
    Returns:
        dict: Analysis results containing:
            - success (bool): Whether analysis completed successfully
            - results (dict): Main analysis results
            - metadata (dict): Analysis metadata
    
    Raises:
        FileNotFoundError: If data_file doesn't exist
        ValueError: If method is not supported
    
    Example:
        >>> result = analyze("data.csv", method="advanced")
        >>> print(result['success'])
        True
    """
    pass
```

## 7. GitHub Actions Deployment

### Automated Workflow (.github/workflows/docs.yml)

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:  # Allow manual triggering

permissions:
  contents: read
  pages: write
  id-token: write
  actions: read

# Allow only one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for git info

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-docs-${{ hashFiles('requirements-docs.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-docs-
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-docs.txt

      - name: Setup Pages (with error handling)
        if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
        id: pages
        continue-on-error: true
        uses: actions/configure-pages@v4

      - name: Get Pages URL (fallback)
        if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request' && steps.pages.outcome == 'failure'
        id: pages-fallback
        run: |
          echo "Pages setup failed, using repository-based URL"
          echo "base_url=https://${{ github.repository_owner }}.github.io/${{ github.event.repository.name }}/" >> $GITHUB_OUTPUT

      - name: Configure MkDocs site_url
        if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
        run: |
          if [ "${{ steps.pages.outcome }}" == "success" ]; then
            SITE_URL="${{ steps.pages.outputs.base_url }}"
          else
            SITE_URL="${{ steps.pages-fallback.outputs.base_url }}"
          fi
          echo "Using site URL: $SITE_URL"
          if [ -f mkdocs.yml ]; then
            sed -i "s|site_url:.*|site_url: $SITE_URL|" mkdocs.yml
          fi

      - name: Build documentation
        run: |
          echo "Building MkDocs documentation..."
          mkdocs build --clean --verbose
          echo "Documentation built successfully"

      - name: Upload artifact
        if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./site

  # Deployment job
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

## 8. Homepage Template (docs/index.md)

```markdown
# Your Project Documentation

Welcome to **Your Project** - a powerful tool for [your project's purpose].

## What is Your Project?

Your Project provides [brief description of what your software does and its main benefits].

### The Core Problem

[Describe the problem your software solves and why it matters]

## Key Features

=== "Core Functionality"
    
    - **Feature 1**: Description with mathematical context if relevant
    - **Feature 2**: $\mathcal{F}(x) = \int_0^x f(t) dt$ 
    - **Feature 3**: Symbolic computation using SymPy
    - **Feature 4**: High-performance algorithms

=== "Visualization & Analysis"
    
    - **Interactive Plots**: Dynamic visualizations
    - **3D Graphics**: Advanced 3D rendering
    - **Data Analysis**: Statistical analysis tools
    - **Export Options**: Multiple output formats

=== "Performance & Usability"
    
    - **Fast Algorithms**: Optimized implementations
    - **CLI Interface**: Simple command-line usage
    - **Python API**: Programmatic access
    - **Documentation**: Comprehensive guides

## Quick Start

Get started in just a few commands:

=== "Command Line"

    ```bash
    # Install the package
    pip install your-project
    
    # Run basic analysis
    your-project analyze data.csv
    
    # View results
    your-project results --show
    ```

=== "Python API"

    ```python
    import your_project as yp
    
    # Load data and run analysis
    data = yp.load_data("data.csv")
    results = yp.analyze(data)
    
    # Visualize results
    yp.plot_results(results)
    ```

## Mathematical Foundation

Your Project is built on solid mathematical foundations:

The core algorithm computes:

\[
\mathbf{R} = \mathbf{A}^{-1}\mathbf{B} + \lambda\mathbf{I}
\]

Where:
- $\mathbf{A}$ is the data matrix
- $\mathbf{B}$ contains the measurements  
- $\lambda$ is the regularization parameter
- $\mathbf{I}$ is the identity matrix

!!! tip "Getting Started"
    
    New to the project? Start with our [Quick Start Guide](quickstart.md) and then explore the [Examples](examples/simple.md).

## Next Steps

- **[Installation](installation.md)**: Install and set up the package
- **[Quick Start](quickstart.md)**: Your first analysis in 5 minutes
- **[User Guide](guide/configuration.md)**: Comprehensive usage documentation
- **[API Reference](api/core.md)**: Complete API documentation
```

## 9. GitHub Pages Configuration

### Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under "Source", select **GitHub Actions**
4. The workflow will automatically deploy on pushes to main branch

### Repository Settings

Ensure your repository has:
- GitHub Pages enabled with Actions source
- Workflow permissions set to "Read and write permissions"

## 10. Development Workflow

### Local Development

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Serve documentation locally with hot reload
mkdocs serve

# Build documentation (outputs to site/ directory)
mkdocs build

# Deploy to GitHub Pages (if configured)
mkdocs gh-deploy
```

### Quality Checks

```bash
# Check for broken internal links
mkdocs build --strict

# Validate mathematical notation
# (manually review equations in browser)

# Check API documentation coverage
# (ensure all public functions have docstrings)
```

## 11. Customization Options

### Color Schemes

Customize the theme colors in `mkdocs.yml`:

```yaml
theme:
  palette:
    - scheme: default
      primary: deep-purple    # Options: red, pink, purple, deep-purple, indigo, blue, light-blue, cyan, teal, green, light-green, lime, yellow, amber, orange, deep-orange, brown, grey, blue-grey
      accent: deep-purple
```

### Additional Features

Enable more Material theme features:

```yaml
theme:
  features:
    - navigation.instant     # Instant loading (SPA-like)
    - navigation.indexes     # Section index pages
    - navigation.tabs.sticky # Sticky navigation tabs
    - toc.follow            # Follow table of contents
    - toc.integrate         # Integrate TOC with navigation
```

### Custom CSS

Add custom styling in `docs/stylesheets/extra.css`:

```css
/* Custom styles */
.md-header {
  background-color: var(--md-primary-fg-color);
}

/* Mathematical equation styling */
.arithmatex {
  margin: 1em 0;
}
```

Then reference it in `mkdocs.yml`:

```yaml
extra_css:
  - stylesheets/extra.css
```

## 12. Best Practices

### Documentation Structure

1. **Start broad, get specific**: Overview → Guide → Reference
2. **Include examples**: Every concept should have practical examples
3. **Mathematical context**: Explain the theory behind the implementation
4. **Cross-reference**: Link related sections extensively

### Content Guidelines

1. **Clear headings**: Use descriptive, scannable headings
2. **Code examples**: Include working, tested code snippets
3. **Visual aids**: Use diagrams, equations, and screenshots
4. **Accessibility**: Ensure content works for all users

### Maintenance

1. **Regular updates**: Keep documentation in sync with code
2. **Link checking**: Verify all links work correctly
3. **User feedback**: Monitor and respond to documentation issues
4. **Version control**: Tag documentation releases with code releases

## 13. Troubleshooting

### Common Issues

1. **MathJax not rendering**: Check JavaScript console for errors
2. **API docs missing**: Verify mkdocstrings can find your source code
3. **GitHub Pages not updating**: Check Actions workflow status
4. **Local serve errors**: Ensure all dependencies are installed

### Performance Optimization

1. **Image optimization**: Compress images and use appropriate formats
2. **Caching**: Leverage GitHub Actions caching for faster builds
3. **Selective builds**: Only rebuild when documentation files change

---

## Quick Setup Checklist

- [ ] Copy `requirements-docs.txt` with exact versions
- [ ] Create `mkdocs.yml` with template configuration
- [ ] Set up `docs/` directory structure
- [ ] Add `docs/javascripts/mathjax.js` for math rendering
- [ ] Create `.github/workflows/docs.yml` for automated deployment
- [ ] Enable GitHub Pages with Actions source
- [ ] Write content following markdown examples
- [ ] Test locally with `mkdocs serve`
- [ ] Push to main branch to trigger deployment

This setup provides a professional, maintainable documentation system that scales with your project and impresses users with its polish and functionality.