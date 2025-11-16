# Command Line Interface

This guide covers all aspects of using NullStrike from the command line, including basic usage, advanced options, and troubleshooting.

## Basic Usage

### Simple Analysis

The most basic command runs a complete analysis:

```bash
nullstrike model_name
```

**Example**:
```bash
nullstrike C2M
```

This will:
1. Load the `C2M` model from `custom_models/C2M.py`
2. Use options from `custom_options/options_C2M.py` (if exists) or defaults
3. Run complete identifiability analysis
4. Generate visualizations
5. Save results to `results/C2M/`

### Specify Options File

Use a specific configuration file:

```bash
nullstrike model_name options_file
```

**Example**:
```bash
nullstrike C2M options_custom_C2M
```

This loads options from `custom_options/options_custom_C2M.py`.

## Command Line Options

### Parameters-Only Mode

For quick analysis without visualizations:

```bash
nullstrike model_name --parameters-only
# or
nullstrike model_name -p
```

**When to use**:
- Quick testing during model development
- Large models where visualization is slow
- Debugging model definitions
- When only identifiability results are needed

**Example**:
```bash
nullstrike calibration_single --parameters-only
```

### Help and Version

Get help information:

```bash
nullstrike --help
nullstrike -h
```

Get version information:
```bash
nullstrike --version
```

## Complete Command Syntax

```bash
nullstrike [model_name] [options_file] [--parameters-only] [--help] [--version]
```

### Argument Details

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `model_name` | No | Model to analyze | `calibration_single` |
| `options_file` | No | Configuration file | `options_{model_name}` or `options_default` |
| `--parameters-only` | No | Skip visualizations | Full analysis |
| `--help` | No | Show help message | - |
| `--version` | No | Show version info | - |

## Usage Patterns

### Development Workflow

During model development, use this progression:

```bash
# 1. Quick syntax check
nullstrike my_model --parameters-only

# 2. Full analysis with default options  
nullstrike my_model

# 3. Custom analysis with specific options
nullstrike my_model options_my_custom
```

### Batch Analysis

Analyze multiple models:

```bash
# Analyze several built-in models
nullstrike C2M
nullstrike Bolie  
nullstrike calibration_single

# Analyze with different configurations
nullstrike my_model options_config1
nullstrike my_model options_config2
```

### Comparative Analysis

Compare different model variants:

```bash
nullstrike enzyme_simple
nullstrike enzyme_complex
nullstrike enzyme_with_inhibition
```

## Working with Different Models

### Built-in Models

NullStrike includes several pre-defined models:

```bash
# Pharmacokinetic models
nullstrike C2M                    # Two-compartment model
nullstrike C2M_2outputs          # Two-compartment with multiple outputs

# Biological models  
nullstrike Bolie                 # Glucose-insulin dynamics
nullstrike calibration_single   # Enzyme kinetics
nullstrike calibration_double   # Two-enzyme system

# Test models
nullstrike SimpleExample         # Basic linear system
nullstrike 1A_integral          # Control theory example
```

### Custom Models

For your own models in `custom_models/`:

```bash
# Analyze custom model
nullstrike my_pharmacokinetic_model

# With custom options
nullstrike my_model options_my_model
```

### Model Discovery

List available models:

```bash
# See what's in custom_models/
ls custom_models/

# See what configurations exist
ls custom_options/
```

## Output and Results

### Output Locations

Results are saved to:
```
results/
└── {model_name}/
    ├── analysis_report.txt
    ├── nullspace_analysis.txt  
    ├── observability_matrix.txt
    ├── visualizations/
    └── checkpoints/
```

### Progress Monitoring

During analysis, you'll see progress information:

```
=== NULLSTRIKE ANALYSIS: C2M ===
Loading model: C2M
Loading options: options_C2M

=== STRIKE-GOLDD Analysis ===
Computing Lie derivatives... [■■■■■■    ] 60%
Building observability matrix... Done
Rank analysis... Done
Found 2 identifiable parameters out of 4

=== Nullspace Analysis ===  
Computing nullspace basis... Done
Nullspace dimension: 2
Identifying parameter combinations... Done
Found 2 identifiable combinations

=== Visualization Generation ===
Creating 3D manifolds... [■■■■      ] 40%
Creating 2D projections... [■■■■■■■■  ] 80%  
Building parameter graph... Done

Analysis complete!
Results saved to: results/C2M/
Total computation time: 45.3 seconds
```

### Understanding Output Messages

=== "Success Messages"

    ```
    Model loaded successfully
    Observability matrix computed (rank: 6)
    Nullspace analysis complete
    Visualizations generated
    Results saved to results/C2M/
    ```

=== "Warning Messages"

    ```
    WARNING: Large symbolic expressions detected
    WARNING: Computation time exceeded 2 minutes
    WARNING: Some parameters may be numerically sensitive
    ```

=== "Error Messages"

    ```
    ERROR: Model file 'my_model.py' not found
    ERROR: Invalid options file 'options_bad.py'
    ERROR: Symbolic computation failed
    ```

## Advanced Usage

### Environment Variables

Control NullStrike behavior with environment variables:

```bash
# Set custom model/options directories
export NULLSTRIKE_MODELS_DIR="/path/to/my/models"
export NULLSTRIKE_OPTIONS_DIR="/path/to/my/options"

# Set visualization backend
export MPLBACKEND=Agg  # For headless systems

# Run analysis
nullstrike my_model
```

### Integration with Scripts

Use NullStrike in shell scripts:

```bash
#!/bin/bash
# analyze_all_models.sh

models=("C2M" "Bolie" "calibration_single")

for model in "${models[@]}"; do
    echo "Analyzing $model..."
    nullstrike "$model" --parameters-only
    
    if [ $? -eq 0 ]; then
        echo "$model analysis complete"
    else
        echo "ERROR: $model analysis failed"
    fi
done
```

### Performance Monitoring

Time your analyses:

```bash
# Time a single analysis
time nullstrike C2M

# Compare performance
time nullstrike simple_model
time nullstrike complex_model
```

### Memory Usage

Monitor memory for large models:

```bash
# On Linux/macOS
/usr/bin/time -v nullstrike large_model

# Monitor with top/htop during analysis
nullstrike large_model &
htop -p $!
```

## Troubleshooting

### Common Command Line Issues

=== "Command Not Found"

    **Error**: `command not found: nullstrike`
    
    **Solutions**:
    ```bash
    # Check installation
    pip list | grep nullstrike
    
    # Reinstall if needed
    pip install -e .
    
    # Check PATH
    which nullstrike
    ```

=== "Model Not Found"

    **Error**: `Error: Model file 'my_model.py' not found`
    
    **Solutions**:
    ```bash
    # Check current directory
    pwd
    
    # List available models
    ls custom_models/
    
    # Check filename exactly
    ls custom_models/my_model.py
    ```

=== "Options File Issues"

    **Error**: `Error loading options file`
    
    **Solutions**:
    ```bash
    # Check options file exists
    ls custom_options/options_my_model.py
    
    # Test Python syntax
    python -m py_compile custom_options/options_my_model.py
    
    # Use default options
    nullstrike my_model  # Will use defaults
    ```

### Runtime Errors

=== "Symbolic Computation Failures"

    **Symptoms**: Analysis hangs or crashes during STRIKE-GOLDD phase
    
    **Solutions**:
    ```bash
    # Try parameters-only mode
    nullstrike my_model --parameters-only
    
    # Reduce time limits in options file
    # Set maxLietime = 60 in options file
    
    # Simplify model for testing
    # Comment out complex terms temporarily
    ```

=== "Visualization Errors"

    **Symptoms**: Analysis completes but no plots generated
    
    **Solutions**:
    ```bash
    # Check matplotlib backend
    python -c "import matplotlib; print(matplotlib.get_backend())"
    
    # Set backend for headless systems
    export MPLBACKEND=Agg
    nullstrike my_model
    
    # Skip visualization for testing
    nullstrike my_model --parameters-only
    ```

=== "Memory Issues"

    **Symptoms**: `MemoryError` or system becomes unresponsive
    
    **Solutions**:
    ```bash
    # Monitor memory usage
    nullstrike my_model &
    htop -p $!
    
    # Use parameters-only mode
    nullstrike my_model --parameters-only
    
    # Reduce model complexity
    # Reduce maxLietime in options
    # Limit visualization parameters
    ```

### Performance Issues

=== "Slow Analysis"

    **Problem**: Analysis takes very long time
    
    **Solutions**:
    ```bash
    # Check progress output
    # Look for which phase is slow
    
    # Reduce computation limits
    # Set maxLietime = 120 in options file
    
    # Skip observability check  
    # Set checkObser = 0 in options file
    
    # Use parameters-only for testing
    nullstrike my_model --parameters-only
    ```

=== "Large Output Files"

    **Problem**: Results directory becomes very large
    
    **Solutions**:
    ```bash
    # Check results directory size
    du -sh results/my_model/
    
    # Limit visualization plots
    # Reduce max_triplets_3d, max_pairs_2d in options
    
    # Clean old results
    rm -rf results/old_model/
    ```

## Integration with Other Tools

### IDE Integration

Use NullStrike from within IDEs:

**VS Code**:
```json
{
    "name": "Run NullStrike",
    "type": "shell", 
    "command": "nullstrike",
    "args": ["${workspaceFolder}/my_model"],
    "group": "build"
}
```

**PyCharm**: 
Configure external tool with command `nullstrike` and arguments `$FileDirName$`.

### Jupyter Notebooks

Run NullStrike from Jupyter:

```python
# In Jupyter cell
!nullstrike my_model --parameters-only

# Or using subprocess for better control
import subprocess
result = subprocess.run(['nullstrike', 'my_model'], 
                       capture_output=True, text=True)
print(result.stdout)
```

### Continuous Integration

Include NullStrike in CI pipelines:

```yaml
# GitHub Actions example
- name: Test Models
  run: |
    nullstrike test_model --parameters-only
    if [ $? -ne 0 ]; then exit 1; fi
```

## Best Practices

### Command Line Workflow

1. **Start simple**: Use `--parameters-only` for initial testing
2. **Check syntax**: Verify model and options files load correctly  
3. **Incremental analysis**: Start with simple models, add complexity
4. **Monitor progress**: Watch output for bottlenecks
5. **Save results**: Don't overwrite important analysis results

### Debugging Strategy

1. **Isolate issues**: Test model and options separately
2. **Use minimal examples**: Start with simple test cases
3. **Check file paths**: Verify all files exist and are accessible
4. **Monitor resources**: Watch CPU and memory usage
5. **Read error messages**: Pay attention to specific error details

### Production Usage

1. **Script analyses**: Automate repeated analyses with shell scripts
2. **Document commands**: Keep records of successful command patterns
3. **Version control**: Track model and options files in git
4. **Backup results**: Save important analysis results
5. **Performance testing**: Benchmark analysis times for planning

---

## Further Reading

- **[Model Definition Guide](models.md)**: Creating models for analysis
- **[Configuration Guide](configuration.md)**: Setting up options files
- **[Python API](python-api.md)**: Programmatic interface
- **[Advanced Features](../advanced/batch.md)**: Batch processing and automation

!!! tip "Command Line Efficiency"
    
    - Use tab completion for model names
    - Create aliases for common commands: `alias nsp="nullstrike --parameters-only"`
    - Keep a history of successful commands for reference
    - Use `--parameters-only` during development to save time