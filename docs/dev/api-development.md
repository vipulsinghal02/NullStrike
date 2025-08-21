# API Development and Extensions

This guide covers how to extend NullStrike with new functionality, develop custom analysis methods, and integrate with external tools. It provides patterns and examples for creating robust extensions that integrate seamlessly with the core system.

## Extension Architecture

NullStrike is designed with extensibility in mind through several key mechanisms:

1. **Plugin System**: Register new analysis methods
2. **Inheritance Patterns**: Extend existing classes
3. **Hooks and Callbacks**: Inject custom logic into workflows
4. **Configuration Extensions**: Add new options and parameters

## Creating Analysis Plugins

### Basic Plugin Structure

```python
# my_extensions/fisher_information_plugin.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import sympy as sym

from nullstrike.analysis.base import AnalysisPlugin, AnalysisResults
from nullstrike.core.models import ModelDefinition
from nullstrike.configs.base import AnalysisOptions

class FisherInformationPlugin(AnalysisPlugin):
    """Fisher Information Matrix analysis plugin for identifiability."""
    
    def __init__(self):
        super().__init__()
        self.name = "fisher_information"
        self.description = "Fisher Information Matrix-based identifiability analysis"
        self.version = "1.0.0"
    
    def supports_model(self, model: ModelDefinition) -> bool:
        """Check if plugin can analyze the given model."""
        # Fisher analysis requires differentiable outputs
        return all(isinstance(output, sym.Expr) for output in model.outputs)
    
    def analyze(self, model: ModelDefinition, options: AnalysisOptions) -> 'FisherResults':
        """Perform Fisher Information analysis."""
        try:
            # Compute Fisher Information Matrix
            fim = self._compute_fisher_matrix(model, options)
            
            # Analyze eigenvalues for identifiability
            eigenvalues, eigenvectors = self._eigenvalue_analysis(fim)
            
            # Extract identifiable directions
            identifiable_directions = self._extract_identifiable_directions(
                eigenvalues, eigenvectors, options.tolerance
            )
            
            # Create results object
            return FisherResults(
                fisher_matrix=fim,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                identifiable_directions=identifiable_directions,
                condition_number=np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > options.tolerance])
            )
            
        except Exception as e:
            raise AnalysisError(f"Fisher Information analysis failed: {str(e)}")
    
    def _compute_fisher_matrix(self, model: ModelDefinition, options: AnalysisOptions) -> np.ndarray:
        """Compute Fisher Information Matrix."""
        # Get sensitivity matrix (∂h/∂p)
        sensitivity_matrix = self._compute_sensitivity_matrix(model)
        
        # Fisher Information Matrix: F = S^T * S (assuming unit noise)
        fim = sensitivity_matrix.T @ sensitivity_matrix
        
        return fim
    
    def _compute_sensitivity_matrix(self, model: ModelDefinition) -> np.ndarray:
        """Compute sensitivity matrix ∂h/∂p."""
        n_outputs = len(model.outputs)
        n_params = len(model.parameters)
        
        sensitivity = sym.zeros(n_outputs, n_params)
        
        for i, output in enumerate(model.outputs):
            for j, param in enumerate(model.parameters):
                sensitivity[i, j] = sym.diff(output, param)
        
        return np.array(sensitivity, dtype=float)
    
    def _eigenvalue_analysis(self, fim: np.ndarray) -> tuple:
        """Analyze eigenvalues of Fisher Information Matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(fim)
        
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _extract_identifiable_directions(self, eigenvalues: np.ndarray, 
                                       eigenvectors: np.ndarray, 
                                       tolerance: float) -> list:
        """Extract identifiable parameter directions."""
        # Parameters with eigenvalues > tolerance are identifiable
        identifiable_mask = eigenvalues > tolerance
        identifiable_directions = eigenvectors[:, identifiable_mask]
        
        return identifiable_directions.T.tolist()

class FisherResults(AnalysisResults):
    """Results from Fisher Information analysis."""
    
    def __init__(self, fisher_matrix, eigenvalues, eigenvectors, 
                 identifiable_directions, condition_number):
        super().__init__()
        self.fisher_matrix = fisher_matrix
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.identifiable_directions = identifiable_directions
        self.condition_number = condition_number
        self.analysis_type = "fisher_information"
    
    def summary(self) -> Dict[str, Any]:
        """Return analysis summary."""
        n_identifiable = len(self.identifiable_directions)
        return {
            'analysis_type': self.analysis_type,
            'identifiable_directions': n_identifiable,
            'condition_number': self.condition_number,
            'smallest_eigenvalue': np.min(self.eigenvalues),
            'largest_eigenvalue': np.max(self.eigenvalues)
        }
```

### Registering Plugins

```python
# Plugin registration system
from nullstrike.core.plugin_manager import PluginManager

def register_fisher_plugin():
    """Register the Fisher Information plugin."""
    plugin_manager = PluginManager.get_instance()
    fisher_plugin = FisherInformationPlugin()
    plugin_manager.register_plugin(fisher_plugin)

# Auto-registration through entry points in setup.py
setup(
    name="nullstrike-fisher-extension",
    entry_points={
        'nullstrike.plugins': [
            'fisher_information = my_extensions.fisher_information_plugin:FisherInformationPlugin',
        ],
    },
)
```

### Using Custom Plugins

```python
# Use custom plugin through API
from nullstrike import load_model, AnalysisOptions
from nullstrike.core.plugin_manager import PluginManager

# Load model and options
model = load_model('C2M')
options = AnalysisOptions(tolerance=1e-8)

# Get plugin manager and run custom analysis
plugin_manager = PluginManager.get_instance()
results = plugin_manager.run_analysis('fisher_information', model, options)

print(f"Condition number: {results.condition_number:.2e}")
print(f"Identifiable directions: {len(results.identifiable_directions)}")
```

## Extending Visualization

### Custom Visualization Classes

```python
# my_extensions/custom_visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, Optional

from nullstrike.visualization.base import VisualizationBase
from nullstrike.analysis.results import AnalysisResults

class ParameterHeatmapVisualizer(VisualizationBase):
    """Create heatmap visualization of parameter correlations."""
    
    def __init__(self, results: AnalysisResults, options: Dict[str, Any]):
        super().__init__(results, options)
        self.name = "parameter_heatmap"
    
    def generate(self) -> plt.Figure:
        """Generate correlation heatmap."""
        # Compute parameter correlation matrix
        correlation_matrix = self._compute_parameter_correlations()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.options.get('figsize', (10, 8)))
        
        sns.heatmap(
            correlation_matrix,
            ax=ax,
            cmap=self.options.get('colormap', 'RdBu_r'),
            center=0,
            annot=True,
            fmt='.3f',
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        ax.set_title('Parameter Correlation Matrix')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Parameters')
        
        return fig
    
    def _compute_parameter_correlations(self) -> np.ndarray:
        """Compute correlation matrix from Fisher Information or sensitivity."""
        if hasattr(self.results, 'fisher_matrix'):
            # Use Fisher Information Matrix
            fim = self.results.fisher_matrix
            
            # Convert to correlation matrix
            diagonal = np.sqrt(np.diag(fim))
            correlation = fim / np.outer(diagonal, diagonal)
            
        elif hasattr(self.results, 'observability_matrix'):
            # Use observability matrix
            obs_matrix = self.results.observability_matrix
            correlation = np.corrcoef(obs_matrix.T)
            
        else:
            raise ValueError("No suitable matrix found for correlation analysis")
        
        return correlation

class IdentifiabilityTreeVisualizer(VisualizationBase):
    """Visualize parameter identifiability as hierarchical tree."""
    
    def __init__(self, results: AnalysisResults, options: Dict[str, Any]):
        super().__init__(results, options)
        self.name = "identifiability_tree"
    
    def generate(self) -> plt.Figure:
        """Generate tree visualization."""
        # Build tree structure
        tree_graph = self._build_identifiability_tree()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=self.options.get('figsize', (12, 8)))
        
        # Layout tree
        pos = nx.spring_layout(tree_graph, k=2, iterations=50)
        
        # Draw nodes
        identifiable_nodes = [node for node, data in tree_graph.nodes(data=True) 
                            if data.get('identifiable', False)]
        unidentifiable_nodes = [node for node, data in tree_graph.nodes(data=True) 
                              if not data.get('identifiable', False)]
        
        nx.draw_networkx_nodes(tree_graph, pos, nodelist=identifiable_nodes,
                             node_color='lightgreen', node_size=500, ax=ax)
        nx.draw_networkx_nodes(tree_graph, pos, nodelist=unidentifiable_nodes,
                             node_color='lightcoral', node_size=500, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(tree_graph, pos, ax=ax, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(tree_graph, pos, ax=ax)
        
        ax.set_title('Parameter Identifiability Tree')
        ax.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=10, label='Identifiable'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=10, label='Unidentifiable')
        ]
        ax.legend(handles=legend_elements)
        
        return fig
    
    def _build_identifiability_tree(self) -> nx.Graph:
        """Build tree graph from identifiability results."""
        G = nx.Graph()
        
        # Add parameter nodes
        for i, param in enumerate(self.results.model.parameters):
            is_identifiable = i in self.results.identifiable_parameter_indices
            G.add_node(str(param), identifiable=is_identifiable, type='parameter')
        
        # Add combination nodes
        for i, combo in enumerate(self.results.identifiable_combinations):
            combo_name = f"combo_{i}"
            G.add_node(combo_name, identifiable=True, type='combination')
            
            # Connect to constituent parameters
            for param in combo.free_symbols:
                if str(param) in G.nodes:
                    G.add_edge(str(param), combo_name)
        
        return G
```

### Integrating Custom Visualizations

```python
# Register custom visualizations
from nullstrike.visualization.manager import VisualizationManager

def register_custom_visualizations():
    """Register custom visualization plugins."""
    viz_manager = VisualizationManager.get_instance()
    
    viz_manager.register_visualizer('parameter_heatmap', ParameterHeatmapVisualizer)
    viz_manager.register_visualizer('identifiability_tree', IdentifiabilityTreeVisualizer)

# Use in analysis workflow
from nullstrike.cli.complete_analysis import main

# Run analysis with custom visualizations
result = main('C2M', custom_visualizations=['parameter_heatmap', 'identifiability_tree'])
```

## Model Format Extensions

### Custom Model Loaders

```python
# my_extensions/sbml_loader.py
import libsbml
from typing import List, Dict, Any
import sympy as sym

from nullstrike.core.models import ModelDefinition, ModelLoader

class SBMLModelLoader(ModelLoader):
    """Load SBML models and convert to NullStrike format."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.sbml', '.xml']
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the file."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def load_model(self, file_path: str) -> ModelDefinition:
        """Load SBML model and convert to NullStrike format."""
        try:
            # Read SBML file
            document = libsbml.readSBML(file_path)
            
            if document.getNumErrors() > 0:
                raise ModelLoadError(f"SBML parsing errors: {document.getErrorLog()}")
            
            model = document.getModel()
            
            # Extract model components
            states = self._extract_species(model)
            parameters = self._extract_parameters(model)
            dynamics = self._extract_reactions(model, states, parameters)
            outputs = self._extract_outputs(model, states)
            
            return ModelDefinition(
                states=states,
                parameters=parameters, 
                dynamics=dynamics,
                outputs=outputs,
                name=model.getId() or "sbml_model"
            )
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SBML model: {str(e)}")
    
    def _extract_species(self, model) -> List[sym.Symbol]:
        """Extract species (states) from SBML model."""
        species = []
        for s in model.getListOfSpecies():
            if not s.getBoundaryCondition():  # Only dynamic species
                species.append(sym.Symbol(s.getId()))
        return species
    
    def _extract_parameters(self, model) -> List[sym.Symbol]:
        """Extract parameters from SBML model."""
        parameters = []
        
        # Global parameters
        for p in model.getListOfParameters():
            if not p.getConstant():  # Only non-constant parameters
                parameters.append(sym.Symbol(p.getId()))
        
        # Kinetic parameters from reactions
        for reaction in model.getListOfReactions():
            kinetic_law = reaction.getKineticLaw()
            if kinetic_law:
                for p in kinetic_law.getListOfParameters():
                    parameters.append(sym.Symbol(p.getId()))
        
        return parameters
    
    def _extract_reactions(self, model, states: List[sym.Symbol], 
                          parameters: List[sym.Symbol]) -> List[sym.Expr]:
        """Extract reaction dynamics from SBML model."""
        # Create dynamics equations from reactions
        dynamics = {state: sym.Integer(0) for state in states}
        
        for reaction in model.getListOfReactions():
            # Parse kinetic law
            kinetic_law = reaction.getKineticLaw()
            if not kinetic_law:
                continue
            
            rate_formula = kinetic_law.getFormula()
            rate_expr = self._parse_math_formula(rate_formula, states + parameters)
            
            # Add to reactant dynamics (negative)
            for reactant in reaction.getListOfReactants():
                species_id = reactant.getSpecies()
                stoichiometry = reactant.getStoichiometry()
                species_symbol = next((s for s in states if str(s) == species_id), None)
                if species_symbol:
                    dynamics[species_symbol] -= stoichiometry * rate_expr
            
            # Add to product dynamics (positive)
            for product in reaction.getListOfProducts():
                species_id = product.getSpecies()
                stoichiometry = product.getStoichiometry()
                species_symbol = next((s for s in states if str(s) == species_id), None)
                if species_symbol:
                    dynamics[species_symbol] += stoichiometry * rate_expr
        
        return [dynamics[state] for state in states]
    
    def _extract_outputs(self, model, states: List[sym.Symbol]) -> List[sym.Expr]:
        """Extract outputs from SBML model."""
        # Default: all states are observable
        outputs = list(states)
        
        # TODO: Extract from SBML annotations or rules if available
        
        return outputs
    
    def _parse_math_formula(self, formula: str, symbols: List[sym.Symbol]) -> sym.Expr:
        """Parse SBML mathematical formula to SymPy expression."""
        # Create symbol mapping
        symbol_dict = {str(symbol): symbol for symbol in symbols}
        
        # Parse with SymPy (this is simplified - real implementation would
        # need to handle SBML-specific functions)
        try:
            expr = sym.sympify(formula, locals=symbol_dict)
            return expr
        except Exception as e:
            raise ModelLoadError(f"Failed to parse formula '{formula}': {str(e)}")

# Register the SBML loader
from nullstrike.core.model_manager import ModelManager

def register_sbml_loader():
    """Register SBML model loader."""
    model_manager = ModelManager.get_instance()
    sbml_loader = SBMLModelLoader()
    model_manager.register_loader(sbml_loader)
```

### Using Custom Model Loaders

```python
# Use custom model loader
from my_extensions.sbml_loader import register_sbml_loader
from nullstrike import load_model

# Register the SBML loader
register_sbml_loader()

# Load SBML model
model = load_model('path/to/model.sbml')

# Run analysis
from nullstrike.cli.complete_analysis import main
result = main(model)
```

## Configuration Extensions

### Custom Options Classes

```python
# my_extensions/advanced_options.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

from nullstrike.configs.base import AnalysisOptions

@dataclass
class FisherAnalysisOptions(AnalysisOptions):
    """Extended options for Fisher Information analysis."""
    
    # Fisher-specific options
    noise_model: str = "gaussian"  # "gaussian", "poisson", "custom"
    noise_parameters: Dict[str, float] = None
    regularization: float = 1e-12
    condition_threshold: float = 1e12
    
    # Sensitivity analysis options
    perturbation_size: float = 1e-6
    finite_difference_method: str = "central"  # "forward", "central", "complex"
    
    # Advanced numerical options
    matrix_solver: str = "numpy"  # "numpy", "scipy", "eigen"
    eigenvalue_method: str = "eigh"  # "eigh", "eig", "svd"
    
    def __post_init__(self):
        """Validate and set defaults for custom options."""
        super().__post_init__()
        
        if self.noise_parameters is None:
            self.noise_parameters = {}
        
        # Validate noise model
        valid_noise_models = ["gaussian", "poisson", "custom"]
        if self.noise_model not in valid_noise_models:
            raise ValueError(f"noise_model must be one of {valid_noise_models}")
        
        # Validate finite difference method
        valid_fd_methods = ["forward", "central", "complex"]
        if self.finite_difference_method not in valid_fd_methods:
            raise ValueError(f"finite_difference_method must be one of {valid_fd_methods}")

@dataclass  
class BatchAnalysisOptions(AnalysisOptions):
    """Options for batch analysis of multiple models."""
    
    # Batch processing options
    parallel_workers: int = 4
    batch_size: int = 10
    timeout_per_model: float = 300.0  # seconds
    
    # Output options
    aggregate_results: bool = True
    generate_comparison_plots: bool = True
    save_individual_results: bool = True
    
    # Error handling
    continue_on_error: bool = True
    max_failures: int = 5
    
    def validate_batch_options(self):
        """Validate batch-specific options."""
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.timeout_per_model <= 0:
            raise ValueError("timeout_per_model must be positive")
```

### Configuration Validation and Defaults

```python
# my_extensions/config_validators.py
from typing import Any, Dict, List, Optional
from nullstrike.configs.validation import ConfigValidator, ValidationError

class FisherConfigValidator(ConfigValidator):
    """Validator for Fisher Information analysis configuration."""
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Fisher analysis configuration."""
        validated = super().validate(config)
        
        # Validate Fisher-specific options
        if 'noise_model' in config:
            self._validate_noise_model(config['noise_model'])
        
        if 'regularization' in config:
            self._validate_regularization(config['regularization'])
        
        if 'condition_threshold' in config:
            self._validate_condition_threshold(config['condition_threshold'])
        
        return validated
    
    def _validate_noise_model(self, noise_model: str):
        """Validate noise model specification."""
        valid_models = ["gaussian", "poisson", "custom"]
        if noise_model not in valid_models:
            raise ValidationError(f"Invalid noise_model '{noise_model}'. Must be one of {valid_models}")
    
    def _validate_regularization(self, regularization: float):
        """Validate regularization parameter."""
        if not isinstance(regularization, (int, float)):
            raise ValidationError("regularization must be a number")
        
        if regularization < 0:
            raise ValidationError("regularization must be non-negative")
        
        if regularization > 1e-3:
            import warnings
            warnings.warn("Large regularization value may affect results")
    
    def _validate_condition_threshold(self, threshold: float):
        """Validate condition number threshold."""
        if not isinstance(threshold, (int, float)):
            raise ValidationError("condition_threshold must be a number")
        
        if threshold <= 1:
            raise ValidationError("condition_threshold must be > 1")

# Register custom validators
from nullstrike.configs.validation import ValidationManager

def register_custom_validators():
    """Register custom configuration validators."""
    validation_manager = ValidationManager.get_instance()
    validation_manager.register_validator('fisher_analysis', FisherConfigValidator())
```

## Workflow Extensions

### Custom Analysis Workflows

```python
# my_extensions/custom_workflows.py
from typing import List, Dict, Any, Optional
import time
import logging

from nullstrike.analysis.base import AnalysisWorkflow, AnalysisResults
from nullstrike.core.models import ModelDefinition
from nullstrike.configs.base import AnalysisOptions

class ComparativeAnalysisWorkflow(AnalysisWorkflow):
    """Workflow for comparing multiple identifiability methods."""
    
    def __init__(self, model: ModelDefinition, options: AnalysisOptions, 
                 methods: List[str] = None):
        super().__init__(model, options)
        self.methods = methods or ['strike_goldd', 'fisher_information', 'monte_carlo']
        self.comparison_results = {}
    
    def run_analysis(self) -> 'ComparativeResults':
        """Run comparative analysis using multiple methods."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting comparative analysis with methods: {self.methods}")
        
        results = {}
        timings = {}
        
        for method in self.methods:
            logger.info(f"Running {method} analysis...")
            
            start_time = time.time()
            try:
                method_result = self._run_single_method(method)
                results[method] = method_result
                timings[method] = time.time() - start_time
                logger.info(f"✓ {method} completed in {timings[method]:.2f}s")
                
            except Exception as e:
                logger.error(f"✗ {method} failed: {str(e)}")
                results[method] = None
                timings[method] = None
        
        # Generate comparison analysis
        comparison = self._compare_methods(results)
        
        return ComparativeResults(
            individual_results=results,
            timings=timings,
            comparison=comparison,
            model=self.model,
            methods=self.methods
        )
    
    def _run_single_method(self, method: str) -> AnalysisResults:
        """Run analysis using a single method."""
        from nullstrike.core.plugin_manager import PluginManager
        
        plugin_manager = PluginManager.get_instance()
        
        if plugin_manager.has_plugin(method):
            return plugin_manager.run_analysis(method, self.model, self.options)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    def _compare_methods(self, results: Dict[str, AnalysisResults]) -> Dict[str, Any]:
        """Compare results from different methods."""
        comparison = {
            'agreement': {},
            'discrepancies': [],
            'recommendations': []
        }
        
        # Compare identifiable parameter counts
        identifiable_counts = {}
        for method, result in results.items():
            if result is not None:
                if hasattr(result, 'identifiable_parameters'):
                    identifiable_counts[method] = len(result.identifiable_parameters)
                elif hasattr(result, 'identifiable_directions'):
                    identifiable_counts[method] = len(result.identifiable_directions)
        
        comparison['identifiable_counts'] = identifiable_counts
        
        # Check agreement between methods
        if len(set(identifiable_counts.values())) == 1:
            comparison['agreement']['identifiable_count'] = True
        else:
            comparison['discrepancies'].append(
                f"Methods disagree on identifiable parameter count: {identifiable_counts}"
            )
        
        # Generate recommendations
        if 'strike_goldd' in results and 'fisher_information' in results:
            comparison['recommendations'].append(
                "Compare STRIKE-GOLDD structural results with Fisher Information numerical analysis"
            )
        
        return comparison

class ComparativeResults(AnalysisResults):
    """Results from comparative analysis."""
    
    def __init__(self, individual_results, timings, comparison, model, methods):
        super().__init__()
        self.individual_results = individual_results
        self.timings = timings
        self.comparison = comparison
        self.model = model
        self.methods = methods
        self.analysis_type = "comparative"
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary of comparative analysis."""
        successful_methods = [m for m, r in self.individual_results.items() if r is not None]
        
        return {
            'analysis_type': self.analysis_type,
            'methods_requested': len(self.methods),
            'methods_successful': len(successful_methods),
            'successful_methods': successful_methods,
            'total_time': sum(t for t in self.timings.values() if t is not None),
            'fastest_method': min(self.timings, key=lambda k: self.timings[k] or float('inf')),
            'agreement_summary': self.comparison.get('agreement', {}),
            'discrepancy_count': len(self.comparison.get('discrepancies', []))
        }
    
    def generate_report(self) -> str:
        """Generate detailed comparison report."""
        report = ["Comparative Identifiability Analysis Report"]
        report.append("=" * 50)
        report.append("")
        
        # Model information
        report.append(f"Model: {self.model.name}")
        report.append(f"States: {len(self.model.states)}")
        report.append(f"Parameters: {len(self.model.parameters)}")
        report.append("")
        
        # Method results
        report.append("Method Results:")
        report.append("-" * 20)
        for method in self.methods:
            result = self.individual_results.get(method)
            timing = self.timings.get(method)
            
            if result is not None:
                report.append(f"✓ {method}: Success ({timing:.2f}s)")
                if hasattr(result, 'identifiable_parameters'):
                    report.append(f"  Identifiable parameters: {len(result.identifiable_parameters)}")
                if hasattr(result, 'condition_number'):
                    report.append(f"  Condition number: {result.condition_number:.2e}")
            else:
                report.append(f"✗ {method}: Failed")
            report.append("")
        
        # Comparison summary
        report.append("Comparison Summary:")
        report.append("-" * 20)
        
        if self.comparison['discrepancies']:
            report.append("Discrepancies found:")
            for discrepancy in self.comparison['discrepancies']:
                report.append(f"  • {discrepancy}")
        else:
            report.append("All methods agree on key results")
        
        if self.comparison['recommendations']:
            report.append("")
            report.append("Recommendations:")
            for rec in self.comparison['recommendations']:
                report.append(f"  • {rec}")
        
        return "\n".join(report)
```

## Integration APIs

### External Tool Integration

```python
# my_extensions/matlab_integration.py
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional

from nullstrike.core.models import ModelDefinition
from nullstrike.analysis.results import AnalysisResults

class MatlabIntegration:
    """Integration with MATLAB-based identifiability tools."""
    
    def __init__(self, matlab_path: str = "matlab"):
        self.matlab_path = matlab_path
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def export_to_matlab(self, model: ModelDefinition, output_file: str):
        """Export NullStrike model to MATLAB format."""
        matlab_code = self._generate_matlab_model(model)
        
        with open(output_file, 'w') as f:
            f.write(matlab_code)
    
    def run_matlab_analysis(self, model: ModelDefinition, 
                          script_template: str = "default") -> Dict[str, Any]:
        """Run identifiability analysis in MATLAB."""
        # Export model to MATLAB
        model_file = self.temp_dir / "model.m"
        self.export_to_matlab(model, model_file)
        
        # Generate analysis script
        script_file = self.temp_dir / "analyze.m"
        self._generate_analysis_script(model, script_file, script_template)
        
        # Run MATLAB
        result = self._execute_matlab_script(script_file)
        
        return result
    
    def _generate_matlab_model(self, model: ModelDefinition) -> str:
        """Generate MATLAB model definition."""
        lines = ["% Auto-generated NullStrike model"]
        lines.append("")
        
        # States
        lines.append("% State variables")
        state_names = [str(state) for state in model.states]
        lines.append(f"syms {' '.join(state_names)}")
        lines.append("")
        
        # Parameters  
        lines.append("% Parameters")
        param_names = [str(param) for param in model.parameters]
        lines.append(f"syms {' '.join(param_names)}")
        lines.append("")
        
        # Dynamics
        lines.append("% System dynamics")
        lines.append("f = [")
        for i, dynamic in enumerate(model.dynamics):
            matlab_expr = self._sympy_to_matlab(dynamic)
            lines.append(f"    {matlab_expr};")
        lines.append("];")
        lines.append("")
        
        # Outputs
        lines.append("% Output functions")
        lines.append("h = [")
        for output in model.outputs:
            matlab_expr = self._sympy_to_matlab(output)
            lines.append(f"    {matlab_expr};")
        lines.append("];")
        
        return "\n".join(lines)
    
    def _sympy_to_matlab(self, expr) -> str:
        """Convert SymPy expression to MATLAB syntax."""
        # This is a simplified conversion - real implementation would
        # need comprehensive SymPy -> MATLAB translation
        matlab_expr = str(expr)
        
        # Basic replacements
        replacements = {
            '**': '^',
            'log': 'log',
            'exp': 'exp',
            'sin': 'sin',
            'cos': 'cos'
        }
        
        for sympy_func, matlab_func in replacements.items():
            matlab_expr = matlab_expr.replace(sympy_func, matlab_func)
        
        return matlab_expr
    
    def _generate_analysis_script(self, model: ModelDefinition, 
                                script_file: Path, template: str):
        """Generate MATLAB analysis script."""
        if template == "strike_goldd":
            script_content = self._generate_strike_goldd_script(model)
        elif template == "daisy":
            script_content = self._generate_daisy_script(model)
        else:
            script_content = self._generate_default_script(model)
        
        with open(script_file, 'w') as f:
            f.write(script_content)
    
    def _generate_strike_goldd_script(self, model: ModelDefinition) -> str:
        """Generate MATLAB script for STRIKE-GOLDD analysis."""
        lines = ["% STRIKE-GOLDD analysis script"]
        lines.append("addpath('/path/to/strike-goldd');")
        lines.append("")
        lines.append("% Load model")
        lines.append("run('model.m');")
        lines.append("")
        lines.append("% Run STRIKE-GOLDD")
        lines.append("options.maxLietime = 300;")
        lines.append("results = strike_goldd(f, h, {}, options);")
        lines.append("")
        lines.append("% Save results")
        lines.append("save('results.mat', 'results');")
        
        return "\n".join(lines)
    
    def _execute_matlab_script(self, script_file: Path) -> Dict[str, Any]:
        """Execute MATLAB script and return results."""
        # Change to temporary directory
        old_cwd = Path.cwd()
        
        try:
            import os
            os.chdir(self.temp_dir)
            
            # Run MATLAB
            cmd = [self.matlab_path, '-batch', f"run('{script_file.name}')"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"MATLAB execution failed: {result.stderr}")
            
            # Load results
            results_file = self.temp_dir / "results.mat"
            if results_file.exists():
                # Parse MATLAB results (simplified)
                return {'status': 'success', 'output': result.stdout}
            else:
                return {'status': 'no_results', 'output': result.stdout}
                
        finally:
            os.chdir(old_cwd)

# R Integration
class RIntegration:
    """Integration with R-based identifiability tools."""
    
    def __init__(self, r_path: str = "Rscript"):
        self.r_path = r_path
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def run_r_analysis(self, model: ModelDefinition, 
                      packages: List[str] = None) -> Dict[str, Any]:
        """Run identifiability analysis in R."""
        packages = packages or ['identifiability', 'deSolve']
        
        # Generate R script
        r_script = self._generate_r_script(model, packages)
        script_file = self.temp_dir / "analyze.R"
        
        with open(script_file, 'w') as f:
            f.write(r_script)
        
        # Execute R script
        result = subprocess.run([self.r_path, str(script_file)], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise RuntimeError(f"R execution failed: {result.stderr}")
        
        return {'status': 'success', 'output': result.stdout}
    
    def _generate_r_script(self, model: ModelDefinition, packages: List[str]) -> str:
        """Generate R script for identifiability analysis."""
        lines = ["# Auto-generated R analysis script"]
        lines.append("")
        
        # Load packages
        for package in packages:
            lines.append(f"library({package})")
        lines.append("")
        
        # Model definition (simplified)
        lines.append("# Model definition")
        lines.append("# TODO: Convert SymPy model to R format")
        lines.append("")
        
        # Analysis
        lines.append("# Run identifiability analysis")
        lines.append("# results <- identifiability_analysis(model)")
        lines.append("print('R analysis completed')")
        
        return "\n".join(lines)
```

This comprehensive API development guide provides the foundation for extending NullStrike with custom functionality while maintaining consistency with the core architecture. The plugin system, visualization extensions, and integration APIs enable powerful customizations for specialized use cases.

---

## Next Steps

1. **Study the existing plugin examples** in the NullStrike codebase
2. **Implement a simple plugin** following the patterns shown here
3. **Test your extensions** using the testing framework
4. **Contribute back** successful extensions to the main project
5. **Explore [Performance Optimization](performance.md)** for efficient implementations