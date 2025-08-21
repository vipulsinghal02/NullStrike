# Custom Analysis Workflows

This guide covers creating custom analysis workflows for specialized use cases, combining multiple analysis methods, and building sophisticated analysis pipelines.

## Workflow Architecture

### Basic Workflow Structure

```python
# workflow_base.py
"""Base framework for custom analysis workflows."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time
import logging
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    name: str
    function: callable
    dependencies: List[str] = None
    timeout: float = 300.0
    retry_count: int = 0
    critical: bool = True

@dataclass
class WorkflowResults:
    """Results from workflow execution."""
    workflow_name: str
    total_time: float
    step_results: Dict[str, Any]
    success: bool
    errors: List[str] = None

class WorkflowBase(ABC):
    """Base class for custom analysis workflows."""
    
    def __init__(self, name: str, model, options):
        self.name = name
        self.model = model
        self.options = options
        self.steps: List[WorkflowStep] = []
        self.results = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup workflow logging."""
        logger = logging.getLogger(f'workflow_{self.name}')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def define_workflow(self):
        """Define the workflow steps."""
        pass
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps.append(step)
    
    def execute(self) -> WorkflowResults:
        """Execute the complete workflow."""
        start_time = time.time()
        errors = []
        
        self.logger.info(f"Starting workflow: {self.name}")
        
        # Define workflow if not already done
        if not self.steps:
            self.define_workflow()
        
        # Execute steps in dependency order
        execution_order = self._resolve_dependencies()
        
        for step_name in execution_order:
            step = next(s for s in self.steps if s.name == step_name)
            
            try:
                self.logger.info(f"Executing step: {step_name}")
                step_start = time.time()
                
                result = self._execute_step(step)
                step_time = time.time() - step_start
                
                self.results[step_name] = {
                    'result': result,
                    'execution_time': step_time,
                    'success': True
                }
                
                self.logger.info(f"✓ {step_name} completed in {step_time:.1f}s")
                
            except Exception as e:
                error_msg = f"Step {step_name} failed: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
                self.results[step_name] = {
                    'result': None,
                    'execution_time': 0,
                    'success': False,
                    'error': str(e)
                }
                
                if step.critical:
                    self.logger.error(f"Critical step {step_name} failed, stopping workflow")
                    break
        
        total_time = time.time() - start_time
        success = len(errors) == 0
        
        workflow_results = WorkflowResults(
            workflow_name=self.name,
            total_time=total_time,
            step_results=self.results,
            success=success,
            errors=errors
        )
        
        self.logger.info(f"Workflow completed in {total_time:.1f}s (success: {success})")
        
        return workflow_results
    
    def _execute_step(self, step: WorkflowStep):
        """Execute a single workflow step."""
        # Prepare arguments from previous results
        kwargs = {
            'model': self.model,
            'options': self.options,
            'previous_results': self.results
        }
        
        # Execute with timeout and retry logic
        for attempt in range(step.retry_count + 1):
            try:
                result = step.function(**kwargs)
                return result
            except Exception as e:
                if attempt < step.retry_count:
                    self.logger.warning(f"Step {step.name} failed (attempt {attempt + 1}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve step dependencies to determine execution order."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(step_name):
            if step_name in visited:
                return
            
            step = next(s for s in self.steps if s.name == step_name)
            
            if step.dependencies:
                for dep in step.dependencies:
                    visit(dep)
            
            visited.add(step_name)
            order.append(step_name)
        
        for step in self.steps:
            visit(step.name)
        
        return order
```

## Specialized Workflow Examples

### 1. Multi-Method Comparison Workflow

```python
# comparison_workflow.py
"""Workflow for comparing multiple identifiability methods."""

class IdentifiabilityComparisonWorkflow(WorkflowBase):
    """Compare multiple identifiability analysis methods."""
    
    def __init__(self, model, options, methods=None):
        super().__init__("identifiability_comparison", model, options)
        self.methods = methods or ['strike_goldd', 'fisher_information', 'profile_likelihood']
    
    def define_workflow(self):
        """Define comparison workflow steps."""
        
        # Step 1: Model validation
        self.add_step(WorkflowStep(
            name="model_validation",
            function=self._validate_model,
            critical=True
        ))
        
        # Step 2: Run each method
        for method in self.methods:
            self.add_step(WorkflowStep(
                name=f"run_{method}",
                function=lambda **kwargs, m=method: self._run_method(m, **kwargs),
                dependencies=["model_validation"],
                critical=False,  # Allow other methods to continue if one fails
                timeout=600.0
            ))
        
        # Step 3: Compare results
        self.add_step(WorkflowStep(
            name="compare_results",
            function=self._compare_methods,
            dependencies=[f"run_{method}" for method in self.methods]
        ))
        
        # Step 4: Generate report
        self.add_step(WorkflowStep(
            name="generate_report",
            function=self._generate_comparison_report,
            dependencies=["compare_results"]
        ))
    
    def _validate_model(self, model, **kwargs):
        """Validate model definition."""
        from nullstrike.core.validation import ModelValidator
        
        validator = ModelValidator()
        validation_result = validator.validate(model)
        
        if not validation_result.is_valid:
            raise ValueError(f"Model validation failed: {validation_result.errors}")
        
        return validation_result
    
    def _run_method(self, method_name, model, options, **kwargs):
        """Run specific identifiability method."""
        
        if method_name == 'strike_goldd':
            return self._run_strike_goldd(model, options)
        elif method_name == 'fisher_information':
            return self._run_fisher_information(model, options)
        elif method_name == 'profile_likelihood':
            return self._run_profile_likelihood(model, options)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _run_strike_goldd(self, model, options):
        """Run STRIKE-GOLDD analysis."""
        from nullstrike.cli.complete_analysis import main as strike_analysis
        
        result = strike_analysis(model.name, parameters_only=True)
        
        return {
            'method': 'strike_goldd',
            'identifiable_count': len(result.strike_goldd_results.identifiable_parameters),
            'nullspace_dimension': result.nullspace_results.nullspace_basis.shape[1],
            'computation_time': result.computation_time,
            'identifiable_combinations': result.nullspace_results.identifiable_combinations
        }
    
    def _run_fisher_information(self, model, options):
        """Run Fisher Information analysis."""
        from nullstrike.extensions.fisher_information import FisherInformationAnalyzer
        
        analyzer = FisherInformationAnalyzer(model, options)
        result = analyzer.analyze()
        
        return {
            'method': 'fisher_information',
            'condition_number': result.condition_number,
            'identifiable_directions': len(result.identifiable_directions),
            'computation_time': result.computation_time,
            'eigenvalues': result.eigenvalues
        }
    
    def _run_profile_likelihood(self, model, options):
        """Run Profile Likelihood analysis."""
        # Placeholder for profile likelihood implementation
        return {
            'method': 'profile_likelihood',
            'confidence_intervals': [],
            'computation_time': 0,
            'status': 'not_implemented'
        }
    
    def _compare_methods(self, previous_results, **kwargs):
        """Compare results from different methods."""
        
        comparison = {
            'methods_successful': [],
            'methods_failed': [],
            'identifiability_agreement': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        successful_results = {}
        
        for method in self.methods:
            step_key = f"run_{method}"
            if step_key in previous_results and previous_results[step_key]['success']:
                result = previous_results[step_key]['result']
                successful_results[method] = result
                comparison['methods_successful'].append(method)
            else:
                comparison['methods_failed'].append(method)
        
        # Compare identifiability results
        if 'strike_goldd' in successful_results and 'fisher_information' in successful_results:
            sg_count = successful_results['strike_goldd']['identifiable_count']
            fi_count = successful_results['fisher_information']['identifiable_directions']
            
            comparison['identifiability_agreement']['strike_goldd_vs_fisher'] = {
                'strike_goldd_count': sg_count,
                'fisher_count': fi_count,
                'agreement': abs(sg_count - fi_count) <= 1  # Allow small differences
            }
        
        # Performance comparison
        for method, result in successful_results.items():
            comparison['performance_comparison'][method] = {
                'computation_time': result['computation_time'],
                'method_specific_metrics': self._extract_method_metrics(result)
            }
        
        # Generate recommendations
        if len(successful_results) > 1:
            fastest_method = min(successful_results.keys(), 
                               key=lambda m: successful_results[m]['computation_time'])
            comparison['recommendations'].append(f"Fastest method: {fastest_method}")
        
        if comparison['methods_failed']:
            comparison['recommendations'].append(
                f"Failed methods: {', '.join(comparison['methods_failed'])} - consider model simplification"
            )
        
        return comparison
    
    def _extract_method_metrics(self, result):
        """Extract method-specific performance metrics."""
        method = result['method']
        
        if method == 'strike_goldd':
            return {
                'nullspace_dimension': result['nullspace_dimension'],
                'identifiable_combinations': len(result['identifiable_combinations'])
            }
        elif method == 'fisher_information':
            return {
                'condition_number': result['condition_number'],
                'smallest_eigenvalue': min(result['eigenvalues']) if result['eigenvalues'] else None
            }
        else:
            return {}
    
    def _generate_comparison_report(self, previous_results, **kwargs):
        """Generate comprehensive comparison report."""
        
        comparison = previous_results['compare_results']['result']
        
        report = []
        report.append("# Identifiability Method Comparison Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Methods attempted: {len(self.methods)}")
        report.append(f"- Methods successful: {len(comparison['methods_successful'])}")
        report.append(f"- Methods failed: {len(comparison['methods_failed'])}")
        report.append("")
        
        # Method results
        report.append("## Method Results")
        for method in comparison['methods_successful']:
            step_result = previous_results[f"run_{method}"]
            result = step_result['result']
            
            report.append(f"### {method.upper()}")
            report.append(f"- Computation time: {result['computation_time']:.1f}s")
            
            if method == 'strike_goldd':
                report.append(f"- Identifiable parameters: {result['identifiable_count']}")
                report.append(f"- Nullspace dimension: {result['nullspace_dimension']}")
                report.append(f"- Parameter combinations: {len(result['identifiable_combinations'])}")
            elif method == 'fisher_information':
                report.append(f"- Condition number: {result['condition_number']:.2e}")
                report.append(f"- Identifiable directions: {result['identifiable_directions']}")
            
            report.append("")
        
        # Failed methods
        if comparison['methods_failed']:
            report.append("## Failed Methods")
            for method in comparison['methods_failed']:
                step_result = previous_results[f"run_{method}"]
                error = step_result.get('error', 'Unknown error')
                report.append(f"- {method}: {error}")
            report.append("")
        
        # Agreements and disagreements
        if comparison['identifiability_agreement']:
            report.append("## Method Agreement")
            for comparison_name, agreement_data in comparison['identifiability_agreement'].items():
                if agreement_data['agreement']:
                    report.append(f"✓ {comparison_name}: Methods agree")
                else:
                    report.append(f"✗ {comparison_name}: Methods disagree")
                    report.append(f"  - Details: {agreement_data}")
            report.append("")
        
        # Recommendations
        if comparison['recommendations']:
            report.append("## Recommendations")
            for rec in comparison['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Save report
        report_content = '\n'.join(report)
        
        from pathlib import Path
        output_file = Path(f'results/{self.model.name}_comparison_report.md')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        return {
            'report_file': str(output_file),
            'report_content': report_content,
            'comparison_summary': comparison
        }
```

### 2. Sensitivity Analysis Workflow

```python
# sensitivity_workflow.py
"""Workflow for comprehensive sensitivity analysis."""

class SensitivityAnalysisWorkflow(WorkflowBase):
    """Comprehensive parameter sensitivity analysis."""
    
    def __init__(self, model, options, sensitivity_config=None):
        super().__init__("sensitivity_analysis", model, options)
        self.sensitivity_config = sensitivity_config or self._default_sensitivity_config()
    
    def _default_sensitivity_config(self):
        """Default sensitivity analysis configuration."""
        return {
            'perturbation_levels': [0.01, 0.05, 0.1, 0.2],  # 1%, 5%, 10%, 20%
            'parameters_to_test': 'all',  # or list of specific parameters
            'metrics': ['nullspace_dimension', 'identifiable_count', 'condition_number'],
            'monte_carlo_samples': 100
        }
    
    def define_workflow(self):
        """Define sensitivity analysis workflow."""
        
        # Step 1: Baseline analysis
        self.add_step(WorkflowStep(
            name="baseline_analysis",
            function=self._run_baseline_analysis,
            critical=True
        ))
        
        # Step 2: Parameter perturbation analysis
        self.add_step(WorkflowStep(
            name="parameter_perturbation",
            function=self._run_parameter_perturbation,
            dependencies=["baseline_analysis"],
            timeout=1200.0  # 20 minutes
        ))
        
        # Step 3: Monte Carlo sensitivity
        self.add_step(WorkflowStep(
            name="monte_carlo_sensitivity",
            function=self._run_monte_carlo_sensitivity,
            dependencies=["baseline_analysis"],
            timeout=1800.0  # 30 minutes
        ))
        
        # Step 4: Analyze sensitivity patterns
        self.add_step(WorkflowStep(
            name="analyze_sensitivity",
            function=self._analyze_sensitivity_patterns,
            dependencies=["parameter_perturbation", "monte_carlo_sensitivity"]
        ))
        
        # Step 5: Generate sensitivity report
        self.add_step(WorkflowStep(
            name="generate_sensitivity_report",
            function=self._generate_sensitivity_report,
            dependencies=["analyze_sensitivity"]
        ))
    
    def _run_baseline_analysis(self, model, options, **kwargs):
        """Run baseline analysis for comparison."""
        from nullstrike.cli.complete_analysis import main as analyze
        
        baseline_result = analyze(model.name, parameters_only=True)
        
        return {
            'nullspace_dimension': baseline_result.nullspace_results.nullspace_basis.shape[1],
            'identifiable_count': len(baseline_result.strike_goldd_results.identifiable_parameters),
            'observability_rank': baseline_result.nullspace_results.observability_rank,
            'computation_time': baseline_result.computation_time,
            'parameter_values': self._extract_parameter_values(model)
        }
    
    def _run_parameter_perturbation(self, model, options, previous_results, **kwargs):
        """Run parameter perturbation sensitivity analysis."""
        
        baseline = previous_results['baseline_analysis']['result']
        parameter_values = baseline['parameter_values']
        
        sensitivity_results = {}
        
        for param_name, base_value in parameter_values.items():
            if self.sensitivity_config['parameters_to_test'] != 'all':
                if param_name not in self.sensitivity_config['parameters_to_test']:
                    continue
            
            param_sensitivity = {}
            
            for perturbation in self.sensitivity_config['perturbation_levels']:
                # Test both positive and negative perturbations
                for direction in [1, -1]:
                    perturbed_value = base_value * (1 + direction * perturbation)
                    
                    try:
                        # Create perturbed model
                        perturbed_model = self._create_perturbed_model(
                            model, param_name, perturbed_value
                        )
                        
                        # Run analysis
                        result = self._analyze_perturbed_model(perturbed_model, options)
                        
                        key = f"{perturbation:g}{'_pos' if direction > 0 else '_neg'}"
                        param_sensitivity[key] = result
                        
                    except Exception as e:
                        self.logger.warning(f"Perturbation failed for {param_name} "
                                          f"({perturbation:g}, {direction}): {e}")
            
            sensitivity_results[param_name] = param_sensitivity
        
        return sensitivity_results
    
    def _run_monte_carlo_sensitivity(self, model, options, previous_results, **kwargs):
        """Run Monte Carlo sensitivity analysis."""
        
        baseline = previous_results['baseline_analysis']['result']
        parameter_values = baseline['parameter_values']
        
        n_samples = self.sensitivity_config['monte_carlo_samples']
        mc_results = []
        
        for i in range(n_samples):
            try:
                # Generate random parameter perturbations
                perturbed_params = {}
                for param_name, base_value in parameter_values.items():
                    # Use log-normal distribution for parameter perturbations
                    import numpy as np
                    perturbation_factor = np.random.lognormal(0, 0.1)  # 10% std deviation
                    perturbed_params[param_name] = base_value * perturbation_factor
                
                # Create and analyze perturbed model
                perturbed_model = self._create_perturbed_model_multiple(
                    model, perturbed_params
                )
                
                result = self._analyze_perturbed_model(perturbed_model, options)
                result['sample_id'] = i
                result['parameter_perturbations'] = perturbed_params
                
                mc_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Monte Carlo sample {i} failed: {e}")
        
        return {
            'samples': mc_results,
            'successful_samples': len(mc_results),
            'total_samples': n_samples,
            'success_rate': len(mc_results) / n_samples
        }
    
    def _analyze_sensitivity_patterns(self, previous_results, **kwargs):
        """Analyze sensitivity patterns from perturbation and Monte Carlo results."""
        
        baseline = previous_results['baseline_analysis']['result']
        perturbation_results = previous_results['parameter_perturbation']['result']
        mc_results = previous_results['monte_carlo_sensitivity']['result']
        
        analysis = {
            'parameter_sensitivity_ranking': {},
            'robust_metrics': {},
            'fragile_metrics': {},
            'correlation_analysis': {},
            'recommendations': []
        }
        
        # Analyze parameter perturbation sensitivity
        for param_name, param_results in perturbation_results.items():
            sensitivity_scores = []
            
            for perturbation_key, result in param_results.items():
                if 'nullspace_dimension' in result:
                    # Calculate sensitivity score
                    baseline_value = baseline['nullspace_dimension']
                    perturbed_value = result['nullspace_dimension']
                    
                    if baseline_value > 0:
                        sensitivity = abs(perturbed_value - baseline_value) / baseline_value
                        sensitivity_scores.append(sensitivity)
            
            if sensitivity_scores:
                analysis['parameter_sensitivity_ranking'][param_name] = {
                    'mean_sensitivity': np.mean(sensitivity_scores),
                    'max_sensitivity': np.max(sensitivity_scores),
                    'sensitivity_scores': sensitivity_scores
                }
        
        # Analyze Monte Carlo results for robustness
        if mc_results['samples']:
            mc_samples = mc_results['samples']
            
            # Calculate coefficient of variation for each metric
            for metric in ['nullspace_dimension', 'identifiable_count']:
                values = [sample[metric] for sample in mc_samples if metric in sample]
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / mean_val if mean_val > 0 else float('inf')
                    
                    if cv < 0.1:  # CV < 10% considered robust
                        analysis['robust_metrics'][metric] = {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv
                        }
                    else:
                        analysis['fragile_metrics'][metric] = {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv
                        }
        
        # Generate recommendations
        if analysis['parameter_sensitivity_ranking']:
            most_sensitive = max(analysis['parameter_sensitivity_ranking'].keys(),
                               key=lambda p: analysis['parameter_sensitivity_ranking'][p]['mean_sensitivity'])
            analysis['recommendations'].append(f"Most sensitive parameter: {most_sensitive}")
        
        if analysis['fragile_metrics']:
            analysis['recommendations'].append(
                f"Fragile metrics detected: {list(analysis['fragile_metrics'].keys())} - "
                "results may be unreliable"
            )
        
        if analysis['robust_metrics']:
            analysis['recommendations'].append(
                f"Robust metrics: {list(analysis['robust_metrics'].keys())} - "
                "results are reliable across parameter variations"
            )
        
        return analysis
    
    def _create_perturbed_model(self, model, param_name, new_value):
        """Create model with single parameter perturbation."""
        # This is a placeholder - real implementation would need
        # to modify the model's parameter values and regenerate expressions
        return model
    
    def _create_perturbed_model_multiple(self, model, param_dict):
        """Create model with multiple parameter perturbations."""
        # This is a placeholder - real implementation would need
        # to modify multiple parameter values
        return model
    
    def _analyze_perturbed_model(self, perturbed_model, options):
        """Analyze perturbed model and return key metrics."""
        from nullstrike.cli.complete_analysis import main as analyze
        
        result = analyze(perturbed_model.name, parameters_only=True)
        
        return {
            'nullspace_dimension': result.nullspace_results.nullspace_basis.shape[1],
            'identifiable_count': len(result.strike_goldd_results.identifiable_parameters),
            'observability_rank': result.nullspace_results.observability_rank,
            'computation_time': result.computation_time
        }
    
    def _extract_parameter_values(self, model):
        """Extract current parameter values from model."""
        # This is a placeholder - real implementation would extract
        # actual parameter values from the model definition
        parameter_values = {}
        for i, param in enumerate(model.parameters):
            parameter_values[str(param)] = 1.0  # Default value
        return parameter_values
    
    def _generate_sensitivity_report(self, previous_results, **kwargs):
        """Generate comprehensive sensitivity analysis report."""
        
        analysis = previous_results['analyze_sensitivity']['result']
        
        report = []
        report.append("# Parameter Sensitivity Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Parameter ranking
        if analysis['parameter_sensitivity_ranking']:
            report.append("## Parameter Sensitivity Ranking")
            
            sorted_params = sorted(
                analysis['parameter_sensitivity_ranking'].items(),
                key=lambda x: x[1]['mean_sensitivity'],
                reverse=True
            )
            
            for i, (param, data) in enumerate(sorted_params):
                report.append(f"{i+1}. **{param}**: {data['mean_sensitivity']:.3f} "
                            f"(max: {data['max_sensitivity']:.3f})")
            report.append("")
        
        # Robustness analysis
        if analysis['robust_metrics']:
            report.append("## Robust Metrics")
            for metric, data in analysis['robust_metrics'].items():
                report.append(f"- **{metric}**: CV = {data['cv']:.3f} "
                            f"(mean: {data['mean']:.2f}, std: {data['std']:.2f})")
            report.append("")
        
        if analysis['fragile_metrics']:
            report.append("## Fragile Metrics")
            for metric, data in analysis['fragile_metrics'].items():
                report.append(f"- **{metric}**: CV = {data['cv']:.3f} "
                            f"(mean: {data['mean']:.2f}, std: {data['std']:.2f})")
            report.append("")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("## Recommendations")
            for rec in analysis['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        report_content = '\n'.join(report)
        
        from pathlib import Path
        output_file = Path(f'results/{self.model.name}_sensitivity_report.md')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        return {
            'report_file': str(output_file),
            'report_content': report_content,
            'sensitivity_analysis': analysis
        }
```

## Using Custom Workflows

### Command Line Usage

```python
# workflow_cli.py
"""Command line interface for custom workflows."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="NullStrike custom workflow runner")
    parser.add_argument('workflow', choices=['comparison', 'sensitivity', 'validation'])
    parser.add_argument('model', help='Model name to analyze')
    parser.add_argument('--config', help='Workflow configuration file')
    parser.add_argument('--output', default='workflow_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model and options
    from nullstrike import load_model, AnalysisOptions
    model = load_model(args.model)
    options = AnalysisOptions()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Run specified workflow
    if args.workflow == 'comparison':
        workflow = IdentifiabilityComparisonWorkflow(model, options)
    elif args.workflow == 'sensitivity':
        workflow = SensitivityAnalysisWorkflow(model, options)
    elif args.workflow == 'validation':
        workflow = ModelValidationWorkflow(model, options)
    else:
        raise ValueError(f"Unknown workflow: {args.workflow}")
    
    # Execute workflow
    results = workflow.execute()
    
    # Save results
    import json
    results_file = output_dir / f"{args.workflow}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'workflow_name': results.workflow_name,
            'success': results.success,
            'total_time': results.total_time,
            'errors': results.errors
        }, f, indent=2)
    
    print(f"Workflow completed: {results.success}")
    print(f"Results saved to: {output_dir}")
    
    return 0 if results.success else 1

if __name__ == '__main__':
    exit(main())
```

### Python API Usage

```python
# Example usage of custom workflows
from nullstrike import load_model, AnalysisOptions

# Load model
model = load_model('C2M')
options = AnalysisOptions(max_lie_time=300)

# Method comparison workflow
comparison_workflow = IdentifiabilityComparisonWorkflow(
    model, options, 
    methods=['strike_goldd', 'fisher_information']
)

comparison_results = comparison_workflow.execute()

if comparison_results.success:
    print("Comparison completed successfully!")
    
    # Access specific results
    comparison_data = comparison_results.step_results['compare_results']['result']
    
    print(f"Successful methods: {comparison_data['methods_successful']}")
    print(f"Failed methods: {comparison_data['methods_failed']}")
else:
    print("Comparison workflow failed!")
    print(f"Errors: {comparison_results.errors}")

# Sensitivity analysis workflow
sensitivity_workflow = SensitivityAnalysisWorkflow(model, options)
sensitivity_results = sensitivity_workflow.execute()

if sensitivity_results.success:
    print("Sensitivity analysis completed!")
    
    # Access sensitivity results
    sensitivity_data = sensitivity_results.step_results['analyze_sensitivity']['result']
    
    print("Parameter sensitivity ranking:")
    for param, data in sensitivity_data['parameter_sensitivity_ranking'].items():
        print(f"  {param}: {data['mean_sensitivity']:.3f}")
```

This comprehensive workflow system provides a flexible framework for creating sophisticated analysis pipelines that combine multiple methods, perform systematic parameter studies, and generate detailed comparative reports.

---

## Next Steps

1. **Start with simple workflows** using the base framework
2. **Customize workflow steps** for your specific analysis needs
3. **Combine multiple methods** for comprehensive analysis
4. **Develop domain-specific workflows** for your research area
5. **Explore [Mathematical Extensions](extensions.md)** for advanced algorithms