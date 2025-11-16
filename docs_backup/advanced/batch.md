# Batch Processing and Automation

This guide covers advanced techniques for processing multiple models, automating analyses, and building efficient computational pipelines with NullStrike.

## Overview

Batch processing enables you to:

- Analyze multiple models systematically
- Compare different parameter configurations
- Process model variants efficiently
- Automate repetitive analysis tasks
- Generate comparative reports and visualizations

## Batch Analysis Strategies

### 1. Simple Batch Processing

```python
# simple_batch.py
"""Simple batch processing for multiple models."""

from nullstrike.cli.complete_analysis import main as analyze_model
from pathlib import Path
import time
import json

def simple_batch_analysis(models, output_dir="batch_results"):
    """Run analysis on multiple models sequentially."""
    
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for model_name in models:
        print(f"Analyzing {model_name}...")
        
        try:
            start_time = time.time()
            result = analyze_model(model_name, parameters_only=True)
            analysis_time = time.time() - start_time
            
            # Store results
            results[model_name] = {
                'success': True,
                'analysis_time': analysis_time,
                'nullspace_dimension': result.nullspace_results.nullspace_basis.shape[1],
                'identifiable_count': len(result.strike_goldd_results.identifiable_parameters),
                'observability_rank': result.nullspace_results.observability_rank
            }
            
            print(f"SUCCESS: {model_name} completed in {analysis_time:.1f}s")
            
        except Exception as e:
            results[model_name] = {
                'success': False,
                'error': str(e),
                'analysis_time': None
            }
            print(f"FAILED: {model_name} failed: {e}")
    
    # Save summary results
    with open(output_path / "batch_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage example
models = ['C2M', 'Bolie', 'calibration_single', 'calibration_double']
results = simple_batch_analysis(models)

# Print summary
successful = [m for m, r in results.items() if r['success']]
failed = [m for m, r in results.items() if not r['success']]

print(f"\nBatch Analysis Summary:")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
```

### 2. Advanced Batch Processing Class

```python
# advanced_batch.py
"""Advanced batch processing with sophisticated features."""

import concurrent.futures
import multiprocessing as mp
import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging

@dataclass
class BatchJob:
    """Individual batch job specification."""
    model_name: str
    options_file: Optional[str] = None
    custom_options: Optional[Dict[str, Any]] = None
    priority: int = 1
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 2

@dataclass
class BatchResult:
    """Results from a single batch job."""
    model_name: str
    success: bool
    analysis_time: float
    memory_peak: float
    error_message: Optional[str] = None
    nullspace_dimension: Optional[int] = None
    identifiable_count: Optional[int] = None
    observability_rank: Optional[int] = None
    parameter_combinations: Optional[List[str]] = None

class AdvancedBatchProcessor:
    """Advanced batch processing with parallel execution and monitoring."""
    
    def __init__(self, 
                 max_workers: int = None,
                 timeout_default: float = 300.0,
                 memory_limit_gb: float = 8.0,
                 output_dir: str = "batch_results"):
        
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.timeout_default = timeout_default
        self.memory_limit_gb = memory_limit_gb
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Track progress
        self.completed_jobs = 0
        self.total_jobs = 0
        self.start_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processing."""
        logger = logging.getLogger('batch_processor')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / 'batch_processing.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def process_batch(self, jobs: List[BatchJob]) -> List[BatchResult]:
        """Process a batch of jobs with parallel execution."""
        self.total_jobs = len(jobs)
        self.completed_jobs = 0
        self.start_time = time.time()
        
        self.logger.info(f"Starting batch processing: {self.total_jobs} jobs")
        self.logger.info(f"Using {self.max_workers} workers")
        
        # Sort jobs by priority (higher number = higher priority)
        sorted_jobs = sorted(jobs, key=lambda x: x.priority, reverse=True)
        
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._process_single_job, job): job 
                for job in sorted_jobs
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result(timeout=job.timeout)
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"SUCCESS: {job.model_name} completed in {result.analysis_time:.1f}s")
                    else:
                        self.logger.error(f"FAILED: {job.model_name} failed: {result.error_message}")
                        
                        # Retry if configured
                        if job.retry_count < job.max_retries:
                            self.logger.info(f"Retrying {job.model_name} (attempt {job.retry_count + 1})")
                            job.retry_count += 1
                            retry_future = executor.submit(self._process_single_job, job)
                            future_to_job[retry_future] = job
                
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"TIMEOUT: {job.model_name} timed out after {job.timeout}s")
                    results.append(BatchResult(
                        model_name=job.model_name,
                        success=False,
                        analysis_time=job.timeout,
                        memory_peak=0,
                        error_message="Timeout"
                    ))
                
                except Exception as e:
                    self.logger.error(f"CRASHED: {job.model_name} crashed: {str(e)}")
                    results.append(BatchResult(
                        model_name=job.model_name,
                        success=False,
                        analysis_time=0,
                        memory_peak=0,
                        error_message=str(e)
                    ))
                
                self.completed_jobs += 1
                self._log_progress()
        
        # Save results and generate reports
        self._save_batch_results(results)
        self._generate_batch_report(results)
        
        return results
    
    def _process_single_job(self, job: BatchJob) -> BatchResult:
        """Process a single batch job."""
        import psutil
        import gc
        
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_time = time.time()
        
        try:
            # Import here to avoid issues with multiprocessing
            from nullstrike.cli.complete_analysis import main as analyze_model
            
            # Run analysis
            if job.options_file:
                result = analyze_model(job.model_name, job.options_file, parameters_only=True)
            else:
                result = analyze_model(job.model_name, parameters_only=True)
            
            # Collect results
            end_time = time.time()
            peak_memory = process.memory_info().rss
            
            # Extract key metrics
            nullspace_dim = result.nullspace_results.nullspace_basis.shape[1]
            identifiable_count = len(result.strike_goldd_results.identifiable_parameters)
            obs_rank = result.nullspace_results.observability_rank
            param_combos = [str(combo) for combo in result.nullspace_results.identifiable_combinations]
            
            # Clean up
            del result
            gc.collect()
            
            return BatchResult(
                model_name=job.model_name,
                success=True,
                analysis_time=end_time - start_time,
                memory_peak=(peak_memory - start_memory) / 1024**3,  # GB
                nullspace_dimension=nullspace_dim,
                identifiable_count=identifiable_count,
                observability_rank=obs_rank,
                parameter_combinations=param_combos
            )
            
        except Exception as e:
            end_time = time.time()
            current_memory = process.memory_info().rss
            
            return BatchResult(
                model_name=job.model_name,
                success=False,
                analysis_time=end_time - start_time,
                memory_peak=(current_memory - start_memory) / 1024**3,
                error_message=str(e)
            )
    
    def _log_progress(self):
        """Log current progress."""
        if self.start_time and self.total_jobs > 0:
            elapsed = time.time() - self.start_time
            progress = self.completed_jobs / self.total_jobs
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            self.logger.info(
                f"Progress: {self.completed_jobs}/{self.total_jobs} "
                f"({progress*100:.1f}%) - "
                f"Elapsed: {elapsed:.1f}s, "
                f"Estimated remaining: {remaining:.1f}s"
            )
    
    def _save_batch_results(self, results: List[BatchResult]):
        """Save batch results to files."""
        # Save as JSON
        results_dict = [asdict(result) for result in results]
        with open(self.output_dir / 'batch_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(results_dict)
        df.to_csv(self.output_dir / 'batch_results.csv', index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_batch_report(self, results: List[BatchResult]):
        """Generate comprehensive batch report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        report = []
        report.append("# Batch Processing Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## Summary")
        report.append(f"- Total jobs: {len(results)}")
        report.append(f"- Successful: {len(successful)}")
        report.append(f"- Failed: {len(failed)}")
        report.append(f"- Success rate: {len(successful)/len(results)*100:.1f}%")
        report.append("")
        
        if successful:
            # Performance statistics
            times = [r.analysis_time for r in successful]
            memories = [r.memory_peak for r in successful if r.memory_peak]
            
            report.append("## Performance Statistics")
            report.append(f"- Average analysis time: {sum(times)/len(times):.1f}s")
            report.append(f"- Fastest analysis: {min(times):.1f}s")
            report.append(f"- Slowest analysis: {max(times):.1f}s")
            
            if memories:
                report.append(f"- Average memory usage: {sum(memories)/len(memories):.2f} GB")
                report.append(f"- Peak memory usage: {max(memories):.2f} GB")
            report.append("")
            
            # Analysis results summary
            nullspace_dims = [r.nullspace_dimension for r in successful if r.nullspace_dimension is not None]
            identifiable_counts = [r.identifiable_count for r in successful if r.identifiable_count is not None]
            
            if nullspace_dims:
                report.append("## Analysis Results Summary")
                report.append(f"- Average nullspace dimension: {sum(nullspace_dims)/len(nullspace_dims):.1f}")
                report.append(f"- Average identifiable parameters: {sum(identifiable_counts)/len(identifiable_counts):.1f}")
                report.append("")
        
        # Failed jobs
        if failed:
            report.append("## Failed Jobs")
            for result in failed:
                report.append(f"- {result.model_name}: {result.error_message}")
            report.append("")
        
        # Top performers
        if successful:
            report.append("## Top Performers")
            fastest = sorted(successful, key=lambda x: x.analysis_time)[:3]
            for i, result in enumerate(fastest):
                report.append(f"{i+1}. {result.model_name}: {result.analysis_time:.1f}s")
            report.append("")
        
        # Save report
        with open(self.output_dir / 'batch_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info("Batch report generated")
```

### 3. Comparative Analysis Workflows

```python
# comparative_analysis.py
"""Comparative analysis workflows for systematic model comparison."""

from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ComparativeAnalysisWorkflow:
    """Workflow for comparing multiple models or configurations."""
    
    def __init__(self, output_dir: str = "comparative_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def compare_models(self, models: List[str], 
                      configurations: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models across different configurations."""
        
        configurations = configurations or ['default']
        
        # Set up batch jobs
        batch_processor = AdvancedBatchProcessor(output_dir=self.output_dir / "batch_data")
        jobs = []
        
        for model in models:
            for config in configurations:
                job = BatchJob(
                    model_name=model,
                    options_file=config if config != 'default' else None,
                    priority=1
                )
                jobs.append(job)
        
        # Run batch analysis
        results = batch_processor.process_batch(jobs)
        
        # Organize results by model and configuration
        organized_results = self._organize_comparative_results(results, models, configurations)
        
        # Generate comparative visualizations
        self._generate_comparative_plots(organized_results)
        
        # Generate comparative report
        self._generate_comparative_report(organized_results)
        
        return organized_results
    
    def parameter_sensitivity_study(self, model: str, 
                                  parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform parameter sensitivity analysis."""
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        # Create batch jobs for each combination
        jobs = []
        for i, params in enumerate(param_combinations):
            # Create custom options for this parameter set
            custom_options = self._create_custom_options(params)
            
            job = BatchJob(
                model_name=model,
                custom_options=custom_options,
                priority=1
            )
            jobs.append(job)
        
        # Run sensitivity analysis
        batch_processor = AdvancedBatchProcessor(output_dir=self.output_dir / "sensitivity_data")
        results = batch_processor.process_batch(jobs)
        
        # Analyze sensitivity
        sensitivity_analysis = self._analyze_parameter_sensitivity(results, param_combinations)
        
        return sensitivity_analysis
    
    def model_size_scaling_study(self, base_model: str, 
                                scale_factors: List[int]) -> Dict[str, Any]:
        """Study how analysis scales with model size."""
        
        # Generate scaled models
        scaled_models = self._generate_scaled_models(base_model, scale_factors)
        
        # Run scaling analysis
        jobs = [BatchJob(model_name=model, priority=1) for model in scaled_models]
        
        batch_processor = AdvancedBatchProcessor(output_dir=self.output_dir / "scaling_data")
        results = batch_processor.process_batch(jobs)
        
        # Analyze scaling behavior
        scaling_analysis = self._analyze_scaling_behavior(results, scale_factors)
        
        return scaling_analysis
    
    def _organize_comparative_results(self, results: List[BatchResult], 
                                    models: List[str], 
                                    configurations: List[str]) -> Dict[str, Any]:
        """Organize batch results for comparative analysis."""
        organized = {
            'models': models,
            'configurations': configurations,
            'data': {},
            'summary': {}
        }
        
        # Group results by model and configuration
        for result in results:
            model = result.model_name
            # Extract configuration from model name or use default logic
            config = 'default'  # Simplified - would need more sophisticated mapping
            
            if model not in organized['data']:
                organized['data'][model] = {}
            
            organized['data'][model][config] = result
        
        # Calculate summary statistics
        organized['summary'] = self._calculate_comparative_summary(organized['data'])
        
        return organized
    
    def _generate_comparative_plots(self, organized_results: Dict[str, Any]):
        """Generate comparative visualization plots."""
        
        # Extract data for plotting
        plot_data = []
        for model, configs in organized_results['data'].items():
            for config, result in configs.items():
                if result.success:
                    plot_data.append({
                        'model': model,
                        'configuration': config,
                        'analysis_time': result.analysis_time,
                        'memory_usage': result.memory_peak,
                        'nullspace_dimension': result.nullspace_dimension,
                        'identifiable_count': result.identifiable_count,
                        'observability_rank': result.observability_rank
                    })
        
        df = pd.DataFrame(plot_data)
        
        if len(df) == 0:
            print("No successful results to plot")
            return
        
        # Performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Analysis time comparison
        sns.barplot(data=df, x='model', y='analysis_time', hue='configuration', ax=axes[0,0])
        axes[0,0].set_title('Analysis Time Comparison')
        axes[0,0].set_ylabel('Time (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        if 'memory_usage' in df.columns and df['memory_usage'].notna().any():
            sns.barplot(data=df, x='model', y='memory_usage', hue='configuration', ax=axes[0,1])
            axes[0,1].set_title('Memory Usage Comparison')
            axes[0,1].set_ylabel('Memory (GB)')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Nullspace dimension comparison
        if 'nullspace_dimension' in df.columns:
            sns.barplot(data=df, x='model', y='nullspace_dimension', hue='configuration', ax=axes[1,0])
            axes[1,0].set_title('Nullspace Dimension Comparison')
            axes[1,0].set_ylabel('Nullspace Dimension')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Identifiable parameters comparison
        if 'identifiable_count' in df.columns:
            sns.barplot(data=df, x='model', y='identifiable_count', hue='configuration', ax=axes[1,1])
            axes[1,1].set_title('Identifiable Parameters Comparison')
            axes[1,1].set_ylabel('Count')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance vs accuracy scatter plot
        if len(df) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(df['analysis_time'], df['identifiable_count'], 
                               c=df['nullspace_dimension'], s=df['observability_rank']*10,
                               alpha=0.7, cmap='viridis')
            
            ax.set_xlabel('Analysis Time (seconds)')
            ax.set_ylabel('Identifiable Parameters')
            ax.set_title('Performance vs Accuracy\n(Color: Nullspace Dim, Size: Obs. Rank)')
            
            # Add model labels
            for i, row in df.iterrows():
                ax.annotate(row['model'], (row['analysis_time'], row['identifiable_count']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.colorbar(scatter, label='Nullspace Dimension')
            plt.savefig(self.output_dir / 'performance_vs_accuracy.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_comparative_report(self, organized_results: Dict[str, Any]):
        """Generate comprehensive comparative report."""
        
        report = []
        report.append("# Comparative Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Models analyzed
        report.append("## Models Analyzed")
        for model in organized_results['models']:
            report.append(f"- {model}")
        report.append("")
        
        # Summary table
        report.append("## Results Summary")
        report.append("")
        report.append("| Model | Config | Time (s) | Memory (GB) | Nullspace Dim | Identifiable |")
        report.append("|-------|--------|----------|-------------|---------------|--------------|")
        
        for model, configs in organized_results['data'].items():
            for config, result in configs.items():
                if result.success:
                    report.append(f"| {model} | {config} | {result.analysis_time:.1f} | "
                                f"{result.memory_peak:.2f} | {result.nullspace_dimension} | "
                                f"{result.identifiable_count} |")
                else:
                    report.append(f"| {model} | {config} | FAILED | - | - | - |")
        
        report.append("")
        
        # Best performers
        successful_results = []
        for model, configs in organized_results['data'].items():
            for config, result in configs.items():
                if result.success:
                    successful_results.append((model, config, result))
        
        if successful_results:
            report.append("## Performance Rankings")
            
            # Fastest analysis
            fastest = min(successful_results, key=lambda x: x[2].analysis_time)
            report.append(f"- **Fastest Analysis**: {fastest[0]} ({fastest[1]}) - {fastest[2].analysis_time:.1f}s")
            
            # Most memory efficient
            most_efficient = min(successful_results, key=lambda x: x[2].memory_peak)
            report.append(f"- **Most Memory Efficient**: {most_efficient[0]} ({most_efficient[1]}) - {most_efficient[2].memory_peak:.2f} GB")
            
            # Most identifiable parameters
            most_identifiable = max(successful_results, key=lambda x: x[2].identifiable_count)
            report.append(f"- **Most Identifiable Parameters**: {most_identifiable[0]} ({most_identifiable[1]}) - {most_identifiable[2].identifiable_count} parameters")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if successful_results:
            avg_time = sum(r[2].analysis_time for r in successful_results) / len(successful_results)
            if avg_time > 300:  # 5 minutes
                report.append("- Consider using --parameters-only mode for development")
                report.append("- Enable checkpointing for long computations")
            
            avg_memory = sum(r[2].memory_peak for r in successful_results) / len(successful_results)
            if avg_memory > 4:  # 4 GB
                report.append("- Consider memory optimization techniques for large models")
                report.append("- Use streaming computation for very large systems")
        
        # Save report
        with open(self.output_dir / 'comparative_report.md', 'w') as f:
            f.write('\n'.join(report))
```

## Automation Scripts

### 1. Automated Model Discovery and Analysis

```python
# automated_discovery.py
"""Automated model discovery and analysis pipeline."""

import os
import glob
from pathlib import Path
import json
import argparse

class AutomatedAnalysisPipeline:
    """Automated pipeline for discovering and analyzing models."""
    
    def __init__(self, model_directories: List[str], 
                 output_dir: str = "automated_analysis"):
        self.model_directories = [Path(d) for d in model_directories]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def discover_models(self) -> List[str]:
        """Discover all models in specified directories."""
        discovered_models = []
        
        for directory in self.model_directories:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist")
                continue
            
            # Find Python model files
            model_files = glob.glob(str(directory / "*.py"))
            
            for model_file in model_files:
                model_name = Path(model_file).stem
                
                # Skip __init__ and other non-model files
                if model_name.startswith('__') or model_name.startswith('options_'):
                    continue
                
                # Validate that it's a valid model
                if self._validate_model_file(model_file):
                    discovered_models.append(model_name)
                else:
                    print(f"Skipping invalid model: {model_name}")
        
        return discovered_models
    
    def _validate_model_file(self, model_file: str) -> bool:
        """Validate that a file contains a valid NullStrike model."""
        try:
            with open(model_file, 'r') as f:
                content = f.read()
            
            # Check for required model components
            required_components = ['x', 'p', 'f', 'h']  # states, params, dynamics, outputs
            
            for component in required_components:
                if f"{component} =" not in content and f"{component}=" not in content:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def run_automated_analysis(self, 
                             filter_patterns: List[str] = None,
                             exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Run automated analysis on discovered models."""
        
        # Discover models
        all_models = self.discover_models()
        print(f"Discovered {len(all_models)} models")
        
        # Apply filters
        filtered_models = self._apply_filters(all_models, filter_patterns, exclude_patterns)
        print(f"Analyzing {len(filtered_models)} models after filtering")
        
        # Set up batch processing
        batch_processor = AdvancedBatchProcessor(
            output_dir=self.output_dir / "batch_data",
            max_workers=4  # Conservative for automated runs
        )
        
        # Create jobs
        jobs = []
        for model in filtered_models:
            job = BatchJob(
                model_name=model,
                priority=1,
                timeout=600.0,  # 10 minutes per model
                max_retries=1
            )
            jobs.append(job)
        
        # Run analysis
        results = batch_processor.process_batch(jobs)
        
        # Generate automated report
        self._generate_automated_report(results, filtered_models)
        
        return {
            'discovered_models': all_models,
            'analyzed_models': filtered_models,
            'results': results
        }
    
    def _apply_filters(self, models: List[str], 
                      include_patterns: List[str] = None,
                      exclude_patterns: List[str] = None) -> List[str]:
        """Apply include/exclude filters to model list."""
        
        filtered = models.copy()
        
        # Apply include filters
        if include_patterns:
            filtered = [m for m in filtered 
                       if any(pattern in m for pattern in include_patterns)]
        
        # Apply exclude filters
        if exclude_patterns:
            filtered = [m for m in filtered 
                       if not any(pattern in m for pattern in exclude_patterns)]
        
        return filtered
    
    def _generate_automated_report(self, results: List[BatchResult], 
                                 analyzed_models: List[str]):
        """Generate report for automated analysis."""
        
        report = []
        report.append("# Automated Analysis Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        report.append("## Summary")
        report.append(f"- Models analyzed: {len(analyzed_models)}")
        report.append(f"- Successful analyses: {len(successful)}")
        report.append(f"- Failed analyses: {len(failed)}")
        report.append(f"- Success rate: {len(successful)/len(results)*100:.1f}%")
        report.append("")
        
        # Quick statistics
        if successful:
            times = [r.analysis_time for r in successful]
            report.append("## Performance Overview")
            report.append(f"- Total analysis time: {sum(times):.1f}s")
            report.append(f"- Average time per model: {sum(times)/len(times):.1f}s")
            report.append(f"- Fastest analysis: {min(times):.1f}s")
            report.append(f"- Slowest analysis: {max(times):.1f}s")
            report.append("")
        
        # Model categories
        if successful:
            # Categorize by nullspace dimension
            nullspace_categories = {}
            for result in successful:
                dim = result.nullspace_dimension
                if dim not in nullspace_categories:
                    nullspace_categories[dim] = []
                nullspace_categories[dim].append(result.model_name)
            
            report.append("## Model Categories by Nullspace Dimension")
            for dim in sorted(nullspace_categories.keys()):
                models = nullspace_categories[dim]
                report.append(f"- Dimension {dim}: {len(models)} models")
                for model in models[:5]:  # Show first 5
                    report.append(f"  - {model}")
                if len(models) > 5:
                    report.append(f"  - ... and {len(models)-5} more")
            report.append("")
        
        # Failed models
        if failed:
            report.append("## Failed Analyses")
            for result in failed:
                report.append(f"- {result.model_name}: {result.error_message}")
            report.append("")
        
        # Save report
        with open(self.output_dir / 'automated_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        # Save machine-readable summary
        summary = {
            'timestamp': time.time(),
            'analyzed_models': analyzed_models,
            'success_count': len(successful),
            'failure_count': len(failed),
            'total_time': sum(r.analysis_time for r in successful),
            'successful_models': [r.model_name for r in successful],
            'failed_models': [r.model_name for r in failed]
        }
        
        with open(self.output_dir / 'automated_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

# Command-line interface for automated analysis
def main():
    parser = argparse.ArgumentParser(description="Automated NullStrike model analysis")
    parser.add_argument('directories', nargs='+', 
                       help='Directories to search for models')
    parser.add_argument('--output', '-o', default='automated_analysis',
                       help='Output directory')
    parser.add_argument('--include', nargs='+', 
                       help='Include patterns for model names')
    parser.add_argument('--exclude', nargs='+',
                       help='Exclude patterns for model names')
    
    args = parser.parse_args()
    
    # Run automated analysis
    pipeline = AutomatedAnalysisPipeline(args.directories, args.output)
    results = pipeline.run_automated_analysis(args.include, args.exclude)
    
    print(f"\nAutomated analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"Analyzed {len(results['analyzed_models'])} models")

if __name__ == '__main__':
    main()
```

### 2. Continuous Integration Analysis

```bash
#!/bin/bash
# scripts/ci_analysis.sh
"""Continuous integration script for model analysis."""

set -e

# Configuration
MODELS_DIR="custom_models"
OUTPUT_DIR="ci_analysis_results"
MAX_TIME_PER_MODEL=300  # 5 minutes
EMAIL_RECIPIENTS="team@example.com"

echo "=== NullStrike CI Analysis ==="
echo "Timestamp: $(date)"
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run automated analysis
python3 automated_discovery.py "$MODELS_DIR" \
    --output "$OUTPUT_DIR" \
    --exclude "test_" "debug_" "old_"

# Check results
RESULTS_FILE="$OUTPUT_DIR/automated_summary.json"

if [ -f "$RESULTS_FILE" ]; then
    # Extract statistics
    SUCCESS_COUNT=$(python3 -c "import json; print(json.load(open('$RESULTS_FILE'))['success_count'])")
    FAILURE_COUNT=$(python3 -c "import json; print(json.load(open('$RESULTS_FILE'))['failure_count'])")
    TOTAL_COUNT=$((SUCCESS_COUNT + FAILURE_COUNT))
    
    echo "Analysis Results:"
    echo "- Total models: $TOTAL_COUNT"
    echo "- Successful: $SUCCESS_COUNT"
    echo "- Failed: $FAILURE_COUNT"
    
    # Check for failures
    if [ "$FAILURE_COUNT" -gt 0 ]; then
        echo "WARNING: $FAILURE_COUNT models failed analysis"
        
        # Send notification if configured
        if [ -n "$EMAIL_RECIPIENTS" ]; then
            echo "Sending failure notification..."
            python3 scripts/send_notification.py \
                --subject "NullStrike CI: $FAILURE_COUNT model failures" \
                --recipients "$EMAIL_RECIPIENTS" \
                --results "$RESULTS_FILE"
        fi
        
        exit 1
    else
        echo "All models analyzed successfully!"
    fi
else
    echo "ERROR: Results file not found"
    exit 1
fi

# Archive results
ARCHIVE_NAME="ci_analysis_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE_NAME" "$OUTPUT_DIR"
echo "Results archived as: $ARCHIVE_NAME"

echo "CI analysis complete!"
```

This comprehensive batch processing and automation guide provides the tools and workflows needed to efficiently analyze multiple models, compare different configurations, and automate repetitive analysis tasks with NullStrike. The combination of parallel processing, sophisticated error handling, and comprehensive reporting makes it suitable for both research and production environments.

---

## Next Steps

1. **Start with simple batch processing** using the provided examples
2. **Explore parallel processing** for faster analysis of multiple models
3. **Set up automated pipelines** for continuous model validation
4. **Customize comparative analysis** for your specific research needs
5. **Study [Advanced Workflows](workflows.md)** for complex analysis patterns