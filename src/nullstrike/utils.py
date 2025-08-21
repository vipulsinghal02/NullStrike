"""
Utility functions for NullStrike package.
"""

from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    # This file is at src/nullstrike/utils.py, so go up 3 levels to get to project root
    return Path(__file__).parent.parent.parent

def get_results_dir():
    """Get the results directory path."""
    return get_project_root() / "results"

def get_checkpoints_dir():
    """Get the checkpoints directory path."""
    return get_project_root() / "checkpoints"