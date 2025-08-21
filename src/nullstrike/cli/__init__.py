"""
Command line interface for NullStrike analysis.
"""

from .complete_analysis import main, cli_main
from ..core import strike_goldd

__all__ = ['main', 'cli_main', 'strike_goldd']