"""
StrikePy Core Functionality

This module contains the original StrikePy implementation by David Rey Rostro,
which is a Python implementation of the STRIKE-GOLDD algorithm originally 
developed by Alejandro Fernandez Villaverde.

Original StrikePy License: GPL-3.0
Original Author: David Rey Rostro (davidreyrostro@gmail.com)
Based on STRIKE-GOLDD by: Alejandro Fernandez Villaverde (afvillaverde@uvigo.gal)

For full attribution see ATTRIBUTION.md in the project root.
"""

from .strike_goldd import strike_goldd

__all__ = ['strike_goldd']