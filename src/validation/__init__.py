"""
Validation module for dipole selection algorithms.

Compares forward model predictions against real FEM solver simulations.
"""

from . import comparison
from . import plotting
from . import report_generator

__all__ = ['comparison', 'plotting', 'report_generator']
