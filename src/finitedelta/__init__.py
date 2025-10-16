"""
finitedelta
====

A finite difference coefficient generator.
"""

from importlib.metadata import version

__all__ = ["get_coef1d", "get_coefnd", "get_partials", "grid_handler1d", "grid_handlernd"]
__version__ = version("finitedelta")

from .get_coef1d import get_coef1d
from .get_coefnd import get_coefnd
from .get_partials import get_partials
from .grid_handler1d import grid_handler1d
from .grid_handlernd import grid_handlernd
