"""
Soul package - Convolution-based image processors.

This package provides various "souls" (convolution processors) that transform
images using different strategies and mythical personas.
"""

from .base import Soul
from .seraphim import Seraphim
from .imp import Imp
from .ophanim import Ophanim
from .cherubim import Cherubim

__all__ = ['Soul', 'Seraphim', 'Imp', 'Ophanim', 'Cherubim']
