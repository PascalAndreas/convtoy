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
from .nephilim import Nephilim
from .thrones import Thrones
from .dominions import Dominions
from .powers import Powers
from .metatron import Metatron
from .leviathan import Leviathan
from .abaddon import Abaddon
from .pandemonium import Pandemonium
from .janus import Janus
from .azazel import Azazel
from .eidolon import Eidolon

__all__ = ['Soul', 'Seraphim', 'Imp', 'Ophanim', 'Cherubim', 'Nephilim',
           'Thrones', 'Dominions', 'Powers', 'Metatron', 'Leviathan',
           'Abaddon', 'Pandemonium', 'Janus', 'Azazel', 'Eidolon']
