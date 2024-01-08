from __future__ import absolute_import

import os

from ._version import get_versions

# Import the version

#
#

if os.environ.get("ASTROMODELS_DEBUG", None) is None:
    #
    from .core.serialization import *
    from .core.units import get_units
    from .functions import has_atomdb, has_ebltable, has_gsl, has_naima

    if has_ebltable:
        pass

    if has_gsl:
        pass

    if has_naima:
        pass

    if has_atomdb:
        pass

    astromodels_units = get_units()
    from astromodels.utils.logging import setup_logger


log = setup_logger(__name__)

__version__ = get_versions()["version"]
del get_versions
