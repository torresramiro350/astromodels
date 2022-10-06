import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import astropy.units as astropy_units
import numba as nb
import numpy as np

from astromodels.functions.function import Function1D, FunctionMeta
from astromodels.utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass(frozen=True)
class _ExtinctionCurve:
    """
    simple extinction container
    """

    a: np.ndarray
    lamb: np.ndarray
    b: np.ndarray
    n: np.ndarray


class ZDust(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Extinction by dust grains from Pei (1992), suitable for IR, optical and UV energy bands, including the full energy ranges of the Swift UVOT and XMM-Newton OM detectors. Three models are included which characterize the extinction curves of (1) the Milky Way, (2) the LMC and (3) the SMC. The models can be modified by redshift and can therefore be applied to extragalactic sources. The transmission is set to unity shortward of 912 Angstroms in the rest frame of the dust. This is incorrect physically but does allow the model to be used in combination with an X-ray photoelectric absorption model such as phabs. Parameter 1 (method) describes which extinction curve (MW, LMC or SMC) will be constructed and should never be allowed to float during a fit. The extinction at V, A(V) = E(B-V) x Rv. Rv should typically remain frozen for a fit. Standard values for Rv are MW = 3.08, LMC = 3.16 and SMC = 2.93 (from table 2 of Pei 1992), although these may not be applicable to more distant dusty sources.
    parameters :
        e_bmv :
            desc : color excess
            initial value : 1.0
            min: 0
            is_normalization : False
            delta : 0.1

        rv :
            desc :  ratio of total to selective extinction
            initial value : 3.08
            is_normalization : False
            delta : 0.1
            fix: True

        redshift :
            desc : the redshift of the source
            initial value : 0.
            is_normalization : False
            min : 0
            max : 15
            delta : 0.1
            fix: True

    properties:
        extinction_law:
            desc: the abundance table for the model
            initial value: mw
            allowed values:
            - mw
            - lmc
            - smc

            function: _set_extinction_law



    """

    def _setup(self):
        self._fixed_units = (
            astropy_units.keV,
            astropy_units.dimensionless_unscaled,
        )

    def _set_units(self, x_unit, y_unit):
        self.rv.unit = astropy_units.dimensionless_unscaled
        self.e_bmv.unit = astropy_units.dimensionless_unscaled
        self.redshift.unit = astropy_units.dimensionless_unscaled

    def _set_extinction_law(self) -> None:

        self._extinction_law: _ExtinctionCurve = _extinctions_laws[
            self.extinction_law.value
        ]

    def evaluate(self, x, e_bmv, rv, redshift):

        if isinstance(x, astropy_units.Quantity):

            _x = np.array(x.to("keV").value, ndmin=1, copy=False, dtype=float)

            _unit = astropy_units.cm**2
            _y_unit = astropy_units.dimensionless_unscaled

            _e_bmv = e_bmv.value
            _rv = rv.value
            _redshift = redshift.value

        else:

            _unit = 1.0
            _y_unit = 1.0
            _redshift = redshift
            _e_bmv = e_bmv
            _rv = rv
            _x = x

        # _x = np.append(_x, _x[-1]* 1.01)

        zfac = _redshift + 1.0

        return (
            ms_dust(
                _x * zfac,
                _e_bmv,
                _rv,
                self._extinction_law.a,
                self._extinction_law.lamb,
                self._extinction_law.b,
                self._extinction_law.n,
            )
            * _y_unit
        )


@nb.njit(fastmath=True)
def ms_dust(x, e_bmv, rv, a, lamb, b, n):

    # conversion constant in keV*\AA

    hc = 12.3963

    # extinction at B (a_b)

    a_b = e_bmv * (1 + rv)

    ne = len(x)

    xx = hc / x

    out = np.zeros(ne)

    for i in range(ne):

        out[i] = pei(xx[i], a_b, a, lamb, b, n)

    return out


@nb.njit(fastmath=True)
def ms_dust_xspec(x, e_bmv, rv, a, lamb, b, n):

    # conversion constant in keV*\AA

    hc = 12.3963

    #  photon index of the base power law function
    index = 2.0

    # extinction at B (a_b)

    a_b = rv * (1 + e_bmv)

    ne = len(x)

    out = np.zeros(ne)

    xx = hc / x

    xl = xx[0]
    sl = math.pow(xl, -index)
    fl = pei(xl, a_b, a, lamb, b, n)

    for i in range(ne):
        xh = xx[i + 1]
        sh = math.pow(xh, -index)
        fh = pei(xh, a_b, a, lamb, b, n)

        out[i] = (sl * fl + sh * fh) / (sl + sh)

        xl = xh
        sl = sh
        fl = fh

    return out


@nb.njit(fastmath=True)
def pei(rlambda, a_b, a, lamb, b, n) -> float:
    """
    ported from XSPEC originally by
    Martin.Still@gsfc.nasa.gov


    """

    if rlambda < 800.0:

        return 1.0

    # convert angstroms to microns

    lambda_mu = rlambda / 1.0e4

    # compute terms of sum

    ratio = lambda_mu / lamb

    term = np.power(ratio, n)

    inv_term = 1.0 / term

    bottom = term + inv_term + b

    xi = np.sum(a / bottom)

    # remove a_b normalization on the extinction curve

    a_lambda = a_b * xi

    # linearize extinction factor

    return math.pow(10.0, -a_lambda / 2.512)


class Standard_Rv(Enum):

    MW = 3.08
    LMC = 3.16
    SMC = 2.93


# create some frozen data classes to hold the extinction curves


_mw_extinction = _ExtinctionCurve(
    a=np.array([165.0, 14.0, 0.045, 0.002, 0.002, 0.012]),
    lamb=np.array([0.047, 0.08, 0.22, 9.7, 18.0, 25.0]),
    b=np.array([90.0, 4.0, -1.95, -1.95, -1.8, 0.0]),
    n=np.array([2.0, 6.5, 2.0, 2.0, 2.0, 2.0]),
)

_lmc_extinction = _ExtinctionCurve(
    a=np.array([175.0, 19.0, 0.023, 0.005, 0.062, 0.02]),
    lamb=np.array([0.046, 0.08, 0.22, 9.7, 18.0, 25.0]),
    b=np.array([90.0, 5.5, -1.95, -1.95, -1.8, 0.0]),
    n=np.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0]),
)


_smc_extinction = _ExtinctionCurve(
    a=np.array([185.0, 27.0, 0.005, 0.01, 0.012, 0.03]),
    lamb=np.array([0.042, 0.08, 0.22, 9.7, 18.0, 25.0]),
    b=np.array([90.0, 5.5, -1.95, -1.95, -1.8, 0.0]),
    n=np.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0]),
)


_extinctions_laws: Dict[str, _ExtinctionCurve] = dict(
    mw=_mw_extinction, lmc=_lmc_extinction, smc=_smc_extinction
)
