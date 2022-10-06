from __future__ import print_function

try:
    from threeML import *
except Exception:
    pass

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pytest
from astropy import wcs
from astropy.coordinates import SkyCoord

from astromodels.core.model import Model
from astromodels.core.model_parser import clone_model
from astromodels.functions.spatial_model import HaloModel, ModelFactory
from astromodels.sources.extended_source import ExtendedSource

__author__ = "torresramiro350"

# NOTE: test implementation for the HaloModel class, which combines functionality
# from the Galprop_Template3D (parameter interpolation per template map) and template_model (spectral interpolation)


def make_test_template(ra, dec, fitsfile):

    test_wcs = False

    if test_wcs:
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [100, 100]
        w.wcs.cdelt = np.array([-0.2, 0.2])
        w.wcs.crval = [ra, dec]
        w.wcs.ctype = ["RA---TAN", "DEC---TAN"]
        dOmega = (
            (abs(w.wcs.cdelt[0] * w.wcs.cdelt[1]) * u.degree * u.degree)
            .to(u.steradian)
            .value
        )
        header = w.to_header()

    else:
        # NOTE: Sample template header used for Geminga template analysis
        # SIMPLE  =                    T / Written by IDL:  Thu Apr  2 22:08:49 2015
        # BITPIX  =                  -64 / number of bits per data pixel
        # NAXIS   =                    3 / number of data axes
        # NAXIS1  =                  201 / length of data axis 1
        # NAXIS2  =                  201 / length of data axis 2
        # NAXIS3  =                   21 / length of data axis 3
        # EXTEND  =                    T / FITS dataset may contain extensions
        # COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
        # COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
        # CRVAL1  =    98.47879999999984 / Value of longitude in pixel CRPIX1
        # CDELT1  =                 0.08 / Step size in longitude
        # CRPIX1  =                101.5 / Pixel that has value CRVAL1
        # CTYPE1  = 'RA-CAR  '           / The type of parameter 1 (Galactic longitude in
        # CUNIT1  = 'deg     '           / The unit of parameter 1
        # CRVAL2  =              17.7732 / Value of latitude in pixel CRPIX2
        # CDELT2  =                 0.08 / Step size in latitude
        # CRPIX2  =                101.5 / Pixel that has value CRVAL2
        # CTYPE2  = 'DEC-CAR '           / The type of parameter 2 (Galactic latitude in C
        # CUNIT2  = 'deg     '           / The unit of parameter 2
        # CRVAL3  =     58.4731330871582 / Energy of pixel CRPIX3
        # CDELT3  =                  1.0 / log10 of step size in energy (if it is logarith
        # CRPIX3  =                   1. / Pixel that has value CRVAL3
        # CTYPE3  = 'Energy  '           / Axis 3 is the spectra
        # CUNIT3  = 'MeV     '           / The unit of axis 3
        # CREATOR = 'chimgtyp 1.5'       /  s/w task that wrote this dataset
        # ...
        # HISTORY ---------------------------------------------------------
        # HISTORY
        # DATE    = '2018-03-19'         / file creation date (YYYY-MM-DDThh:mm:ss UT)
        # HISTORY Scaled for V15 CLEAN IRFs by ratio of 4-year exposures for V10 IRFs
        # HISTORY Scaled for P8V6 Source by apply_scale2.pro

        cards = {
            "SIMPLE": "T",
            "BITPIX": -64,
            "NAXIS": 3,
            "NAXIS1": 201,
            "NAXIS2": 201,
            "NAXIS3": 2,
            "EXTEND": "T",
            "COMMENT": "FITS (Flexible Image Transport System) format is defined in 'Astronomy"
            " and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H",
            "CRVAL1": 98.47879999999984,
            "CDELT1": 0.08,
            "CRPIX1": 101.5,
            "CTYPE1": "RA-CAR  ",
            "CUNIT1": "deg     ",
            "CRVAL2": 17.7732,
            "CDELT2": 0.08,
            "CRPIX2": 101.5,
            "CTYPE2": "DEC-CAR ",
            "CUNIT2": "deg     ",
            "CRVAL3": 58.4731330871582,
            "CDELT3": 1.0,
            "CRPIX3": 1.0,
            "CTYPE3": "Energy  ",
            "CUNIT3": "MeV     ",
        }

        dOmega: float = (
            (abs(cards["CDELT1"] * cards["CDELT2"]) * u.degree * u.degree)
            .to(u.steradian)
            .value
        )

        header = fits.Header(cards)

    data = np.zeros([2, 201, 201])
    for i in range(2):
        data[i][80:120][80:120] = 1

    total = np.sum(data)

    data /= total / dOmega

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(fitsfile, overwrite=True)


@pytest.mark.slow
def test_model_factory_1D():

    ra, dec = (98.47879999999984, 17.7732)

    ra_min = ra - 4.0
    ra_max = ra + 4.0
    dec_min = dec - 4.0
    dec_max = dec + 4.0

    make_test_template(ra, dec, "__test3D_simplefits")

    diff = np.arange(25.0, 25.4, 0.2)

    halo_model_factory = ModelFactory(
        "__test3D_1D",
        "sample 3D template with 1 var interpolation",
        ["diff"],
        degree_of_interpolation=1,
        spline_smoothing_factor=0,
    )

    halo_model_factory.define_parameter_grid("diff", diff)

    for i, dval in enumerate(diff):
        halo_model_factory.add_interpolation_data("__test3D_simple.fits", diff=dval)

    halo_model_factory.save_data(overwrite=True)

    halo_model = HaloModel("__test3D_1D")
    halo_model.define_region(ra_min, ra_max, dec_min, dec_max, galactic=False)
    halo_model.clean()


@pytest.mark.slow
def test_model_factory_2D():

    ra, dec = (98.47879999999984, 17.7732)

    diff = np.arange(25.0, 25.4, 0.2)
    index = np.arange(1.0, 1.4, 0.2)

    ra_min = ra - 4.0
    ra_max = ra + 4.0
    dec_min = dec - 4.0
    dec_max = dec + 4.0

    make_test_template(ra, dec, "__test3D_simple.fits")

    halo_model_factory = ModelFactory(
        "__test3D_2D",
        "Sample 3D template 2 var interpolation",
        ["diff", "index"],
        degree_of_interpolation=1,
        spline_smoothing_factor=0,
    )

    halo_model_factory.define_parameter_grid("diff", diff)
    halo_model_factory.define_parameter_grid("index", index)

    for i, dval in enumerate(diff):
        for j, idx in enumerate(index):
            halo_model_factory.add_interpolation_data(
                "__test3D_simple.fits", diff=dval, index=idx
            )

    halo_model_factory.save_data(overwrite=True)

    halo_model = HaloModel("__test3D_2D")
    halo_model.define_region(ra_min, ra_max, dec_min, dec_max, galactic=False)
    halo_model.clean()
