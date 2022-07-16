import os
import gc
import pathlib
import re
import hashlib
from builtins import object, range, str
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import collections
import h5py
import numpy as np
import scipy.interpolate
from interpolation import interp
from interpolation.splines import eval_linear
import astropy.units as u
import pandas as pd

from astropy.io import fits
from pandas import HDFStore
from future.utils import with_metaclass

from astromodels.core.parameter import Parameter
from astropy.coordinates import SkyCoord, ICRS, BaseCoordinateFrame
from astromodels.functions.function import Function3D, FunctionMeta
from scipy.interpolate import RegularGridInterpolator
from astromodels.utils import get_user_data_path
from astromodels.utils.angular_distance import angular_distance_fast
from astromodels.utils.logging import setup_logger

__author__ = "Ramiro"

# NOTE: Script adapted GalProp and TemplateModelFactory in Astromodels.

log = setup_logger(__name__)

__all__ = [
    "IncompleteGrid",
    "ValuesNotInGrid",
    "MissingDataFile",
    "ModelFactory",
    "SpatialModel",
]


class IncompleteGrid(RuntimeError):
    """Raises if grid is incomplete

    Args:
        RuntimeError (None): Will raise if the grid contains any Nans or None
        values.
    """

    pass


class ValuesNotInGrid(ValueError):
    """Raises if any of the values isn't contained within the grid

    Args:
        ValueError (Exception): Will be present if parameters are not in
        the defined grid.
    """

    pass


class MissingDataFile(RuntimeError):
    """Check if the file exists

    Args:
        RuntimeError (Exception): Check if the file is present, if not raise
        a RuntimeError.
    """

    pass


# This dictionary will keep track of the new classes already
# created in the current session
_classes_cache = {}


class GridInterpolate(object):
    def __init__(self, grid, values):
        self._grid = grid
        self._values = np.ascontiguousarray(values)

    def __call__(self, v):
        return eval_linear(self._grid, self._values, v)


class UnivariateSpline(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __call__(self, v):
        return interp(self._x, self._y, v)


# This class builds a dataframe from morphology parameters for a source
# The information from the source comes from FITS files with
# 3D template model maps
class ModelFactory(object):
    def __init__(
        self,
        name,
        description,
        names_of_parameters,
        degree_of_interpolation=1,
        spline_smoothing_factor=0,
    ):
        """Class builds a data table from 3d mapcube templates with information
        on morphology and spectral parameters for different energy bins. These
        parameters are used for interpolation in the class SpatialModel

        Args:
            name (str): Name of the outfile.
            description (str): A brief summary of the table for reference.
            names_of_parameters (np.ndarray): Provide a name for the
            parameters.
            degree_of_interpolation (int, optional): Degree of interpolation.
            For 3D, interpolation can only be done linearly. Defaults to 1.
            spline_smoothing_factor (int, optional): Smoothing factor in
            in interpolation. Defaults to 0.

        Raises:
            RuntimeError: Name of the file cannot contain spaces or special
            characters.
        """

        # Store the model name
        # Ensure that it contains no spaces nor special characters
        name = str(name)

        if re.match("[a-zA-Z_][a-zA-Z0-9_]*", name) is None:

            log.error(
                f"The provided name {name} is not a valid name. "
                "You cannot use spaces, nor special characters."
            )

            raise RuntimeError()

        self._name = name
        self._description = str(description)
        self._degree_of_interpolation = int(degree_of_interpolation)
        self._spline_smoothing_factor = int(spline_smoothing_factor)

        # Create a dictionary which will contain the grid for each parameter
        self._parameters_grids = collections.OrderedDict()

        for parameter_name in names_of_parameters:

            self._parameters_grids[parameter_name] = None

        self._data_frame = None
        self._parameters_multi_index = None
        self._map_multi_index = None
        self._interpolators = None
        self._fitsfile = None
        self._E = None
        self._L = None
        self._B = None
        self._delLat = None
        self._delLon = None
        self._delEn = None
        self._parameters_grids = None
        self._refLon = None
        self._refLonPix = None
        self._refLat = None
        self._refLatPix = None
        self._refEn = None
        self._nl = None
        self._nb = None
        self._ne = None
        self._map = None

    def define_parameter_grid(self, parameter_name: str, grid: np.ndarray):
        """
        Define the parameter grid for this parameter.
        Pass the name of the parameter and the array of values that will
        take in the grid.
        """

        if parameter_name not in self._parameters_grids:

            log.error(f"Parameter {parameter_name} is not part of this model.")

            raise AssertionError()

        # if the grid is not numpy array, conver it to one
        grid_ = np.array(grid, dtype=float)

        if grid_.shape[0] <= 1:

            log.error(
                "A grid for a parameter must containt at least two "
                "elements for interpolation."
            )

            raise AssertionError()

        # Assert that all elements are unique
        if not np.all(np.unique(grid_) == grid_):
            log.error(
                f"Non-unique elements found in grid of parameter {parameter_name}."
            )

        self._parameters_grids[parameter_name] = grid_

    def add_interpolation_data(
        self, fitsfile: str, ihdu: int = 0, **parameters_values_input: dict
    ):
        """Fill data table with information from 3D mapcube templates.

        Args:
            fitsfile (str): FITS file with 3D mapcube.
            ihdu (int, optional): Primary HDU identifier. Defaults to 0.

        Raises:
            IncompleteGrid: Check if grid contains any meaningful values.
            RuntimeError: Check if a FITS file was specified.
            AssertionError: Ensure the number of parameters is the same as
            declared in the define_parameter_grid method.
        """

        # Verify that a grid has been defined for all parameters
        for grid in list(self._parameters_grids.values()):

            if grid is None:

                log.error(
                    "You need to define a grid for all parameters, "
                    "by using the define_parameter_grid method."
                )

                raise IncompleteGrid()

        # Now load the information from the FITS 3D template model
        # template models contain flux values in terms of energy, RA, and dec

        if fitsfile is None:

            log.error("You need to specify a FITS file with a template map.")

            raise RuntimeError()

        self._fitsfile = fitsfile

        with fits.open(self._fitsfile) as f:

            self._delLon = f[ihdu].header["CDELT1"]
            self._delLat = f[ihdu].header["CDELT2"]
            self._delEn = 0.2  # f[ihdu].header["CDELT3"]
            self._refLon = f[ihdu].header["CRVAL1"]
            self._refLat = f[ihdu].header["CRVAL2"]
            # f[ihdu].header["CRVAL3"] #Log(E/MeV) -> GeV to MeV
            self._refEn = 5
            self._refLonPix = f[ihdu].header["CRPIX1"]
            self._refLatPix = f[ihdu].header["CRPIX2"]
            # self._refEnPix = f[ihdu].header["CRPIX3"]

            self._map = f[ihdu].data  # 3D array containing the flux values

            self._nl = f[ihdu].header["NAXIS1"]  # Longitude
            self._nb = f[ihdu].header["NAXIS2"]  # Latitude
            self._ne = f[ihdu].header["NAXIS3"]  # Energy

            self._L = np.linspace(
                self._refLon - self._refLonPix * self._delLon,
                self._refLon + (self._nl - self._refLonPix - 1) * self._delLon,
                self._nl,
                dtype=float,
            )

            self._B = np.linspace(
                self._refLat - self._refLatPix * self._delLat,
                self._refLat + (self._nb - self._refLatPix - 1) * self._delLat,
                self._nb,
                dtype=float,
            )

            self._E = np.linspace(
                self._refEn,
                self._refEn + (self._ne - 1) * self._delEn,
                self._ne,
                dtype=float,
            )

            # for i, e in enumerate(self._E):
            #
            #    self._map[i] = np.fliplr(self._map[i])

            h = hashlib.sha224()
            h.update(self._map)
            self.hash = int(h.hexdigest(), 16)

        if self._data_frame is None:

            # shape = []

            shape = [len(v) for k, v in self._parameters_grids.items()]

            shape.append(self._E.shape[0])
            shape.append(self._L.shape[0])
            shape.append(self._B.shape[0])

            log.debug(f"grid shape: {shape}")

            self._data_frame = np.zeros(tuple(shape))

            log.debug(f"grid shape actual: {self._data_frame.shape}")

            # This is the first data set, create the data frame
            # The dataframe is indexed by a parameter multi-index for rows and
            # with the 3D template multi-index information as the columns

            log.info("Creating the multi-indices....")

        # Make sure we have all parameters and order the values in the same way as the dictionary
        parameter_idx = []
        for i, (key, val) in enumerate(self._parameters_grids.items()):

            if key not in parameters_values_input:

                log.error(f"Parameter {key} is not in input")

            parameter_idx.append(
                int(np.where(val == parameters_values_input[key])[0][0])
            )

        log.debug(f"have index {parameter_idx}")

        if len(parameter_idx) != len(self._parameters_grids):

            log.error("You didn't specify all parameters' values")

            raise AssertionError()

        for i, e in enumerate(self._E):
            for j, l in enumerate(self._L):
                for k, b in enumerate(self._B):

                    tmp = pd.to_numeric(
                        self._map[i][j][k]
                    )  # filling with map information

                    self._data_frame[tuple(parameter_idx)][i][j][k] = tmp

    def save_data(self, overwrite=False):
        """Save the table into a file for later usage with SpatialModel.

        Args:
            overwrite (bool, optional): Overwrite file it already exists.
            Defaults to False.

        Raises:
            AssertionError: Raises if there are any non-numeric values within
            table.
            IOError: Raises if file exists, but cannot be deleted
            for lack of write privileges.
            IOError: Raises if file exists, and cannot be overwrriten
            (overwrite is False).
        """

        # First make sure that the whole data matrix has been filled
        if np.any(np.isnan(self._data_frame)):

            log.error(
                "You have NaNs in the data matrix. Usually this means "
                "that you didn't fill it up completely, or that some of "
                "your data contains nans. Cannot save the file."
            )

            raise AssertionError()

        # Get the data directory
        data_dir_path: Path = get_user_data_path()

        # Sanitize the data file
        filename_sanitized: Path = data_dir_path / f"{self._name}.h5"

        # Check if the file already exists
        # if os.path.exists(filename_sanitized):
        if filename_sanitized.exists():

            if overwrite:

                try:

                    os.remove(filename_sanitized)

                except IOError:

                    log.error(
                        f"The file {filename_sanitized} already exists. "
                        "and cannot be removed (maybe you do not have "
                        "enough permissions to do so?)."
                    )

                    raise IOError()

            else:

                log.error(
                    f"The file {filename_sanitized} already exists! "
                    "You cannot call two different spatial models with the "
                    "same name."
                )

                raise IOError()

        # Open the HDF5 and write objects
        template_file: TemplateFile = TemplateFile(
            name=self._name,
            description=self._description,
            spline_smoothing_factor=self._spline_smoothing_factor,
            degree_of_interpolation=self._degree_of_interpolation,
            grid=self._data_frame,
            energies=self._E,
            lats=self._B,
            lons=self._L,
            parameters=self._parameters_grids,
            parameter_order=list(self._parameters_grids.keys()),
        )

        template_file.save(filename_sanitized)


# This adds a method to a class at run time
def add_method(self, method, name=None):

    if name is None:

        name = method.func_name

    setattr(self.__class__, name, method)


class RectBivariateSplineWrapper(object):
    """Wrapper around RectBivariateSplien which supplies a __call__ method
    which accepts the same syntax as other interpolation methods.py

    Args:
        object (RectBivariateSpline): Interpolation for 2D.
    """

    def __init__(self, *args, **kwargs):

        # We can use interp2, which features spline interpolation instead of linear interpolation
        self._interpolator = scipy.interpolate.RectBivariateSpline(*args, **kwargs)

    def __call__(self, x):

        res = self._interpolator(*x)

        return res[0][0]


@dataclass
class TemplateFile:
    """
    simple container to read and write the
    data to an hdf5 file

    """

    name: str
    description: str
    grid: np.ndarray
    energies: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    parameters: Dict[str, np.ndarray]
    parameter_order: List[str]
    degree_of_interpolation: int
    spline_smoothing_factor: float

    def save(self, file_name: str):
        """

        serialize the contents to a file
        :param file_name:
        :type file_name: str
        :returns:

        """

        with h5py.File(file_name, "w") as f:

            f.attrs["name"] = self.name
            f.attrs["description"] = self.description
            f.attrs["degree_of_interpolation"] = self.degree_of_interpolation
            f.attrs["spline_smoothing_factor"] = self.spline_smoothing_factor

            f.create_dataset("energies", data=self.energies, compression="gzip")
            f.create_dataset("lats", data=self.lats, compression="gzip")
            f.create_dataset("lons", data=self.lons, compression="gzip")
            f.create_dataset("grid", data=self.grid, compression="gzip")

            # store the parameter order
            dt = h5py.special_dtype(vlen=str)
            po = np.array(self.parameter_order, dtype=dt)
            f.create_dataset("parameter_order", data=po)
            par_group = f.create_group("parameters")
            for k in self.parameter_order:

                par_group.create_dataset(k, data=self.parameters[k], compression="gzip")

    @classmethod
    def from_file(cls, file_name: str):
        """
        read contents from a file
        :param cls:
        :type cls:
        :param file_name:
        :type file_name: str
        :returns

        """
        with h5py.File(file_name, "r") as f:

            name = f.attrs["name"]
            description = f.attrs["description"]
            degree_of_interpolation = f.attrs["degree_of_interpolation"]
            spline_smoothing_factor = f.attrs["spline_smoothing_factor"]

            parameter_order = f["parameter_order"][()]
            energies = f["energies"][()]
            lats = f["lats"][()]
            lons = f["lons"][()]

            grid = f["grid"][()]

            parameters = collections.OrderedDict()

            for k in parameter_order:

                parameters[k] = f["parameters"][k][()]

        return cls(
            name=name,
            description=description,
            degree_of_interpolation=degree_of_interpolation,
            spline_smoothing_factor=spline_smoothing_factor,
            energies=energies,
            lats=lats,
            lons=lons,
            parameter_order=parameter_order,
            parameters=parameters,
            grid=grid,
        )


class SpatialModel(with_metaclass(FunctionMeta, Function3D)):

    r"""
    description: 3D interpolation over morphology of a source using FITS templates
                with spectral and spatial information.

    latex: $n.a.$


    parameters:

        K:

            desc: Normalization (freeze this to 1 if the template provides the
            normalization by itself)
            initial value: 1
            fix: yes

        lon0:

            desc: Longitude of the center of source
            initial value: 0.0
            min: 0.0
            max: 360.0

        lat0:

            desc: Latitude of the center of source
            initial value: 0.0
            min: -90.0
            max: 90.0
    """

    def _custom_init_(self, model_name, other_name=None):
        """
        Custom initialization for this model
        :param model_name: the name of the model, corresponding to the
        root of the .h5 file in the data directory
        :param other_name: (optional) the name to be used as name of the model
        when used in astromodels. If None
        (default), use the same as model_name.
        :return: none
        """

        # Get the data directory
        data_dir_path: Path = get_user_data_path()

        # Sanitize the file
        filename_sanitized = data_dir_path.absolute() / f"{model_name}.h5"

        if not filename_sanitized.exists():

            log.error(
                f"The data file {filename_sanitized} does not exist."
                " Did you use the ModelFactory?"
            )

            raise MissingDataFile

        # Open the template definition and read from it
        self._data_file = filename_sanitized

        # use the file shadow to read
        template_file: TemplateFile = TemplateFile.from_file(filename_sanitized)

        self._parameters_grids = collections.OrderedDict()

        for key in template_file.parameter_order:

            try:

                # sometimes this is stored binary
                k = key.decode()

            except AttributeError:

                # if not, load as a normal str
                k = key

            log.debug(f"reading parameter {str(k)}")

            self._parameters_grids[str(k)] = template_file.parameters[key]

        # get the template parameters
        self._E = template_file.energies
        self._B = template_file.lats
        self._L = template_file.lons

        # get the dataframe
        self.grid = template_file.grid

        # Now get the metadata
        description = template_file.description
        name = template_file.name

        self._degree_of_interpolation = template_file.degree_of_interpolation
        self._spline_smoothing_factor = template_file.spline_smoothing_factor

        # Make the dictionary of parameters for the model
        function_definition = collections.OrderedDict()
        function_definition["description"] = description
        function_definition["latex"] = "n.a."

        # Now build the parameters according to the content of the parameter grid
        parameters = collections.OrderedDict()

        parameters["K"] = Parameter("K", 1.0)
        parameters["lon0"] = Parameter("lon0", 0.0, min_value=0.0, max_value=360.0)
        parameters["lat0"] = Parameter("lat0", 0.0, min_value=-90.0, max_value=90.0)

        for parameter_name in list(self._parameters_grids.keys()):

            grid = self._parameters_grids[parameter_name]

            parameters[parameter_name] = Parameter(
                parameter_name,
                np.median(grid),
                min_value=grid.min(),
                max_value=grid.max(),
            )

        if other_name is None:

            # super(SpatialModel, self).__init__(name, function_definition,
            #  parameters)
            super().__init__(name, function_definition, parameters)

        else:

            # super(SpatialModel, self).__init__(
            # other_name, function_definition, parameters
            # )
            super().__init__(other_name, function_definition, parameters)

        self._setup()

        # clean things up a bit

        del template_file

        gc.collect()

    def _prepare_interpolators(self, log_interp: bool, data_frame: np.ndarray):
        """
        :function reads column of flux values and performs interpolation over
        the parameters specified in ModelFactory
        :param: log_interp: the normalization of flux is done in log scale
        by default.
        :return: (none)
        """

        log.info("Preparing the interpolators...")

        # Figure out the shape of the data matrices
        para_shape = np.array(
            [x.shape[0] for x in list(self._parameters_grids.values())]
        )

        # interpolate over the parameters
        self._interpolators = []

        for i, e in enumerate(self._E):
            for j, l in enumerate(self._L):
                for k, b in enumerate(self._B):

                    if log_interp:

                        this_data = np.array(
                            np.log10(data_frame[..., i, j, k]).reshape(*para_shape),
                            dtype=float,
                        )

                        self._is_log10 = True

                    else:

                        this_data = np.array(
                            data_frame[..., i, j, k].reshape(*para_shape), dtype=float
                        )

                        self._is_log10 = False

                    if len(list(self._parameters_grids.values())) == 1:

                        parameters = list(self._parameters_grids.values())

                        xpoints = np.array(
                            [parameters[x] for x in range(len(parameters))][0]
                        )
                        ypoints = np.array(
                            [this_data[x] for x in range(this_data.shape[0])]
                        )

                        this_interpolator = UnivariateSpline(xpoints, ypoints)

                    elif len(list(self._parameters_grids.values())) == 2:

                        x, y = list(self._parameters_grids.values())

                        # Make sure that the requested polynomial degree is
                        # less thant the number of data sets in
                        # both directions

                        msg = (
                            "You cannot use an interpolation degree of %s if "
                            "you don't provide at least %s points "
                            "in the %s direction. Increase the number of "
                            "templates or decrease interpolation degree."
                        )

                        if len(x) <= self._degree_of_interpolation:

                            log.error(
                                msg
                                % (
                                    self._degree_of_interpolation,
                                    self._degree_of_interpolation + 1,
                                    "x",
                                )
                            )

                            raise RuntimeError()

                        if len(y) <= self._degree_of_interpolation:

                            log.error(
                                msg
                                % (
                                    self._degree_of_interpolation,
                                    self._degree_of_interpolation + 1,
                                    "y",
                                )
                            )

                            raise RuntimeError()

                        this_interpolator = RectBivariateSplineWrapper(
                            x,
                            y,
                            this_data,
                            kx=self._degree_of_interpolation,
                            ky=self._degree_of_interpolation,
                            s=self._spline_smoothing_factor,
                        )

                    else:

                        # In more than 2d, we can only interpolate linearly
                        # this_interpolator = RegularGridInterpolator(
                        this_interpolator = GridInterpolate(
                            tuple(
                                [
                                    np.array(x)
                                    for x in list(self._parameters_grids.values())
                                ]
                            ),
                            this_data,
                        )

                    self._interpolators.append(this_interpolator)

        del data_frame
        gc.collect()

    def _interpolate(
        self,
        energies: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        parameter_values: np.ndarray,
    ):
        """Interpolates over the morphology parameters and creates the
        interpolating function over energy, ra, and dec

        Args:
            energies (np.ndarray): Energy Bins
            lons (np.ndarray): Longitude (RAs)
            lats (np.ndarray): Latitude (Dec) values
            parameter_values (np.ndarray): morphology parameters where to
            evalute the interpolating function from _prepare_interpolators.

        Raises:
            AttributeError: Check if _prepare_interpolators has been run,
            if it hasn't yet been run. Run it now.

        Returns:
            np.ndarray: Returns interpolated values over energies, longiutes,
            and latitudes.
        """

        # gather all interpolations for these parameters' values
        try:

            interpolated_map = np.array(
                [
                    self._interpolators[j](np.atleast_1d(parameter_values))
                    for j in range(len(self._interpolators))
                ]
            )

        except AttributeError:

            self._prepare_interpolators(log_interp=False, data_frame=self.grid)

            interpolated_map = np.array(
                [
                    self._interpolators[j](np.atleast_1d(parameter_values))
                    for j in range(len(self._interpolators))
                ]
            )

        # map_shape = [x.shape[0] for x in list(self._map_grids.values())]
        map_shape = [x.shape[0] for x in [self._E, self._L, self._B]]

        interpolator = RegularGridInterpolator(
            (self._E, self._L, self._B),
            interpolated_map.reshape(*map_shape),
            bounds_error=False,
            fill_value=0.0,
        )

        if lons.size != lats.size:

            log.error("Lons and lats should have the same size!")

            raise AttributeError()

        # f_interpolated = np.zeros([energies.size, lats.size])
        f_interpolated = np.zeros([lons.size, energies.size])

        # evaluate the interpolators over energy, ra, and dec
        for i, e in enumerate(energies):

            engs = np.repeat(e, lats.size)

            # slice_points = tuple((engs, lats, lons))

            if self._is_log10:

                # NOTE: if interpolation is carried using the log10 scale,
                # ensure that values outside range of interpolation remain
                # zero after conversion to linear scale.
                # because if function returns zero, 10**(0) = 1.
                # This affects the fit in 3ML and breaks things.

                log_interpolated_slice = interpolator(tuple([engs, lons, lats]))

                interpolated_slice = np.array(
                    [
                        0.0 if x == 0.0 else np.power(10.0, x)
                        for x in log_interpolated_slice
                    ]
                )

            else:
                interpolated_slice = interpolator(tuple([engs, lons, lats]))

            f_interpolated[:, i] = interpolated_slice

        assert np.all(np.isfinite(f_interpolated)), "some values are wrong!"

        return f_interpolated

    def _setup(self):

        self._frame = "ICRS"  # ICRS()

    def clean(self):
        """
        Table models can consume a lof of memory.
        This method calls a clean method and removes some of the memory
        consumed by the models.
        :returns:
        """
        self._interpolators = None
        del self._interpolators
        gc.collect()

        log.info("You have cleaned the table model and it will no longer be usable.")

    def __del__(self):

        self.clean()

    def _set_units(self, x_unit, y_unit, z_unit, w_unit):

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit

        # self.K.unit = 1/(u.MeV * u.cm**2 * u.s * u.sr)
        # keep this units to if templates have been normalized
        self.K.unit = 1 / (u.sr)

    def evaluate(self, x, y, z, K, lon0, lat0, *args):

        lons = x
        lats = y
        energies = z

        angsep = angular_distance_fast(lon0, lat0, lons, lats)

        # if only one energy is passed, make sure we can iterate just once
        if not isinstance(energies, np.ndarray):

            energies = np.array(energies)

        # transform energy from keV to MeV
        # galprop likes MeV, 3ML likes keV
        log_energies = np.log10(energies) - np.log10((u.MeV.to("keV") / u.keV).value)

        return np.multiply(
            K, self._interpolate(log_energies, lons, lats, args)
        )  # if templates are normalized no need to convert back

    def set_frame(self, new_frame):
        """
        Set a new frame for the coordinates (the default is ICRS J2000)
        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """

        assert isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame

    @property
    def data_file(self):

        return self._data_file

    def to_dict(self, minimal=False):

        data = super(Function3D, self).to_dict(minimal)

        if not minimal:

            data["extra_setup"] = {
                "_frame": self._frame,
                "ramin": self.ramin,
                "ramax": self.ramax,
                "decmin": self.decmin,
                "decmax": self.decmax,
            }

        return data

    # Define the region within the template ROI
    def define_region(
        self, a: float, b: float, c: float, d: float, galactic: bool = False
    ):
        """Define the boundaries of template

        Args:
            a (float): Minimum longitude
            b (float): Maximum longitude
            c (float): Minimum latitude
            d (float): Maximum latitude
            galactic (bool, optional): Converts lon, lat to galactic coordinates.
            Defaults to False.

        Returns:
            tuple(float, float, float, float): Returns a tuple of the
            boundaries of RA and Dec or galactic coordinates if galactic=True.
        """

        if galactic:

            lmin = a
            lmax = b
            bmin = c
            bmax = d

            _coord = SkyCoord(
                l=[lmin, lmin, lmax, lmax], b=[bmin, bmax, bmax, bmin], frame="galactic"
            )

            self.ramin = min(_coord.transform_to("icrs").ra.value)
            self.ramax = max(_coord.transform_to("icrs").ra.value)
            self.decmin = min(_coord.transform_to("icrs").dec.value)
            self.decmax = max(_coord.transform_to("icrs").dec.value)

        else:

            self.ramin = a
            self.ramax = b
            self.decmin = c
            self.decmax = d

        return self.ramin, self.ramax, self.decmin, self.decmax

    def get_boundaries(self):

        min_longitude = self.ramin
        max_longitude = self.ramax
        min_latitude = self.decmin
        max_latitude = self.decmax

        return (min_longitude, max_longitude), (min_latitude, max_latitude)
