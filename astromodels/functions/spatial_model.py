import os
import re
import warnings
import hashlib

import numpy as np
import scipy.interpolate
import astropy.units as u
import pandas as pd
import collections

from astropy.io import fits
from pandas import HDFStore
from builtins import object, range, str
from future.utils import with_metaclass

from astromodels.core.parameter import Parameter
from astropy.coordinates import SkyCoord, ICRS, BaseCoordinateFrame
from astromodels.functions.function import Function3D, FunctionMeta
from scipy.interpolate import RegularGridInterpolator as GridInterpolate
from astromodels.utils.configuration import get_user_data_path
from astromodels.utils.angular_distance import angular_distance_fast
from astromodels.utils.logging import setup_logger

log = setup_logger(__name__)

__all__ = ["IncompleteGrid",
        "ValuesNotInGrid",
        "MissingDataFile",
        "ModelFactory",
        "SpatialModel"]

class IncompleteGrid(RuntimeError):
    pass

class ValuesNotInGrid(ValueError):
    pass

class MissingDataFile(RuntimeError):
    pass

#This dictionary will keep track of the new classes already created in the current session
_classes_cache = {}

#This class builds a dataframe from morphology parameters for a source
#The information from the source comes from FITS files with 3D template model maps
class ModelFactory(object):
    
    def __init__(self,
                name,
                description,
                names_of_parameters,
                degree_of_interpolation=1,
                spline_smoothing_factor=0):

        #Store the model name
        #Ensure that it contains no spaces nor special characters
        name = str(name)

        if re.match("[a-zA-Z_][a-zA-Z0-9_]*", name) is None:

            log.error(f"The provide name {name} is not a valid name."
                    " You cannot use spaces, nor special characters.")

            raise RuntimeError()

        self._name = name
        self._description = str(description)
        self._degree_of_interpolation = int(degree_of_interpolation)
        self._spline_smoothing_factor = int(spline_smoothing_factor)

        #Create a dictionary which will contain the grid for each parameter
        self._parameters_grids = collections.OrderedDict()

        for parameter_name in names_of_parameters:
            
            self._parameters_grids[parameter_name] = None

        self._data_frame = None
        self._parameters_multi_index = None
        self._map_multi_index = None
        self._interpolators = None
        self._fitsfile = None

    def define_parameter_grid(self, parameter_name, grid):

        assert parameter_name in self._parameters_grids, (
            f"Parameter {parameter_name} is not part of this model")

        #if the grid is not numpy array, conver it to one
        grid_ = np.array(grid, dtype=float)

        assert grid_.shape[0] > 1, (
            "A grid for a parameter must contain at least two elements.")

        #Assert that all elements are unique
        assert np.all(np.unique(grid_) == grid_), (
            f"Non-unique elements in grid of parameter {parameter_name}")

        self._parameters_grids[parameter_name] = grid_

    def add_interpolation_data(self, fitsfile, ihdu=0, **parameters_values_input):

        #Verify that a grid has been defined for all parameters
        for grid in list(self._parameters_grids.values()):
            
            if grid is None:
                
                raise IncompleteGrid("You need to define a grid for all parameters, by using the"
                                    "define_parameter_grid method.")

        #Now load the information from the FITS 3D template model
        #template models contain flux values in terms of energy, RA, and dec

        if fitsfile is None:
            
            log.error("You need to specify a FITS file with a template map.")

            raise RuntimeError()

        self._fitsfile = fitsfile

        with fits.open(self._fitsfile) as f:

            self._delLon = f[ihdu].header["CDELT1"]
            self._delLat = f[ihdu].header["CDELT2"]
            self._delEn = 0.2 #f[ihdu].header["CDELT3"]
            self._refLon = f[ihdu].header["CRVAL1"]
            self._refLat = f[ihdu].header["CRVAL2"]
            self._refEn = 5 #f[ihdu].header["CRVAL3"] #Log(E/MeV) -> GeV to MeV
            self._refLonPix = f[ihdu].header["CRPIX1"]
            self._refLatPix = f[ihdu].header["CRPIX2"]
            #self._refEnPix = f[ihdu].header["CRPIX3"]

            self._map = f[ihdu].data #3D array containing the flux values

            self._nl = f[ihdu].header["NAXIS1"] #Longitude
            self._nb = f[ihdu].header["NAXIS2"] #Latitude
            self._ne = f[ihdu].header["NAXIS3"] #Energy


            self._L = np.linspace(self._refLon - self._refLonPix*self._delLon,
                                self._refLon + (self._nl - self._refLonPix - 1)*self._delLon,
                                self._nl, dtype=float)

            self._B = np.linspace(self._refLat - self._refLatPix*self._delLat,
                                self._refLat + (self._nb - self._refLatPix - 1)*self._delLat,
                                self._nb, dtype=float)

            self._E = np.linspace(self._refEn, self._refEn + (self._ne - 1)*self._delEn,
                                self._ne, dtype=float)

            h = hashlib.sha224()
            h.update(self._map)
            self.hash = int(h.hexdigest(), 16)

        #Get the map information into a dictionary
        self._map_grids = collections.OrderedDict()

        self._map_grids["energies"] = self._E
        self._map_grids["lats"] = self._B
        self._map_grids["lons"] = self._L

        if self._data_frame is None:
            
            #This is the first data set, create the data frame
            #The dataframe is indexed by a parameter multi-index for rows and
            #with the 3D template multi-index information as the columns

            log.info("Creating the multi-indices....")
            
            #multi-index with map information
            self._map_multi_index = pd.MultiIndex.from_product(list(self._map_grids.values()),
                                                            names=list(self._map_grids.keys()))

            #multi-index with parameters information
            self._parameters_multi_index = pd.MultiIndex.from_product(list(self._parameters_grids.values()),
                                                                names=list(self._parameters_grids.keys()))

            #Pre-fill the data matrix with nans, so we will know if some elements have not been filled
            self._data_frame = pd.DataFrame(index=self._parameters_multi_index,
                                            columns=self._map_multi_index)

        #Make sure we have all parameters and order the values in the same way as the dictionary
        parameters_values = np.zeros(len(self._parameters_grids))*np.nan

        for key in parameters_values_input:
            
            assert key in self._parameters_grids, (f"Parameter {key} is not known")

            idx = list(self._parameters_grids.keys()).index(key)

            parameters_values[idx] = parameters_values_input[key]

        #If the user did not specy one of the parameters, then the parameter_values will contain nan
        assert np.all(np.isfinite(parameters_values)), ("You didn't specify all parameters' values")
        
        try:
            
            for i, e in enumerate(self._E):
                for j, l in enumerate(self._L):
                    for k, b in enumerate(self._B):
                        
                        tmp = pd.to_numeric(self._map[i][j][k])

                        if len(parameters_values) == 1:
                        
                            self._data_frame.loc[parameters_values.tolist()] = tmp

                        else:
                        
                            self._data_frame.loc[tuple(parameters_values)][(e, b, l)] = tmp

        except:

            raise ValuesNotInGrid(
                                f"The provided parameter values ({parameters_values}) "
                                "are not in the defined grid.") 
    
    def save_data(self, overwrite=False):

        #First make sure that the whole data matrix has been filled
        assert not self._data_frame.isnull().values.any(), (
            "You have NaNs in the data matrix. Usually this means " 
            "that you didn't fill it up completely, or that some of "
            "your data contains nans. Cannot save the file.")

        #Get the data directory
        data_dir_path = get_user_data_path()

        #Sanitize the data file
        filename_sanitized = os.path.abspath(os.path.join(data_dir_path, f"{self._name}.h5"))

        #Check if the file already exists
        if os.path.exists(filename_sanitized):

            if overwrite:

                try:

                    os.remove(filename_sanitized)

                except:

                    log.error(
                            f"The file {filename_sanitized} already exists"
                            " and cannot be removed (maybe you do not have "
                            "permissions to do so?).")

                    raise IOError()

            else:

                log.error(
                        f"The file {filename_sanitized} already exists!"
                        " You cannot call two different "
                        "spatial models with same name.")

                raise IOError()

        #Open the HDF5 and write objects
        with HDFStore(filename_sanitized) as store:

            self._data_frame.apply(pd.to_numeric).to_hdf(store, "data_frame")

            store.get_storer("data_frame").attrs.metadata = {"description": self._description,
                                                            "name": self._name,
                                                            "degree_of_interpolation": self._degree_of_interpolation,
                                                            "spline_smoothing_factor": self._spline_smoothing_factor}

            #store the parameters
            for i, parameter_name in enumerate(self._parameters_grids.keys()):

                store[f"p_{i}_{parameter_name}"] = pd.Series(self._parameters_grids[parameter_name])

            #store the map parameters
            store["energies"] = pd.Series(self._E)
            store["lats"] = pd.Series(self._B)
            store["lons"] = pd.Series(self._L)

#This adds a method to a class at run time
def add_method(self, method, name=None):

    if name is None:

        name = method.func_name

    setattr(self.__class__, name, method)

class RectBivariateSplineWrapper(object):
    """ 
    Wrapper around RectBivariateSpline, which supplies a __call__ method which accepts the same
    syntax as the other interpolations methods.
    """

    def __init__(self, *args, **kwargs):

        #We can use interp2, which features spline interpolation instead of linear interpolation
        self._interpolator = scipy.interpolate.RectBivariateSpline(*args, **kwargs)

    def __call__(self, x):

        res = self._interpolator(*x)

        return res[0][0]

class SpatialModel(with_metaclass(FunctionMeta, Function3D)):

    r"""
        description: 3D interpolation over morphology of a source using FITS templates
                    with spectral and spatial information.
        
        latex: $n.a.$


        parameters:

            K:

                desc: Normalization (freeze this to 1 if the template provides the normalization by itself)
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
        
    def _custom_init_(self, model_name, other_name=None, log_interp=True):
        """
        Custom initialization for this model
        :param model_name: the name of the model, corresponding to the
        root of the .h5 file in the data directory
        :param other_name: (optional) the name to be used as name of the model
        when used in astromodels. If None
        (default), use the same as model_name
        :return: none
        """

        #Get the data directory
        data_dir_path = get_user_data_path()

        #Sanitize the file
        filename_sanitized = os.path.abspath(os.path.join(data_dir_path, f"{model_name}.h5"))

        if not os.path.exists(filename_sanitized):

            raise MissingDataFile(f"The data file {filename_sanitized} does not exist."
                                    " Did you use the ModelFactory?")

        #Open the data file and read from it
        self._data_file = filename_sanitized

        with HDFStore(filename_sanitized) as store:

            self._data_frame = store["data_frame"]
            
            self._parameters_grids = collections.OrderedDict()
            self._map_grids = collections.OrderedDict()

            processed_parameters = 0

            for key in store.keys():
                
                match = re.search("p_([0-9]+)_(.+)", key)

                if match is None:

                    continue

                else:

                    tokens = match.groups()

                    this_parameter_number = int(tokens[0])
                    this_parameter_name = str(tokens[1])

                    assert this_parameter_number == processed_parameters, (
                        "Parameters are out of order")

                    self._parameters_grids[this_parameter_name] = store[key]

                    processed_parameters += 1


            #Read in the map parameters
            self._E = np.array(store["energies"])
            self._B = np.array(store["lats"])
            self._L = np.array(store["lons"])
            
            #Now get the metadata
            metadata = store.get_storer("data_frame").attrs.metadata

            description = metadata["description"]
            name = metadata["name"]
            
            self._degree_of_interpolation = metadata["degree_of_interpolation"]
            self._spline_smoothing_factor = metadata["spline_smoothing_factor"]

        #Get the map parameters into the map template dictionary
        self._map_grids["energies"] = self._E
        self._map_grids["lats"] = self._B
        self._map_grids["lons"] = self._L

        #Make the dictionary of parameters for the model
        function_definition = collections.OrderedDict()
        function_definition["description"] = description
        function_definition["latex"] = "n.a."

        #Now build the parameters according to the content of the parameter grid
        parameters = collections.OrderedDict()

        parameters["K"] = Parameter("K", 1.0)
        parameters["lon0"] = Parameter("lon0",
                                        value=0.0,
                                        min_value=0.0,
                                        max_value=360.0)

        parameters["lat0"] = Parameter("lat0",
                                        value=0.0,
                                        min_value=-90.0,
                                        max_value=90.0)

        for parameter_name in list(self._parameters_grids.keys()):

            grid = self._parameters_grids[parameter_name]

            parameters[parameter_name] = Parameter(parameter_name,
                                                    grid.median(),
                                                    min_value=grid.min(),
                                                    max_value=grid.max())

        if other_name is None:

            super(SpatialModel, self).__init__(name, function_definition, parameters)

        else:

            super(SpatialModel, self).__init__(other_name, function_definition, parameters)

        #finally prepare the interpolators
        self._prepare_interpolators(name, log_interp)

        self._setup()

    def _prepare_interpolators(self, model_name, log_interp):
        """
        :function reads column of flux values and performs interpolation over
        the parameters specified in ModelFactory
        :param: log_interp: the normalization of flux is done in log scale by default
        :return: (none)
        """

        data_dir_path = get_user_data_path()
        
        try:
            
            log.info("Loading Interpolators...")

            infile = os.path.abspath(os.path.join(data_dir_path, f"{model_name}_interpolators.npz"))

            retrieved_results = np.load(infile, allow_pickle=True)

            self._interpolators = retrieved_results["interpolators"]

            self._is_log10 = bool(retrieved_results["is_log_scale"])

        except:

            log.info("Preparing the interpolators...")

            #Figure out the shape of the data matrices 
            para_shape = np.array([x.shape[0] for x in list(self._parameters_grids.values())])

            #interpolate over the parameters
            self._interpolators = []

            for e in self._E:
                for l in self._L:
                    for b in self._B:
                        
                        if log_interp:

                            this_data = np.log10(self._data_frame[(e, b, l)].to_numpy(dtype=float)).reshape(*para_shape)

                            self._is_log10 = True
                        
                        else:

                            this_data = self._data_frame[(e, b, l)].to_numpy(dtype=float).reshape(*para_shape)

                            self._is_log10 = False

                        if len(list(self._parameters_grids.values())) == 2:

                            x, y = list(self._parameters_grids.values())

                            #Make sure that the requested polynomial degree is less thant the number of data sets in
                            # both directions

                            msg = ("You cannot use an interpolation degree of %s if you don't provide at least %s points "
                                    "in the %s direction. Increase the number of templates or decrease interpolation "
                                    "degree.")
                            
                            if len(x) <= self._degree_of_interpolation:

                                log.error( msg % (self._degree_of_interpolation,
                                                self._degree_of_interpolation + 1, "x"))

                                raise RuntimeError()

                            if len(y) <= self._degree_of_interpolation:

                                log.error(msg % (self._degree_of_interpolation,
                                                self._degree_of_interpolation + 1, "y"))

                                raise RuntimeError()

                            this_interpolator = RectBivariateSplineWrapper(
                                                x,
                                                y,
                                                this_data,
                                                kx=self._degree_of_interpolation,
                                                ky=self._degree_of_interpolation,
                                                s=self._spline_smoothing_factor)

                        else:

                            this_interpolator = GridInterpolate(self._parameters_grids.values(),
                                                                this_data,
                                                                method="linear",
                                                                bounds_error=False,
                                                                fill_value=0.0)

                        self._interpolators.append(this_interpolator)

            #NOTE:Save results from first interpolation to save some time
                            
            interpolation_results = os.path.abspath(os.path.join(data_dir_path, f"{model_name}_interpolators"))

            np.savez(interpolation_results, interpolators=self._interpolators, is_log_scale=self._is_log10, allow_pickle=True)

    def _interpolate(self, energies, lats, lons, parameter_values):
        """ 
        interpolates over the morphology parameters and creates the interpolating 
        function for energy, ra, and dec
        param: parameter_values: morphology parameters 
        return: (none)
         """

        #gather all interpolations for these parameters" values
        interpolated_map = np.array([self._interpolators[j](np.atleast_1d(parameter_values))
                                    for j in range(len(self._interpolators))])

        map_shape = [x.shape[0] for x in list(self._map_grids.values())]

        #interpolating function over energy, RA, and Dec
        interpolator = GridInterpolate(self._map_grids.values(),
                                        interpolated_map.reshape(*map_shape),
                                        method="linear", bounds_error=False, fill_value=0.0)
        
        if lons.size != lats.size:

            raise AttributeError("Lons and lats should have the same size!")

        f_interpolated = np.zeros([energies.size, lats.size]) 
        
        #evaluate the interpolators over energy, ra, and dec
        for i, e in enumerate(energies):

            engs = np.repeat(e, lats.size)

            slice_points = tuple((engs, lats, lons))

            if self._is_log10:

                #NOTE: if interpolation is carried using the log10 scale, ensure that values outside
                #range of interpolation remain zero after conversion to linear scale.
                #because if function returns zero, 10**(0) = 1. This affects the fit in 3ML and breaks things.

                log_interpolated_slice = interpolator(slice_points)

                interpolated_slice = np.array([0. if x==0. else np.power(10., x)
                                                for x in log_interpolated_slice])

            else:

                interpolated_slice = interpolator(slice_points)
            
            f_interpolated[i] = interpolated_slice

        assert np.all(np.isfinite(f_interpolated)), ("some values are wrong!")

        values = f_interpolated

        return values

    def _setup(self):
        
        #TODO: _setup is currently not being called automatically within astromodels
        #NOTE: for now, it requires to be called after instanciating the class
 
        self._frame = "ICRS"

    def _set_units(self, x_unit, y_unit, z_unit, w_unit):

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit

        #self.K.unit = (u.MeV * u.cm**2 * u.s * u.sr)**(-1)
        self.K.unit = (u.sr)**(-1) #keep this units to if templates have been normalized

    def evaluate(self, x, y, z, K, lon0, lat0, *args):

        lons = x
        lats = y
        energies = z

        angsep = angular_distance_fast(lon0, lat0, lons, lats)

        #if only one energy is passed, make sure we can iterate just once
        if not isinstance(energies, np.ndarray):

            energies = np.array(energies)

        #transform energy from keV to MeV
        #galprop likes MeV, 3ML likes keV
        log_energies = np.log10(energies) - np.log10((u.MeV.to("keV")/u.keV).value)

        #A = np.multiply(K, self._interpolate(lons, lats, log_energies, args)/(10**convert_val))
        A = np.multiply(K, self._interpolate(log_energies, lats, lons, args)) #if templates are normalized no need to convert back

        return A.T

    def set_frame(self, new_frame):
        """ 
        Set a new frame for the coordinates (the default is ICRS J2000)
        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """

        assert  isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame

    @property
    def data_file(self):

        return self._data_file

    def to_dict(self, minimal=False):

        data = super(Function3D, self).to_dict(minimal)

        if not minimal:

            data["extra_setup"] = {"_frame": self._frame,
                                    "ramin": self.ramin,
                                    "ramax": self.ramax,
                                    "decmin": self.decmin,
                                    "decmax": self.decmax
                                  }

        return data
    
    #Define the region within the template ROI
    def define_region(self, a, b, c, d, galactic=False):

        if galactic:

            lmin = a
            lmax = b
            bmin = c
            bmax = d

            _coord = SkyCoord(l=[lmin, lmin, lmax, lmax],
                            b=[bmin, bmax, bmax, bmin],
                            frame="galactic")

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
