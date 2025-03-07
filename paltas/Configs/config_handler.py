# -*- coding: utf-8 -*-
"""
Interact with the paltas configuration files

Classes used to draw relevant parameters from paltas configuration files.
"""
import os, sys, warnings, copy
from importlib import import_module
import numba
import numpy as np
from ..Sampling.sampler import Sampler
from ..Sources.galaxy_catalog import GalaxyCatalog
from ..Utils.cosmology_utils import get_cosmology, ddt
from ..Utils.hubble_utils import hubblify
from ..Utils.lenstronomy_utils import PSFHelper
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.util as util

# Global filters on the python warnings. Using this since filter
# behaviour is a bit weird.
SERIALIZATIONWARNING = True
KWARGSNUMERICWARNING1 = False
KWARGSNUMERICWARNING2 = False

# Exclude these parameters from the image metadata. These are rarely sampled
# and would bloat the metadata or make it hard to serialize.
EXCLUDE_FROM_METADATA = (
	# Long list of sources that were included
	('source_parameters', 'source_inclusion_list'),
	# Tiny list denoting drizzle pattern
	# (allowing this would complicate check against lists entering metadata)
	('drizzle_parameters', 'offset_pattern'),
	# Path to the COSMOS images. Pointless repeating setup-specific string.
	('source_parameters', 'cosmos_folder'),
)

	
class FailedCriteriaError(Exception):
	"""
    Use this error to skip images that don't pass doubles only, quad only, 
        magnification cut, (etc.) criteria
    """
	def __init__(self):
		# Pass a useful message to base class constructor
		message = 'Criteria for image creation not met.'
		message += 'If this is inteneded (i.e. in a loop) use try/except.'
		super().__init__(message)


class ConfigHandler():
	"""Class that parses the configuration files to extract images and lenstronomy
	configurations.

	Args:
		config_path (str): A path to the config file to parse.
	"""

	def __init__(self,config_path,):
		# Get the dictionary from the provided .py file
		config_dir, config_file = os.path.split(os.path.abspath(config_path))
		sys.path.insert(0, config_dir)
		config_name, _ = os.path.splitext(config_file)
		self.config_module = import_module(config_name)
		self.config_dict = self.config_module.config_dict

		# Get the random seed to use, or draw a random not-too-large one
		# (so it is easy to copy-paste from metadata into config files)
		# Max is 2**32 - 1, see _legacy_seeding in numpy/random/_mt19937.pyx
		self.base_seed = getattr(
			self.config_module,
			'seed',
			(np.random.randint(np.iinfo(np.uint32).max,)))
		# Make sure base_seed is a sequence, not a number
		if isinstance(self.base_seed, (int, float)):
			self.base_seed = (self.base_seed,)
		self.reseed_counter = 0

		# Set up our sampler and draw a sample for initialization
		self.sampler = Sampler(self.config_dict)
		self.sample = None
		self.draw_new_sample()
		sample = self.get_current_sample()

		# Initialize all the parameters and classes we need for drawing
		# lenstronomy inputs and images

		# Get the numerical kwargs numpix from the config
		self.kwargs_numerics = self.config_module.kwargs_numerics
		self.numpix = self.config_module.numpix
		
		# handle doubles_quads_only
		if hasattr(self.config_module, 'doubles_quads_only'):
			self.doubles_quads_only = self.config_module.doubles_quads_only
		else:
			self.doubles_quads_only = False

		# handle quads_only
		if hasattr(self.config_module, 'quads_only'):
			self.quads_only = self.config_module.quads_only
		else:
			self.quads_only = False

		if hasattr(self.config_module,'compute_caustic_area'):
			self.compute_caustic_area = self.config_module.compute_caustic_area
		else:
			self.compute_caustic_area = False

		if hasattr(self.config_module,'compute_mass_enclosed'):
			self.compute_mass_enclosed = self.config_module.compute_mass_enclosed
		else:
			self.compute_mass_enclosed = False
			
		if hasattr(self.config_module, 'ps_magnification_cut'):
			self.ps_magnification_cut = self.config_module.ps_magnification_cut
		else:
			self.ps_magnification_cut = None
			
		if hasattr(self.config_module, 'magnification_limit'):
			self.magnification_limit = self.config_module.magnification_limit
		else:
			self.magnification_limit = None

		# Set up the paltas objects we'll use
		self.los_class = None
		self.subhalo_class = None
		self.main_deflector_class = None
		self.lens_light_class = None
		self.point_source_class = None
		self.do_drizzle = False
		if 'los' in self.config_dict:
			self.los_class = self.config_dict['los']['class'](
				sample['los_parameters'],sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
		if 'subhalo' in self.config_dict:
			self.subhalo_class = self.config_dict['subhalo']['class'](
				sample['subhalo_parameters'],sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
		if 'main_deflector' in self.config_dict:
			self.main_deflector_class = (
				self.config_dict['main_deflector']['class'](
					sample['main_deflector_parameters'],
					sample['cosmology_parameters']))
		if 'drizzle' in self.config_dict:
			self.do_drizzle = True
		if 'lens_light' in self.config_dict:
			self.lens_light_class = self.config_dict['lens_light']['class'](
				sample['cosmology_parameters'], sample['lens_light_parameters'])
		if 'point_source' in self.config_dict:
			self.point_source_class = self.config_dict['point_source']['class'](
				sample['cosmology_parameters'],sample['point_source_parameters'])

		# We always need a source class
		self.source_class = self.config_dict['source']['class'](
			sample['cosmology_parameters'],sample['source_parameters'])

		# See if a magnification cut was specified
		if hasattr(self.config_module, 'mag_cut'):
			self.mag_cut = self.config_module.mag_cut
		else:
			self.mag_cut = None

		# See if ignoring noise was specified
		if hasattr(self.config_module,'no_noise'):
			self.add_noise = not self.config_module.no_noise
		else:
			self.add_noise = True

	def draw_new_sample(self):
		"""Draws a new sample from the config sampler.
		"""
		self.sample = self.sampler.sample()

	def get_current_sample(self):
		"""Returns the current sample from the config sampler.

		Returns:
			(dict): The current sample
		"""
		return self.sample

	def get_lenstronomy_models_kwargs(self,new_sample=True):
		"""Takes a sample from the config and returns the list of lenstronomy
		models, kwargs, and redshifts for the lensing system.

		Args:
			new_sample (bool): If true will draw a new sample from the config
				sampler before returning the lenstronomy kwargs. True by
				default.

		Returns:
			(dict,dict): Two dicts, the first containing the list of lens
			models, lens model redshifts, source light models, source redshifts,
			lens light models, and point source models. The second contains the
			lens kwargs, the source kwargs, the point source kwargs, and the
			lens light kwargs.

		Notes:
			Even if new_sample is False, this function is not guaranteed to be
			deterministic. For example, most of the substructure classes draw
			from populations specified by the input parameters, and therefore
			calling the function repeatedly will return different realizations
			of that population.
		"""
		# Either draw a new sample or use the current sample.
		if new_sample:
			self.draw_new_sample()
		sample = self.get_current_sample()

		# Populate the list of models and kwargs lenstronomy needs
		complete_lens_model_list = []
		complete_lens_model_kwargs = []
		complete_z_list = []
		lens_light_model_list = []
		lens_light_kwargs_list = []
		point_source_model_list = []
		point_source_kwargs_list = []
		source_model_list = []
		source_kwargs_list = []
		source_redshift_list = []

		# For each lensing object that's present, add them to the model and
		# kwargs list
		if self.los_class is not None:
			self.los_class.update_parameters(
				sample['los_parameters'],sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
			los_model_list, los_kwargs_list, los_z_list = (
				self.los_class.draw_los())
			interp_model_list, interp_kwargs_list, interp_z_list = (
				self.los_class.calculate_average_alpha(self.numpix*
					self.kwargs_numerics['supersampling_factor']))
			complete_lens_model_list += los_model_list + interp_model_list
			complete_lens_model_kwargs += los_kwargs_list + interp_kwargs_list
			complete_z_list += los_z_list + interp_z_list
		if self.subhalo_class is not None:
			self.subhalo_class.update_parameters(
				sample['subhalo_parameters'],
				sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
			sub_model_list, sub_kwargs_list, sub_z_list = (
				self.subhalo_class.draw_subhalos())
			complete_lens_model_list += sub_model_list
			complete_lens_model_kwargs += sub_kwargs_list
			complete_z_list += sub_z_list
		if self.main_deflector_class is not None:
			self.main_deflector_class.update_parameters(
				sample['main_deflector_parameters'],sample['cosmology_parameters'])
			main_model_list, main_kwargs_list, main_z_list = (
				self.main_deflector_class.draw_main_deflector())
			complete_lens_model_list += main_model_list
			complete_lens_model_kwargs += main_kwargs_list
			complete_z_list += main_z_list
		if self.lens_light_class is not None:
			self.lens_light_class.update_parameters(
				cosmology_parameters=sample['cosmology_parameters'],
				source_parameters=sample['lens_light_parameters'])
			lens_light_model_list, lens_light_kwargs_list, _ = (
				self.lens_light_class.draw_source())
		if self.point_source_class is not None:
			self.point_source_class.update_parameters(
				cosmology_parameters=sample['cosmology_parameters'],
				point_source_parameters=sample['point_source_parameters'])
			point_source_model_list,point_source_kwargs_list = (
				self.point_source_class.draw_point_source())

		# Now get the model and kwargs from the source class (which must
		# always be present.)
		self.source_class.update_parameters(
			cosmology_parameters=sample['cosmology_parameters'],
			source_parameters=sample['source_parameters'])

		# For catalog objects we also want to save the catalog index
		# and the (possibly randomized) additional rotation angle. We will
		# therefore push these back into the sample object.
		if isinstance(self.source_class,GalaxyCatalog):
			catalog_i, phi = self.source_class.fill_catalog_i_phi_defaults()
			source_model_list, source_kwargs_list, source_redshift_list = (
				self.source_class.draw_source(catalog_i=catalog_i, phi=phi))
			sample['source_parameters']['catalog_i'] = catalog_i
			sample['source_parameters']['phi'] = phi
		else:
			source_model_list, source_kwargs_list, source_redshift_list = (
				self.source_class.draw_source())

		# Check to see if we need multiplane
		multi_plane = False
		if (len(np.unique(source_redshift_list))>1 or
			len(np.unique(complete_z_list))>1):
			multi_plane = True

		# Package all of the lists into a model and parameters dictionary.
		kwargs_model = {}
		kwargs_params = {}
		kwargs_model['lens_model_list'] = complete_lens_model_list
		kwargs_params['kwargs_lens'] = complete_lens_model_kwargs
		kwargs_model['lens_redshift_list'] = complete_z_list
		kwargs_model['lens_light_model_list'] = lens_light_model_list
		kwargs_params['kwargs_lens_light'] = lens_light_kwargs_list
		kwargs_model['point_source_model_list'] = point_source_model_list
		kwargs_params['kwargs_ps'] = point_source_kwargs_list
		kwargs_model['source_light_model_list'] = source_model_list
		kwargs_params['kwargs_source'] = source_kwargs_list
		kwargs_model['source_redshift_list'] = source_redshift_list
		kwargs_model['multi_plane'] = multi_plane

		# The source convention is definied by the source parameters. This is
		# also what the lens model classes use when setting their parameters.
		kwargs_model['z_source'] = sample['source_parameters']['z_source']
		kwargs_model['z_source_convention'] = kwargs_model['z_source']

		return kwargs_model, kwargs_params

	def get_metadata(self):
		"""Returns the values drawn from the configuration file to generate
		the current sample.

		Returns:
			(dict): A dictionary containing the values drawn from the
			configuration file to generate the current sample. This includes
			the parameters of the lensing system (some of which may be
			population level parameters), the parameters of the observation,
			the cosmology, and any other parameters specified within the
			config_dict of the input configuration file.

		Notes:
			The metadata naming scheme is object_parameters_name_of_parameters.
			For example for the Einstein radius of the main deflector the
			key is main_deflector_parameters_theta_E. For the redshift of
			the source it would be source_parameters_z_source.
		"""
		# Setup the warning filter
		global SERIALIZATIONWARNING

		# Get the samples and the metadata.
		sample = self.get_current_sample()
		metadata = {}

		for component in sample:
			for key in sample[component]:
				comp_value = sample[component][key]
				# Make sure that lists and other objects that cannot be
				# serialized well are not written out. Warn about this only
				# once.
				if (component, key) in EXCLUDE_FROM_METADATA:
					continue
				if isinstance(comp_value,bool):
					metadata[component+'_'+key] = int(comp_value)
				elif isinstance(comp_value, (str, int, float)) or comp_value is None:
					metadata[component+'_'+key] = comp_value
				elif SERIALIZATIONWARNING:
					warnings.warn(
						f'Parameter ({component}, {key}) in config_dict, '
						'and possibly others, will not be written to '
						'metadata.csv',
						category=RuntimeWarning)
					SERIALIZATIONWARNING = False
					
		# write caustic area if requested
		if self.compute_caustic_area:
			kwargs_model, kwargs_params = self.get_lenstronomy_models_kwargs(
				new_sample=False)
			lm = LensModel(kwargs_model['lens_model_list'])
			lm_ext = LensModelExtensions(lm)
			compute_window = self.numpix * sample['detector_parameters']['pixel_scale']
			metadata['main_deflector_parameters_caustic_area'] = lm_ext.caustic_area(
				kwargs_params['kwargs_lens'],{'compute_window':compute_window})
			
			# also write minimum separation btwn. caustic & source
			_,_,ra_list,dec_list = lm_ext.critical_curve_caustics(
				kwargs_params['kwargs_lens'],compute_window=compute_window)
			
			# index into list is for "selecting specified vortex"? (default is 0)
			ra_offset = kwargs_params['kwargs_source'][0]['center_x'] - ra_list[0]
			dec_offset = kwargs_params['kwargs_source'][0]['center_y'] - dec_list[0]
			min_distance = np.sqrt(np.min(ra_offset**2 + dec_offset**2))
			metadata['source_parameters_distance_to_caustic'] = min_distance
			
		if self.compute_mass_enclosed:
			kwargs_model, kwargs_params = self.get_lenstronomy_models_kwargs(
				new_sample=False)
			lm = LensModel(kwargs_model['lens_model_list'])
			# r is in arcsec
			mass_enclosed = lm.lens_model.func_list[0].mass_2d_lens(r=1.5,
				sigma0=kwargs_params['kwargs_lens'][0]['sigma0'],
				r_core=kwargs_params['kwargs_lens'][0]['r_core'], 
				gamma=kwargs_params['kwargs_lens'][0]['gamma'])
			metadata['main_deflector_parameters_M_encl_15e-1arcsec'] = mass_enclosed


		return metadata

	def get_sample_cosmology(self,as_astropy=False):
		"""Return the cosmology object for the current sample.

		Args:
			as_astropy (bool): If True, will return an astropy cosmology
				object instead of a colossus cosmology object. Defaults
				to False.
		Returns:
			(colossus.cosmology.cosmology.Cosmology): An instance of
			the colossus cosmology class. If as_astropy is True, this
			will be an astropy object instead.
		"""
		# Grab the cosmology from the sample
		sample = self.get_current_sample()
		cosmo = get_cosmology(sample['cosmology_parameters'])
		if as_astropy is True:
			return cosmo.toAstropy()
		else:
			return cosmo

	def _calculate_ps_metadata(self,metadata,kwargs_params,point_source_model,
		lens_model):
		"""Calculate time delays and image positions and appends them to
		metadata

		Args:
			metadata (dict): A dictionary containing the metadata that
				will be modified in place.
			lenstronomy_dict (dict):  A dictionary containing the list of model
				kwargs. See get_lenstronomy_model_kwargs.
			point_source_model (lenstronomy.PointSource.point_source.PointSource):
				An instance of the lenstronomy point source model that will be
				used to calculate lensing quantitities.
			lens_model (lenstronomy.LensModel.lens_model.LensModel): An instance
				of the lenstronomy lens model that will be used to calculate
				lensing quantitities.

		Return:
			returns 0 if successful, -1 if failed
		"""
		# Extract the sample
		sample = self.get_current_sample()
		cosmo = self.get_sample_cosmology()

		# Calculate image positions
		x_image, y_image = point_source_model.image_position(
			kwargs_params['kwargs_ps'],
			kwargs_params['kwargs_lens'])
		num_images = len(x_image[0])

		# Append to the metadata using the same prefix as the rest of the
		# point source parameters
		pfix = 'point_source_parameters_'
		metadata[pfix+'num_images'] = num_images
		
		# write offset between lens center & PS position
		x_diff = (kwargs_params['kwargs_ps'][0]['ra_source'] 
			- kwargs_params['kwargs_lens'][0]['center_x'])
		y_diff = (kwargs_params['kwargs_ps'][0]['dec_source'] 
	    	- kwargs_params['kwargs_lens'][0]['center_y'])
		metadata[pfix+'lens_ps_offset'] = np.sqrt(x_diff**2 + y_diff**2)
		
		# throw error if num images > 5
		if num_images > 5:
			raise FailedCriteriaError()
		
		# it's no longer a lens in this case!
		if num_images < 2:
			raise FailedCriteriaError()

		if self.doubles_quads_only and num_images != 2 and num_images != 4:
			raise FailedCriteriaError()

		# throw error if not quad & requested quads only
		if self.quads_only and num_images != 4:
			raise FailedCriteriaError()

		# Calculate magnifications using complete_lens_model
		magnifications = lens_model.magnification(x_image[0],y_image[0],
			kwargs_params['kwargs_lens'])
		# If mag_pert is defined, add that pertubation
		if 'mag_pert' in sample['point_source_parameters'].keys():
			magnifications = magnifications * (
				sample['point_source_parameters']['mag_pert'][
					0:len(magnifications)])
			
		# throw error if does not pass point source magnification cut
		if self.ps_magnification_cut is not None:
			avg_magnification = np.mean(np.abs(magnifications))
			if avg_magnification < self.ps_magnification_cut:
				raise FailedCriteriaError()

		# Calculate time delays
		if sample['point_source_parameters']['compute_time_delays']:
			if 'kappa_ext' in sample['point_source_parameters'].keys():
				td = lens_model.arrival_time(x_image[0],y_image[0],
					kwargs_params['kwargs_lens'],
					sample['point_source_parameters']['kappa_ext'])
				# Apply errors if defined in config_dict
				if 'time_delay_errors' in (
					sample['point_source_parameters'].keys()):
					errors = sample['point_source_parameters'][
						'time_delay_errors']
					errors = errors[0:len(td)-1]
					td = td + errors
				td -= td[0]
			else:
				raise ValueError('must define kappa_ext in point_source ' +
					'parameters to compute time delays')

			# Calculate time delay distance
			metadata[pfix+'ddt'] = ddt(sample,cosmo)

		# Write out all of the metadata.
		for i in range(0,4):
			if i < num_images:
				metadata[pfix+'x_image_'+str(i)] = x_image[0][i]
				metadata[pfix+'y_image_'+str(i)] = y_image[0][i]
				metadata[pfix+'magnification_'+str(i)] = magnifications[i]
				if 'mag_pert' in sample['point_source_parameters'].keys():
					metadata[pfix+'mag_pert_'+str(i)] = sample['point_source_parameters']['mag_pert'][i]
			else:
				metadata[pfix+'x_image_'+str(i)] = np.nan
				metadata[pfix+'y_image_'+str(i)] = np.nan
				metadata[pfix+'magnification_'+str(i)] = np.nan
				if 'mag_pert' in sample['point_source_parameters'].keys():
					metadata[pfix+'mag_pert_'+str(i)] = np.nan

			if sample['point_source_parameters']['compute_time_delays']:
				if i < len(td):
					metadata[pfix+'time_delay_' + str(i)] = td[i]
				else:
					metadata[pfix+'time_delay_' + str(i)] = np.nan
					
		return 0

	def _draw_image_standard(self,add_noise=True,apply_psf=True):
		"""Uses the current config sample to generate an image and the
		associated metadata.

		Args:
			add_noise (bool): If False, noise will not be added. Defaults to
				True.
			apply_psf (bool): If False, the psf will not be applied. Defaults
				to True.
		Returns:
			(np.array,dict): A tuple containing a numpy array of the generated
			image and a metavalue dictionary with the corresponding sampled
			values.
		Notes:
			Will raise an error if the produced image does not meet a cut.
		"""
		# Get the lenstronomy parameters and the sample
		sample = self.get_current_sample()
		kwargs_model, kwargs_params = self.get_lenstronomy_models_kwargs(
			new_sample=False)

		# Get the psf, detector, and pixel grid parameters from the sample
		kwargs_psf = sample['psf_parameters']
		kwargs_detector = sample['detector_parameters']
		kwargs_pixel_grid = None
		if 'pixel_grid_parameters' in sample.keys():
			kwargs_pixel_grid = sample['pixel_grid_parameters']

		# Build the psf model
		if apply_psf:
			psf_model = PSF(**kwargs_psf)
		else:
			psf_model = PSF(psf_type='NONE')

		# Build the data and noise models we'll use.
		data_api = DataAPI(numpix=self.numpix,
			kwargs_pixel_grid=kwargs_pixel_grid,**kwargs_detector)
		single_band = SingleBand(**kwargs_detector)
		
		# Pull the cosmology and source redshift
		cosmo = get_cosmology(sample['cosmology_parameters'])

		# Build our lens and source models.
		lens_model = LensModel(kwargs_model['lens_model_list'],
			z_source=kwargs_model['z_source'],
			z_source_convention=kwargs_model['z_source_convention'],
			lens_redshift_list=kwargs_model['lens_redshift_list'],
			cosmo=cosmo.toAstropy(),multi_plane=kwargs_model['multi_plane'])
		source_light_model = LightModel(kwargs_model['source_light_model_list'],
			source_redshift_list=kwargs_model['source_redshift_list'])
		lens_light_model = LightModel(kwargs_model['lens_light_model_list'])

		# Point source may need lens eqn solver kwargs
        # Need to fix how fixed_magnification_list is handled
		lens_equation_params = None
		if 'lens_equation_solver_parameters' in sample.keys():
			lens_equation_params = sample['lens_equation_solver_parameters']
		point_source_model = PointSource(
			kwargs_model['point_source_model_list'],lens_model=lens_model,
			save_cache=True,kwargs_lens_eqn_solver=lens_equation_params,
			magnification_limit=self.magnification_limit,
            fixed_magnification_list=[True])

		# Put it together into an image model
		image_model = ImageModel(data_api.data_class,psf_model,
			lens_model,source_light_model,lens_light_model,
			point_source_model,kwargs_numerics=self.kwargs_numerics)

		# Generate our image
		image = image_model.image(kwargs_params['kwargs_lens'],
			kwargs_params['kwargs_source'],
			kwargs_params['kwargs_lens_light'],
			kwargs_params['kwargs_ps'])

		# Check for the magnification cut and apply it.
        # TODO: these assumptions break down w/ a point source in the model
		if self.mag_cut is not None:
			# Evaluate the light that would have been in the image using
			# the image model
			lens_light_total = np.sum(image_model.lens_surface_brightness(
				kwargs_params['kwargs_lens_light']))
			source_light_total = np.sum(source_light_model.total_flux(
				kwargs_params['kwargs_source']))

			mag = np.sum(image)-lens_light_total
			mag /= source_light_total
			if mag < self.mag_cut:
				raise FailedCriteriaError()

		# If noise is specified, add it.
		if add_noise:
			image += single_band.noise_for_model(image)

		# Extract the metadata from the sample
		metadata = self.get_metadata()

		# If a point source was specified, calculate the time delays
		# and image positions.
		if self.point_source_class is not None:
			try:
				success = self._calculate_ps_metadata(metadata,kwargs_params,
					point_source_model,lens_model)
			except FailedCriteriaError as e:
				raise FailedCriteriaError() from e
			
			# address case w/ 6 PS images
			if success == -1:
				return None,None

		return image, metadata

	def _draw_image_drizzle(self):
		"""Uses the current config sample to generate a drizzled image and the
		associated metadata.

		Returns:
			(np.array,dict): A tuple containing a numpy array of the generated
			image and a metavalue dictionary with the corresponding sampled
			values.
		Notes:
			Will return an error if the produced image does not meet a cut.
			This function will fail if the drizzle parameters are not
			present.
		"""
		# Grab our warning flags
		global KWARGSNUMERICWARNING1
		global KWARGSNUMERICWARNING2

		# Copy the sample since we will be modifying some parameters
		sample_copy = copy.deepcopy(self.sample)
		kwargs_numerics_copy = copy.deepcopy(self.kwargs_numerics)
		numpix_copy = self.numpix

		# Generate a high resolution version of the image.
		supersample_pixel_scale = self.sample['drizzle_parameters'][
			'supersample_pixel_scale']
		output_pixel_scale = (
			self.sample['drizzle_parameters']['output_pixel_scale'])
		detector_pixel_scale = self.sample['detector_parameters']['pixel_scale']
		offset_pattern = self.sample['drizzle_parameters']['offset_pattern']
		wcs_distortion = self.sample['drizzle_parameters']['wcs_distortion']
		ss_scaling = detector_pixel_scale/supersample_pixel_scale
		self.numpix = int(self.numpix*ss_scaling)

		# Temporairly reset the detector pixel_scale to the supersampled scale.
		self.sample['detector_parameters']['pixel_scale'] = (
			supersample_pixel_scale)

		# Modify the numerics kwargs to account for the supersampled pixel scale
		if 'supersampling_factor' in self.kwargs_numerics:
			if KWARGSNUMERICWARNING1:
				warnings.warn('kwargs_numerics supersampling_factor modified '
					+'for drizzle',category=RuntimeWarning)
				KWARGSNUMERICWARNING1 = False
			self.kwargs_numerics['supersampling_factor'] = max(1,
				int(self.kwargs_numerics['supersampling_factor']/ss_scaling))
		if 'point_source_supersampling_factor' in self.kwargs_numerics:
			if KWARGSNUMERICWARNING2:
				warnings.warn('kwargs_numerics point_source_supersampling_factor'
					+ ' modified for drizzle',category=RuntimeWarning)
				KWARGSNUMERICWARNING2 = False
			self.kwargs_numerics['point_source_supersampling_factor'] = max(1,
				int(self.kwargs_numerics['point_source_supersampling_factor']/
					ss_scaling))

		# Use the normal generation class to make our highres image without
		# noise.
		try:
			image_ss, metadata = self._draw_image_standard(add_noise=False,
				apply_psf=False)
		except FailedCriteriaError:
			# Reset the class properties that were modified, then reraise
			self.sample = sample_copy
			self.kwargs_numerics = kwargs_numerics_copy
			self.numpix = numpix_copy
			raise
		
		self.sample['detector_parameters']['pixel_scale'] = detector_pixel_scale
		self.numpix = numpix_copy

		# Grab the PSF supersampling factor if present.
		if 'psf_supersample_factor' in self.sample['drizzle_parameters']:
			psf_supersample_factor = (
				self.sample['drizzle_parameters']['psf_supersample_factor'])
		else:
			warnings.warn('No psf_supersample_factor provided so 1 will be ' +
				'assumed.')
			psf_supersample_factor = 1

		if psf_supersample_factor > ss_scaling:
			raise ValueError(f'psf_supersample_factor {psf_supersample_factor} '
				+ f'larger than the supersampling {ss_scaling} defined in the '
				+'drizzle parameters.')

		# Make sure that if the user provided a PIXEL psf that the user specified
		# the same supersampling factor for the drizzle_parameters
		if self.sample['psf_parameters']['psf_type'] == 'PIXEL' and (
			'point_source_supersampling_factor' not in
			self.sample['psf_parameters'] or
			self.sample['psf_parameters']['point_source_supersampling_factor'] !=
			psf_supersample_factor):
			raise ValueError('Must specify point_source_supersampling_factor for '
				'PIXEL psf and the value must equal psf_supersample_factor in the '
				' drizzle parameters')

		# We'll bypass lenstronomy's supersampling code by modifying
		# the data api.
		self.kwargs_numerics['point_source_supersampling_factor'] = 1
		self.kwargs_numerics['supersampling_factor'] = 1
		self.kwargs_numerics['supersampling_convolution'] = False
		self.sample['psf_parameters']['point_source_supersampling_factor'] = 1
		self.sample['detector_parameters']['pixel_scale'] /= (
			psf_supersample_factor)

		# Create our noise and psf models.
		kwargs_detector = self.sample['detector_parameters']
		kwargs_psf = self.sample['psf_parameters']
		single_band = SingleBand(**kwargs_detector)
		data_class = DataAPI(numpix=self.numpix*psf_supersample_factor,
			**kwargs_detector).data_class
		psf_model_lenstronomy = PSF(**kwargs_psf)

		if self.add_noise:
			noise_model = single_band.noise_for_model
		else:
			def noise_model(image):
				return 0

		# Initialize the psf model we'll pass to hubblify.
		psf_model = PSFHelper(data_class,psf_model_lenstronomy,
			self.kwargs_numerics).psf_model

		# Pass the high resolution image to hubblify
		image = hubblify(image_ss,supersample_pixel_scale,detector_pixel_scale,
			output_pixel_scale,noise_model,psf_model,offset_pattern,
			wcs_distortion=wcs_distortion,pixfrac=1.0,kernel='square',
			psf_supersample_factor=psf_supersample_factor)

		# Reset the class properties that were modified.
		self.sample = sample_copy
		self.kwargs_numerics = kwargs_numerics_copy

		# Update the metadata to account for the original sample
		metadata.update(self.get_metadata())

		return image, metadata

	def draw_image(self,new_sample=True):
		"""Takes a sample from the config and generate an image of the strong
		lensing system along with its metadata.

		Args:
			new_sample (bool): If true will draw a new sample from the config
				sampler before returning the image. True by default.

		Returns:
			(np.array,dict): A tuple containing a numpy array of the generated
			image and a metavalue dictionary with the corresponding sampled
			values.

		Notes:
			Even if new_sample is False, this function is not guaranteed to be
			deterministic. For example, most of the substructure classes draw
			from populations specified by the input parameters, and therefore
			calling the function repeatedly will return images of different
			realizations of that population.
		"""
		# Generate a new random seed for each call to draw_image in order
		# to ensure deterministic behavior
		seed = self.reseed()

		# Draw a new sample if requested
		if new_sample:
			self.draw_new_sample()

		# Use the appropraite generation function
		try:
			if self.do_drizzle:
				image,metadata = self._draw_image_drizzle()
			else:
				# _draw_image_standard has a seperate add_noise parameter so
				# it can be used by _draw_image_drizzle.
				image,metadata = self._draw_image_standard(
					add_noise=self.add_noise)
		except FailedCriteriaError:
			# Image critera not met, return None,None.
			return None, None
		
		# Mask out an interior region of the image if requested
		if hasattr(self.config_module,'mask_radius'):
			kwargs_detector = self.get_current_sample()['detector_parameters']
			x_grid, y_grid = util.make_grid(numPix=image.shape[0],
				deltapix=kwargs_detector['pixel_scale'])
			r = util.array2image(np.sqrt(x_grid**2+y_grid**2))
			image[r<=self.config_module.mask_radius] = 0

		# Save the seed
		metadata['seed'] = seed

		return image,metadata

	def reseed(self):
		"""Generates, sets, and returns a new random seed.

		Returns:
			(tuple): The tuple used to seed numpy.
		"""
		if self.reseed_counter == 0:
			# Use the base seed; perhaps to reproduce one particular image
			seed = self.base_seed
		else:
			# Append the counter to the base_seed tuple to form a new seed
			seed = self.base_seed + (self.reseed_counter,)
		# Seed numpy's random generator. Note this accepts tuples.
		np.random.seed(seed)
		self.reseed_counter += 1
		# Seed numba's separate random generator
		# Unfortunately it only accepts an integer argument
		_set_numba_seed(np.random.randint(np.iinfo(np.uint32).max))
		return seed


# Must be compiled in regular (non-object) mode, see note under
# https://numba.pydata.org/numba-doc/0.22.1/reference/numpysupported.html#random
@numba.njit
def _set_numba_seed(seed):
	"""Reseeds numba's random number generator

	Args:
		seed (int): The integer random seed to use for seeding numba.
	"""
	np.random.seed(seed)