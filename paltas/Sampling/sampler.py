# -*- coding: utf-8 -*-
"""
Define the sampler class which reads goes from the distribution dictionary to
drawing samples on the lens parameters.

This module contains the class used to sample parameters for our train and test
set from the input distributions.
"""
import warnings
import numpy as np
import pandas as pd
# Define the components we need the sampler to consider.
lensing_components = ['subhalo','los','main_deflector','source','lens_light',
	'point_source','lens_equation_solver','cosmology','psf','detector','drizzle',
	'pixel_grid']

# Global filters on the python warnings. Using this since filter
# behaviour is a bit weird.
CROSSOBJECTWARNING = True


class Sampler():
	"""Class for drawing lens parameter values from input distribution
	dictionaries.

	Args:
		configuration_dictionary (dict): An instance of the configuration
			dictionary that will be used to decide how to sample parameter
			values.
	Notes:
		For examples on how to configure the dict object, see the
		ConfigurationDict.ipynb example notebook.
	"""

	def __init__(self,configuration_dictionary):
		self.config_dict = configuration_dictionary
		self.indices = None

	@staticmethod
	def draw_from_dict(draw_dict):
		"""Populates a dict with samples drawn from the specified distributions
		in the input dict.

		Args:
			draw_dict (dict): The dictionary containing keys mapping to values
				or distributions for each parameter.

		Returns:
			(dict): A dict with a drawn value for each parameter.

		Notes:
			Multivariate distribution for parameters should have a key of the
			form 'param_1,param_2,param_3'.
		"""
		param_dict = {}

		# Iterate through the keys in the draw_dict and populate the values of
		# param_dict correctly.
		for key in sorted(draw_dict):
			# If the key implies that multiple parameters will be drawn from
			# the distribution, draw the value and then iterate through the
			# parameters.
			if ',' in key:
				# Get the parameters, removing whitespace
				params = key.replace(' ','').split(',')
				# Draw the values
				draw = draw_dict[key]()
				# Check for consistency
				if len(params) != len(draw):
					raise ValueError('Parameters of length %d do'%(len(params))
						+ ' not match draw of length %d'%(len(draw)))
				# Populate the keys
				for i, param in enumerate(params):
					param_dict[param] = draw[i]
			# If it's a univariate function just call it.
			elif callable(draw_dict[key]):
				param_dict[key] = draw_dict[key]()
			# If it's a fixed value just populate it.
			else:
				param_dict[key] = draw_dict[key]

		return param_dict

	def sample(self):
		"""Samples from the distributions given in the configuration
		dictionary.

		Returns:
			(dict): A dictionary containing the parameter values that will
				be sampled.
		"""
		# Pull the global warning variable and initialize our dict
		global CROSSOBJECTWARNING
		full_param_dict = {}

		# For each possible component of our lensing add the parameters
		for component in lensing_components:
			if component in self.config_dict:
				draw_dict = self.config_dict[component]['parameters']
				param_dict = self.draw_from_dict(draw_dict)
				full_param_dict[component+'_parameters'] = param_dict

		# Populate parameters from distributions that span accross objects.
		if 'cross_object' in self.config_dict:
			draw_dict = self.config_dict['cross_object']['parameters']
			cross_dict = self.draw_from_dict(draw_dict)
			# Go through the params and update the full dict
			for cross_param in cross_dict:
				component, param = cross_param.split(':')
				param_dict = full_param_dict[component+'_parameters']
				# Warn the user the first time an overwrite happens
				if param in param_dict and CROSSOBJECTWARNING:
					warnings.warn('Parameter %s in cross dict specified '%(param)
						+ 'elsewhere! Will be overwritten. This warning only ' +
						'flags once, but other parameters may also be ' +
						'overwritten.')
					CROSSOBJECTWARNING = False
				full_param_dict[component+'_parameters'][param] = (
					cross_dict[cross_param])
				
		# Populate the cross objects
		return full_param_dict
	
	def get_sample_dist(self):
		"""Samples from the distributions given in the configuration
		dictionary.

		Returns:
			(dict): A dictionary containing the parameter values that will
				be sampled.

		Notes:
		# Author: Padma Venkatraman
		# GitHub: padma18-vb
		# email: pv10@illinois.edu
		"""
		# Pull the global warning variable and initialize our dict
		global CROSSOBJECTWARNING
		full_param_dict = {}

		# For each possible component of our lensing add the parameters
		for component in lensing_components:
			if component in self.config_dict:
				draw_dict = self.config_dict[component]['parameters']
				full_param_dict[component] = draw_dict
		
		# Populate parameters from distributions that span accross objects.
		if 'cross_object' in self.config_dict:
			draw_dict = self.config_dict['cross_object']['parameters']
			# Go through the params and update the full dict
			for cross_param in self.config_dict['cross_object']['parameters']:
				component, param = cross_param.split(':')
				full_param_dict[component][param] = draw_dict['cross_object']['parameters'][cross_param]

		return full_param_dict
	

	def catalog_sample(self, index):
		"""Samples parameters from the catalog file given in the config.

		Args:
			index (int): None, single integer value, or range of values to indicate specific rows in the catalog to draw from. 
				If none, will sample all rows.

		Returns:
			(dict): A dictionary containing the parameter values that will
				be sampled.
			
		Notes:
		# Author: Padma Venkatraman
		# GitHub: padma18-vb
		# email: pv10@illinois.edu
		"""
		path_to_deflectors = self.config_dict['main_deflector']['file']
		deflectors_catalog = pd.read_csv(path_to_deflectors, index_col=0)
		# initially, the indices array will be empty (or None); populate it
		if self.indices is None:
			# if the provided index is a container of indices
			if isinstance(index,(list,pd.core.series.Series,np.ndarray)):
				print('index:')
				print(index)
				self.indices = np.array(index)
			# if the provided index is just one index (int)
			if isinstance(index,int):
				print('index:')
				print(index)
				self.indices = np.array([index])
			# if no specific indices are provides, sample all indices
			else:
				print('index:')
				print(index)
				print("Initializing index array - using all indices from catalog.")
				self.indices = np.array(deflectors_catalog.index)
		full_param_dict = {}
		# set the current index equal to first index in self.indices
		#print('self.indices:')
		#print(self.indices)
		index = self.indices[0]
		# delete the index that you just used
		self.indices = np.delete(self.indices, 0)
		#print('self.indices:')
		#print(self.indices)

		# For each possible component of our lensing add the parameters
		for component in lensing_components:
			if component in self.config_dict:
				param_dict = self.draw_from_catalog(component, index)
				full_param_dict[component+'_parameters'] = param_dict
                
		# store this index in the metadata
		full_param_dict['obj_index'] = index
		return full_param_dict
	
	def draw_from_catalog(self, component, index):
		"""Populates a dict with samples drawn from the specified distributions
		in the input dict.

		Args:
			component (str): Lensing component from config_dict.
			index (int): None, single integer value, or range of values to indicate specific rows in the catalog to draw from

		Returns:
			(dict): A dict with a drawn value for each parameter.

		Notes:
		# Author: Padma Venkatraman
		# GitHub: padma18-vb
		# email: pv10@illinois.edu
		"""
		param_dict = {}
		
		draw_from_file = self.config_dict[component]['file']
		draw_dict= self.config_dict[component]['parameters']
		if draw_from_file is not None:
			comp_table = pd.read_csv(draw_from_file, index_col=0)
			draw_from = comp_table.loc[index, :]
		else:
			draw_from = None

		# Iterate through the keys in the draw_dict and populate the values of
		# param_dict correctly.
		for key in sorted(draw_dict):
			# If the key implies that multiple parameters will be drawn from
			# the distribution, draw the value and then iterate through the
			# parameters.
			
			if callable(draw_dict[key]):
				param_dict[key] = draw_dict[key]()
			elif type(draw_dict[key])==str:
				try:
					# look for the value in the catalog
					param_dict[key] = draw_from.loc[draw_dict[key]]
				except:
					# if it doesn't exist, then the key must be PSF / cosmology / detector
					param_dict[key] = draw_dict[key]
			else:
				param_dict[key] = draw_dict[key]

		return param_dict

