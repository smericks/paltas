# -*- coding: utf-8 -*-
"""
Functions for producing and manipulating the posterios produced by the
networks.

This module contains functions to manipulate posteriors generated by our
networks as well as to test the calibration of those posteriors.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def gen_coverage_plots(y_pred,y_test,std_pred,parameter_names,num_lenses=None,
	color_map=["#377eb8", "#4daf4a","#e41a1c","#984ea3"],block=True,
	fontsize=20,show_error_bars=True,n_rows=4):
	""" Generate plots for the 1D coverage of each parameter.

	Args:
		y_pred (np.array): A (batch_size*num_params) array containing the
			mean prediction for each Gaussian
		y_test (np.array): A (batch_size*num_params) array containing the
			true value of the parameter on the test set.
		std_pred (np.array): A (batch_size*num_params) array containing the
			predicted standard deviation for each parameter.
		parameter_names ([str,...]): A list of the parameter names to be
			printed in the plots.
		num_lenses (int): The number of lenses to include in the coverage
			plots. If None all the lenses will be used.
		color_map ([str,...]): A list of at least 4 colors that will be used
			for plotting the different coverage probabilities.
		block (bool): If true, block excecution after plt.show() command.
		fontsize (int): The fontsize to use for the parameter names.
		show_error_bars (bool): If true plot the error bars on the coverage
			plot.
		n_rows (int): The number of rows to include in the subplot.
	"""
	num_params = len(parameter_names)
	error = y_pred - y_test
	# Define the covariance masks for our coverage plots.
	cov_masks = [np.abs(error)<=std_pred,np.abs(error)<2*std_pred,
		np.abs(error)<3*std_pred, np.abs(error)>=3*std_pred]
	cov_masks_names = ['1 sigma =', '2 sigma =', '3 sigma =', '>3 sigma =']
	for i in range(len(parameter_names)):
		plt.subplot(n_rows, int(np.ceil(num_params/n_rows)), i+1)
		# Plot the datapoints in each coverage interval seperately.
		for cm_i in range(len(cov_masks)-1,-1,-1):
			cov_mask = cov_masks[cm_i][:,i]
			yt_plot = y_test[cov_mask,i]
			yp_plot = y_pred[cov_mask,i]
			ys_plot = std_pred[cov_mask,i]
			# Plot with errorbars if requested
			if show_error_bars:
				plt.errorbar(yt_plot,yp_plot,yerr=ys_plot, fmt='.',
					c=color_map[cm_i],
					label=cov_masks_names[cm_i]+'%.2f'%(
						np.sum(cov_mask)/len(error)))
			else:
				plt.errorbar(yt_plot,yp_plot,fmt='.',c=color_map[cm_i],
					label=cov_masks_names[cm_i]+'%.2f'%(
						np.sum(cov_mask)/len(error)))

		# Include the correlation coefficient squared value in the plot.
		_, _, rval, _, _ = linregress(y_test[:,i],y_pred[:,i])
		straight = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
		plt.plot(straight, straight, label='',color='k')
		plt.text(0.8*np.max(straight)+0.2*np.min(straight),np.min(straight),
			'$R^2$: %.3f'%(rval**2),{'fontsize':fontsize})
		plt.title(parameter_names[i],fontsize=fontsize)
		plt.ylabel('Prediction',fontsize=fontsize)
		plt.xlabel('True Value',fontsize=fontsize)
		plt.legend(**{'fontsize':fontsize},loc=2)
	plt.show(block)