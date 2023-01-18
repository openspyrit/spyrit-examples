#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:37:27 2020

@author: crombez
"""
import numpy as np


    
def smooth(y, box_pts): 
	"""Smooth a vectors
    
	Args:
		y (np.darray): 
		box_pts (int) : 

	Returns:
		np.ndarray: 

	Example :
		>>> from PIL import Image
		>>> import numpy as np
		>>> Path_files = "home/Document/Data/"
		>>> list_files = ["Files_1.tiff","Files_2.tiff","Files_3.tiff","Files_4.tiff","Files_5.tiff","Files_6.tiff","Files_7.tiff","Files_8.tiff"]

	"""
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth


def bining_line(Mat,n):
	"""Bin the lines of a 2D matrix
    
	Args:
		Mat (np.darray): Nl by Nc matrix
		n : divid factor of the lines number

	Returns:
		np.ndarray: Nl//n by Nc matrix

	Example :
		>>> import numpy as np
		>>> Mat = np.ones((20,10))
		>>> Mat_bin = bining_line(Mat,2)

	"""

	(Nl,Nc) = np.shape(Mat)
	M_out = np.zeros((Nl//n,Nc))
	for i in range(0,Nl,n):
		for j in range(n):
    			M_out[i//n] += Mat[i+j]
	return(M_out)

def bining_colonne(Mat,n):
	"""Bin the colonnes of a 2D matrix
    
	Args:
		Mat (np.darray): Nl by Nc matrix
		n : divid factor of the colonne number

	Returns:
		np.ndarray: Nl by Nc//n matrix

	Example :

		>>> import numpy as np
		>>> Mat = np.ones((10,20))
		>>> Mat_bin = bining_colonne(Mat,2)

	"""

	(Nl,Nc) = np.shape(Mat)
	M_out = np.zeros((Nl,Nc//n))
	for i in range(0,Nc,n):
		for j in range(n):
			M_out[:,i//n] += Mat[:,i+j]
	return(M_out)
