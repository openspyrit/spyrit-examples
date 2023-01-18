#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:06:19 2020

@author: crombez
"""

import os
import sys
import glob

import numpy as np
from PIL import Image

def Files_names(Path,name_type):
    files = glob.glob(Path+name_type)
    print
    files.sort(key=os.path.getmtime)
    return([os.path.basename(x) for x in files])
    
def Select_data(Dir, Run):
    Data_path = Dir+Run+'/'
    name_type = os.listdir(Data_path)[0]
    name_type = name_type[:-9]+'*'+name_type[-4:]
    files = Files_names(Data_path,str(name_type))
    print('Données du dossier : '+Data_path)
    with open(Dir+'info_run',"r") as fichier:
        print(fichier.read())
    return(Data_path,files)

def change_set_acqui(*args):
    n = len(set_acqui.value)
    run.options = [x[0][n:] for x in os.walk(set_acqui.value)][1:]



def load_hyper_cube(Path_files,list_files,Nl,Nc,Nd):
	
	"""Load the hyper cube (h,y,λ) from the chSPISIM acquisitions
    
	Args:
		Path_files (string): path of the directory where the data are stored
		list_files (list) : list of strings that contained the names of the negative and positive part of the spatio-spectrum acquisition of a 1D modulated Walsh-Hadamard acquisition
		Nl (int): number of element along the y dimensions of the hyperspectral cube
		Nc (int): number of element along the h dimensions of the hyperspectral cube (the h dimensions correpond to the Walsh-Hadamard transformed of the x dimension
		Nd (int): number of element along the λ dimensions of the hyperspectral cube


	Returns:
		np.ndarray: (Nl,Nc,Nd) hyper cube 

	Example :
		>>> from PIL import Image
		>>> import numpy as np
		>>> Path_files = "home/Document/Data/"
		>>> list_files = ["Files_1.tiff","Files_2.tiff","Files_3.tiff","Files_4.tiff","Files_5.tiff","Files_6.tiff","Files_7.tiff","Files_8.tiff"]
		>>> Nl = 2160
		>>> Nc = 128
		>>> Nd = 2560
		>>> Data = load_hyper_cube(Path_files,list_files,Nl,Nc,Nd)	
	
	"""

	Data = np.zeros((Nl,Nc,Nd))

	for i in range(0,2*Nc,2):

		Data[:,i//2] = np.rot90(np.array(Image.open(Path_files+list_files[i])))-np.rot90(np.array(Image.open(Path_files+list_files[i+1]))) #rotate the (x,λ) image and make the difference between to patern (the rotation depend how the data are stored)

	return(Data)


   
