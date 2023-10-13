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

from matrix_tools import bining_colonne, bining_line


def Files_names(Path,name_type):
    files = glob.glob(Path+name_type)
    print
    files.sort(key=os.path.getmtime)
    return([os.path.basename(x) for x in files])
    
def Select_data(Dir,Run):
    Data_path = Dir+Run+'/'
    name_type = os.listdir(Data_path)[0]
    name_type = name_type[:-9]+'*'+name_type[-4:]
    files = Files_names(Data_path,str(name_type))
    print('Données du dossier : '+Data_path)

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

def load_pattern_pos_neg(Dir, Run, c_bin):
    
    Path_files, list_files = Select_data(Dir,Run)
    Nh = len(list_files)//2
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    
    print(f'Found {Nh} patterns of size {Nl}x{Nc}')
    
    pat_pos = np.zeros((Nh,Nc))
    pat_neg = np.zeros((Nh,Nc))
    
    for i in range(0,2*Nh,2):
        
        print(Path_files+list_files[i])
        print(Path_files+list_files[i+1])    
        
        tmp = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i])))) 
        pat_pos[i//2,:] = np.sum(tmp[1000:1048,:],0)
        
        tmp = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i+1]))))
        pat_neg[i//2,:] = np.sum(tmp[1000:1048,:],0)


    pat_pos = bining_colonne(pat_pos, c_bin)
    pat_neg = bining_colonne(pat_neg, c_bin)
    
    return pat_pos, pat_neg

def load_pattern(Dir, Run, c_bin, r_start=1000, r_end=1048):
    """
    
    All profiles are obtained from images by summation across the same set of rows

    Args:
        Dir (str): Data folder.
        Run (str): Experiment subfolder.
        c_bin (int): Binning factor across columns.
        r_start (int): Row summation starting index.
        r_end (int): Row summation ending index.
        

    Returns:
        pat (ndarray): Pattern matrix containg all measurement profiles.

    """
    
    Path_files, list_files = Select_data(Dir,Run)
    Nh = len(list_files)
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    
    print(f'Found {Nh} patterns of size {Nl}x{Nc}')
    
    pat = np.zeros((Nh,Nc))
    
    for i in range(0,Nh):
        
        print(Path_files+list_files[i]) 
        
        tmp = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i])))) 
        pat[i,:] = np.sum(tmp[r_start:r_end,:],0)

    pat = bining_colonne(pat, c_bin)
    
    return pat


def load_data_pos_neg(Dir, Run, l_start, l_end, l_bin, lambda_bin):
      
    Path_files, list_files = Select_data(Dir,Run)
    
    # get shapes
    Nh = len(list_files)//2
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    Nl_bin =  l_end - l_start
    
    # Load raw data
    print((Nl,Nh,Nc))
    Data_pos = np.zeros((Nl,Nh,Nc))
    Data_neg = np.zeros((Nl,Nh,Nc))
    
    for i in range(0,2*Nh,2):      
        print(Path_files+list_files[i])
        print(Path_files+list_files[i+1])       
        Data_pos[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i]))))
        Data_neg[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i+1]))))
    
    # Crop raw data
    # We only have 2048 lines on the imaging camera we remove 56 lines 
    # at the top and the bottom of the spectrale images
    Data_pos = Data_pos[l_start:l_end,:]
    Data_neg = Data_neg[l_start:l_end,:]
    
    # init output
    Nl_bin = Nl_bin // l_bin 
    Nc_bin = Nc // lambda_bin
    stack_pos = np.zeros((Nl_bin,Nh,Nc_bin))
    stack_neg = np.zeros((Nl_bin,Nh,Nc_bin)) 
    
    # Spectral binning AND spatial binning across lines
    for i in range(Nh):
        tmp = bining_colonne(Data_pos[:,i,:], lambda_bin) 
        stack_pos[:,i] = bining_line(tmp, l_bin)
        #
        tmp = bining_colonne(Data_neg[:,i,:], lambda_bin) 
        stack_neg[:,i] = bining_line(tmp, l_bin)
    
    return stack_pos, stack_neg


def load_data(Dir, Run, l_start, l_end, l_bin, lambda_bin):
      
    Path_files, list_files = Select_data(Dir,Run)
    
    # get shapes
    Nh = len(list_files)
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    Nl_bin =  l_end - l_start
    
    # Load raw data
    print((Nl,Nh,Nc))
    Data = np.zeros((Nl,Nh,Nc))
    
    for i in range(0,Nh):      
        print(Path_files+list_files[i])     
        Data[:,i] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i]))))
    
    # Crop raw data
    # We only have 2048 lines on the imaging camera we remove 56 lines 
    # at the top and the bottom of the spectrale images
    Data = Data[l_start:l_end,:]
    
    # init output
    Nl_bin = Nl_bin // l_bin 
    Nc_bin = Nc // lambda_bin
    stack = np.zeros((Nl_bin,Nh,Nc_bin))
    
    # Spectral binning AND spatial binning across lines
    for i in range(Nh):
        tmp = bining_colonne(Data[:,i,:], lambda_bin) 
        stack[:,i] = bining_line(tmp, l_bin)
    
    return stack