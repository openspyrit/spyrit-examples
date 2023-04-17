#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image, ImageOps
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
#from ..misc.disp import *
from itertools import cycle;
import scipy.linalg as lin




#######################################################################
# 1.1 Determine the important Radon Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best 
# Radon patterns for a given image size
#

def matrix2conv(Matrix):
    """
        Returns Convolution filter where each kernel corresponds to a line of the reshaped
        Matrix
    """
    M = Matrix.shape[0];
    N = Matrix.shape[1];
    img_size =int(round(np.sqrt(N)));

    P = nn.Conv2d(1,M,kernel_size=img_size, stride=1, padding=0);
    P.bias.data=torch.zeros(M);
    for i in range(M):
        pattern = Matrix[i,:].view(img_size, img_size);
        P.weight.data[i,0,:,:] = pattern;
    P.bias.requires_grad = False;
    P.weight.requires_grad=False;
    return P

def compute_radon(img,A):
    Q = img.shape[0];
    D = int(A.shape[0] / 181);
    alpha = np.zeros((Q*Q));
    for i in range(0, Q):
        alpha[i * Q:(i + 1) * Q] = img[i, :];
    m = np.dot(A, alpha);

    R = np.zeros((D, 181))
    for angle in range(0, 181):
        R[:, angle] = m[D * angle:D * (angle + 1)];
    return R


def radonSpecifyAngles(A, nbAngles):
    
    angles = generateAngles(nbAngles)
    
    #nbAngles = angles.shape[0]
    D = (int)(A.shape[0]/181)
    QQ = A.shape[1];

    Areduced = np.zeros([ D*nbAngles , QQ ])
    #Areduced = torch.zeros([ D*nbAngles , QQ ]);
    #Areduced = torch.zeros([ D*181 , QQ ]);

    for theta in range(0, nbAngles):
        Areduced[theta * D : theta * D + D - 1, :] = A[(int)(angles[theta] * D) : (int)(angles[theta] * D + D - 1), :]
        #Areduced[(int)(angles[theta] * D):(int)(angles[theta] * D + D - 1), :] = A[(int)(angles[theta] * D):(int)(
            #angles[theta] * D + D - 1), :]
    return Areduced

def radonSpecifyAngles_np(A,angles):
    nbAngles = angles.shape[0]
    D = (int)(A.shape[0]/181)
    QQ = A.shape[1]

    Areduced = np.zeros([ D*nbAngles , QQ ])

    for theta in range(0, nbAngles):
        Areduced[theta * D : theta * D + D - 1, :] = A [(int)(angles[theta] * D) : (int)(angles[theta] * D + D - 1), :]
        #Areduced[(int)(angles[theta] * D):(int)(angles[theta] * D + D - 1), :] = A[(int)(angles[theta] * D):(int)(
            #angles[theta] * D + D - 1), :]
    return Areduced

def generateAngles(nbAngles):
    angles = np.zeros([nbAngles]);
    for i in range(0, nbAngles):
        angles[i] = (int)(180 * i / nbAngles);
    return angles;

def vector2matrix(vector, d):
    vector_affichage = vector.view(d[0], d[1])
    vector_affichage = vector_affichage.t()
    matrix = vector_affichage.numpy()
    return matrix