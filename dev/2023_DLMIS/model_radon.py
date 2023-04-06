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




class Linear(nn.Module):
    r""" 
        Computes linear measurements from incoming images: :math:`y = Hx`, 
        where :math:`H` is a linear operator (matrix) and :math:`x` is a 
        vectorized image.
        
        The class is constructed from a :math:`M` by :math:`N` matrix :math:`H`, 
        where :math:`N` represents the number of pixels in the image and 
        :math:`M` the number of measurements.
        
        Args:
            :attr:`H`: measurement matrix (linear operator) with shape :math:`(M, N)`.
            
            :attr:`pinv`: Option to have access to pseudo inverse solutions. 
            Defaults to `None` (the pseudo inverse is not initiliazed). 
            
            :attr:`reg` (optional): Regularization parameter (cutoff for small 
            singular values, see :mod:`numpy.linal.pinv`). Only relevant when 
            :attr:`pinv` is not `None`.


        Attributes:
              :attr:`H`: The learnable measurement matrix of shape 
              :math:`(M,N)` initialized as :math:`H`
             
              :attr:`H_adjoint`: The learnable adjoint measurement matrix 
              of shape :math:`(N,M)` initialized as :math:`H^\top`
              
              :attr:`H_pinv` (optional): The learnable adjoint measurement 
              matrix of shape :math:`(N,M)` initialized as :math:`H^\dagger`.
              Only relevant when :attr:`pinv` is not `None`.
        
        Example:
            >>> H = np.random.random([400, 1000])
            >>> meas_op = Linear(H)
            >>> print(meas_op)
            Linear(
              (H): Linear(in_features=1000, out_features=400, bias=False)
              (H_adjoint): Linear(in_features=400, out_features=1000, bias=False)
            )
            
        Example 2:
            >>> H = np.random.random([400, 1000])
            >>> meas_op = Linear(H, True)
            >>> print(meas_op)
            Linear(
              (H): Linear(in_features=1000, out_features=400, bias=False)
              (H_adjoint): Linear(in_features=400, out_features=1000, bias=False)
              (H_pinv): Linear(in_features=400, out_features=1000, bias=False)
            )
    """

    def __init__(self, H: np.ndarray, pinv = None, reg: float = 1e-15):  
        super().__init__()
        # instancier nn.linear
        self.M = H.shape[0]
        self.N = H.shape[1]
        self.H = nn.Linear(self.N, self.M, False) 
        self.H.weight.data = torch.from_numpy(H).float()
        # Data must be of type float (or double) rather than the default float64 when creating torch tensor
        self.H.weight.requires_grad = False

        # adjoint (Remove?)
        self.H_adjoint = nn.Linear(self.M, self.N, False)
        self.H_adjoint.weight.data = torch.from_numpy(H.transpose()).float()
        self.H_adjoint.weight.requires_grad = False
        
        if pinv is None:
            H_pinv = pinv
            print('Pseudo inverse will not instanciated')
            
        else: 
            H_pinv = np.linalg.pinv(H, rcond = reg)
            self.H_pinv = nn.Linear(self.M, self.N, False)
            self.H_pinv.weight.data = torch.from_numpy(H_pinv).float()
            self.H_pinv.weight.requires_grad = False
               
    def forward(self, x: torch.tensor) -> torch.tensor: 
        r""" Applies linear transform to incoming images: :math:`y = Hx`.

        Args:
            :math:`x`: Batch of vectorized (flatten) images.
            
        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N` 
            the total number of pixels in the image.
            
            Output: :math:`(*, M)` where * denotes the batch size and `M` 
            the number of measurements.
            
        Example:        
            >>> x = torch.rand([10,1000], dtype=torch.float)
            >>> y = meas_op(x)
            >>> print('forward:', y.shape)
            forward: torch.Size([10, 400])
            
        """
        # x.shape[b*c,N]
        x = self.H(x)    
        return x
    
    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r""" Applies adjoint transform to incoming measurements :math:`y = H^{T}x`

        Args:
            :math:`x`:  batch of measurement vectors.
            
        Shape:
            :math:`x`: :math:`(*, M)`
            
            Output: :math:`(*, N)`
            
        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)        
            >>> y = meas_op.adjoint(x)
            >>> print('adjoint:', y.shape)
            adjoint: torch.Size([10, 1000])
        """
        #Pmat.transpose()*f
        x = self.H_adjoint(x)        
        return x

    def get_H(self) -> torch.tensor:          
        r""" Returns the measurement matrix :math:`H`.
        
        Shape:
            Output: :math:`(M, N)`
        
        Example:     
            >>> H = meas_op.get_H()
            >>> print('get_mat:', H.shape)
            get_mat: torch.Size([400, 1000])
            
        """
        return self.H.weight.data
    
    def pinv(self, x: torch.tensor) -> torch.tensor:
        r""" Computer pseudo inverse solution :math:`y = H^\dagger x`

        Args:
            :math:`x`:  batch of measurement vectors.
            
        Shape:
            :math:`x`: :math:`(*, M)`
            
            Output: :math:`(*, N)`
            
        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)        
            >>> y = meas_op.pinv(x)
            >>> print('pinv:', y.shape)
            adjoint: torch.Size([10, 1000])
        """
        #Pmat.transpose()*f
        x = self.H_pinv(x)        
        return x

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

#######################################################################
# 1.2 Important functions for statistical completion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best
# Radon patterns for a given image size
#

def generateActivationAngles(nbAngles):
    listOfAngles = np.floor(np.linspace(0,180,nbAngles,endpoint=True))
    listOfAnglesInt = listOfAngles.astype(int)
    angles = np.zeros(181)
    c = 0
    for i in range(0,181):
        if listOfAnglesInt[c] == i:
            angles[i] = 1
            c = c+1

    return angles;

def separateMu(mu, angles, nbAngles):
    mu = np.reshape(mu,[181, 64])
    mu = np.transpose(mu)

    mu1 = np.zeros([64,nbAngles], dtype="float32")
    mu2 = np.zeros([64, 181-nbAngles], dtype="float32")
    cmu1 = 0
    cmu2 = 0
    for i in range(0,181):
        if angles[i]==1:
            mu1[:,cmu1] = mu[:,i]
            cmu1 = cmu1 + 1
        else:
            mu2[:,cmu2] = mu[:,i]
            cmu2 = cmu2 + 1

    #mu1 = np.resize(mu1, [nbAngles*64,1])
    #mu2 = np.resize(mu2, [(181-nbAngles) * 64, 1])

    mu1 = np.transpose(mu1)
    mu1 = np.resize(mu1, [nbAngles*64,1])
    mu2 = np.transpose(mu2)
    mu2 = np.resize(mu2, [(181-nbAngles) * 64, 1])


    return mu1, mu2



def separateSigma(sigma, angles, nbAngles):
    sigma1 = np.zeros([64 * nbAngles, 64 * nbAngles], dtype="float32")
    sigma21 = np.zeros([64*(181-nbAngles), 64 * nbAngles], dtype="float32")
    csigma1_i = 0
    csigma1_j = 0
    csigma21_i = 0
    csigma21_j = 0
    for i in range(0,181):
        if (angles[i] == 1):
            csigma1_j = 0
            for j in range(0, 181):
                if (angles[j] == 1):
                    sigma1[csigma1_i*64:csigma1_i*64+64,csigma1_j*64:csigma1_j*64+64] = sigma[i*64:i*64+64,j*64:j*64+64]
                    csigma1_j = csigma1_j + 1
            csigma1_i = csigma1_i + 1

        if (angles[i] == 0):
            csigma21_j = 0
            for j in range(0, 181):
                if (angles[j] == 1):
                    sigma21[csigma21_i*64:csigma21_i*64 + 64, csigma21_j*64:csigma21_j*64 + 64] = sigma[i * 64:i * 64 + 64, j * 64:j * 64 + 64]
                    csigma21_j = csigma21_j + 1
            csigma21_i = csigma21_i + 1

    return sigma1, sigma21

def computeUnknownData(mu1,mu2,sigma1,sigma21,angles,nbAngles):
    w = np.zeros([181*64, nbAngles*64], dtype="float32")
    b = np.zeros([181*64,1], dtype="float32")

    sigmaMix = np.dot(sigma21, lin.pinv(sigma1))
    #sigmaMix = np.dot(sigma21,np.transpose(sigma1))
    bMix = mu2 - np.dot(sigmaMix, mu1)
    #bMix = mu2

    eyeMix = np.eye(nbAngles*64)
    cm1 = 0
    cm2 = 0

    for i in range(0,181):
        if angles[i]==1:
            w[i*64:i*64+64,cm1*64:cm1*64+64] = eyeMix[cm1*64:cm1*64+64,cm1*64:cm1*64+64]
            cm1 = cm1+1
        if angles[i] == 0:
            w[i*64:i*64+64,:] = sigmaMix[cm2*64:cm2*64+64,:]
            b[i*64:i*64+64,:] = bMix[cm2*64:cm2*64+64,:]
            cm2 = cm2+1

    return w, b[:,0]