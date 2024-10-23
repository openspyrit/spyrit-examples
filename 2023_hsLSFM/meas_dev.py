# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:10:48 2023

@author: ducros
"""
import torch.nn as nn
import numpy as np
import torch
from spyrit.core.meas import LinearSplit

class LinearSplitPosNeg(LinearSplit):
# ==================================================================================
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) and :math:`x` is a 
    vectorized image.
    
    The matrix :math:`P` contains only positive values and is obtained by 
    splitting a measurement matrix :math:`H` such that 
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where 
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.
         
    The class is constructed from the :math:`M` by :math:`N` matrices 
    :math:`H_+` and :math:`H_-`, where :math:`N` represents the number of 
    pixels in the image and :math:`M` the number of measurements.
    
    Args:
        - :attr:`H_pos` (np.ndarray): positive component of the measurement 
        matrix :math:`H_+` with shape :math:`(M, N)`.
        
        - :attr:`H_neg` (np.ndarray): negative component of the measurement 
        matrix :math:`H_-` with shape :math:`(M, N)`.
        
    Example:
        >>> H_pos = np.array(np.random.random([400,1000]))
        >>> H_neg = np.array(np.random.random([400,1000]))
        >>> meas_op =  LinearSplitPosNeg(H_pos, H_neg)
        
    .. note::
        Same as :class:`~spyrit.core.meas.LinearSplit` with a different 
        constructor
        
    """

    def __init__(self, H_pos: np.ndarray, H_neg: np.ndarray, 
                         pinv = None, reg: float = 1e-15): 
        
        H = H_pos - H_neg
        
        super().__init__(H, pinv, reg)
                
        even_index = range(0,2*self.M,2);
        odd_index = range(1,2*self.M,2);
        
        P = np.zeros((2*self.M,self.N));
        P[even_index,:] = H_pos
        P[odd_index,:] = H_neg
        
        self.P = nn.Linear(self.N, 2*self.M, False) 
        self.P.weight.data = torch.from_numpy(P)
        self.P.weight.data = self.P.weight.data.float()
        self.P.weight.requires_grad=False
    


#%% Update documentation from here!!
from spyrit.misc.walsh_hadamard import walsh_matrix, walsh_torch
from spyrit.misc.sampling import Permutation_Matrix        

class Hadam1Split(LinearSplit):
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) with positive entries and 
    :math:`x` is an image.
    
    The class is relies on a matrix :math:`H` with 
    shape :math:`(M,N)` where :math:`N` represents the number of pixels in the 
    image and :math:`M \le N` the number of measurements. The matrix :math:`P` 
    is obtained by splitting the matrix :math:`H` such that 
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where 
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`. 
    
    The matrix :math:`H` is obtained by retaining the first :math:`M` rows of 
    a permuted Hadamard matrix :math:`GF`, where :math:`G` is a 
    permutation matrix with shape with shape :math:`(M,N)` and :math:`F` is a 
    "full" Hadamard matrix with shape :math:`(N,N)`. The computation of a
    Hadamard transform :math:`Fx` benefits a fast algorithm, as well as the
    computation of inverse Hadamard transforms.
    
    .. note::
        :math:`H = H_{+} - H_{-}`
    
    Args:
        - :attr:`M`: Number of measurements
        - :attr:`N`: Image width :math:`N`.
        - :attr:`Ord`: Order matrix with shape :math:`(N,N)` used to compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)` (see the :mod:`~spyrit.misc.sampling` submodule)
    
    .. note::
        The matrix H has shape :math:`(M,N)`.
        
    Example:
        >>> Ord = np.random.random([32,32])
        >>> meas_op = Hadam1Split(400, 32, Ord)
    """

    def __init__(self, M: int, N: int, Ord: np.ndarray=None):
        
        F =  walsh_matrix(N) # full matrix
        
        if Ord is not None:
            Perm = Permutation_Matrix(Ord)
            F = Perm@F # If Perm is not learnt, could be computed mush faster
                
        H = F[:M,:]
        
        super().__init__(H)
        
        if Ord is None:
            self.Perm = nn.Identity()
        else:
            self.Perm = nn.Linear(self.N, self.N, False)
            self.Perm.weight.data = torch.from_numpy(Perm.T)
            self.Perm.weight.data = self.Perm.weight.data.float()
            self.Perm.weight.requires_grad = False
        
    
    def inverse(self, x: torch.tensor) -> torch.tensor:
        r""" Inverse transform of Hadamard-domain images 
        :math:`x = H_{had}^{-1}G y` is a Hadamard matrix.
        
        Args:
            :math:`x`:  batch of images in the Hadamard domain
            
        Shape:
            :math:`x`: :math:`(*, N)` with :math:`N` the number of
            pixels in the image.
            
            Output: math:`(*, N)`
            
        Example:

            >>> y = torch.rand([85,32*32], dtype=torch.float)  
            >>> x = meas_op.inverse(y)
            >>> print('Inverse:', x.shape)
            Inverse: torch.Size([85, 1024])
        """
        # permutations
        x = self.Perm(x)
        # inverse of full transform
        x = 1/self.N*walsh_torch(x)       
        return x
    
    def pinv(self, x: torch.tensor) -> torch.tensor:
        r""" Pseudo inverse transform of incoming mesurement vectors :math:`x`
        
        Args:
            :attr:`x`:  batch of measurement vectors.
            
        Shape:
            x: :math:`(*, M)`
            
            Output: :math:`(*, N)`
            
        Example:
            >>> y = torch.rand([85,400], dtype=torch.float)  
            >>> x = meas_op.pinv(y)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        x = self.adjoint(x)/self.N
        return x