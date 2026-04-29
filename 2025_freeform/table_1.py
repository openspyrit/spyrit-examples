# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:53:41 2024

@author: ducros
""" 

import numpy as np

# Cramer-Rao gives a lower bound for the variance of an (unbiased?) estimator

n=5
I = np.eye(n,n)
#J = np.ones((n,n))
#K = J-I
K = 1 - I

K_inv = np.linalg.inv(K)
K_inv_2 = 1/(n-1)*(1 - (n-1)*I)

Cov = K_inv_2 @ K_inv_2
Cov_2 = I + (2-n)/(n-1)**2

#%% S matrix, check formula for S-1 @ S-T
import numpy as np
import spyrit.misc.walsh_hadamard as wh

N = 31
S_inv = wh.iwalsh_S_matrix(N)

SS1 = S_inv @ S_inv.T
SS2 = 4/(N+1)**2 * ((N+1)*np.eye(N) - 1)

print(SS2)

err = np.linalg.norm(SS1-SS2)
print(f'error: {err}')

#%% With subsampling
import torch 
from spyrit.core.torch import walsh_matrix
from  spyrit.misc.walsh_hadamard import walsh_S_matrix

m = 8              # number of measurements
ind = (0,1,3,4)#◘,6,9,11,16,17,20)   # indices to keep
n = len(ind)        # number of pixels

# Hadamard
H = walsh_matrix(m)

# Hadamard split
P = 0.5*(H + 1) # positive
N = 0.5*(1 - H) # negative
#N = N[1:,]      # remove first row. NB: this does not affect the pinv
F = np.concatenate((P,N)) # full

h = m-1
S = walsh_S_matrix(h)

# Hadamard
H_pinv = H.T/m
tr_H = np.trace(H_pinv@H_pinv.T)
print(f"Hadamard ({m}x{m}):", tr_H, 1.0)
tr_H = np.trace(H_pinv[ind,:]@H_pinv[ind,:].T)           
print(f"Hadamard ({m}x{n}):", tr_H, n/m)

#P = P[1:,:] # leads to the same as negative!
tr_P = np.trace(np.linalg.pinv(P.T@P)) 
print(f"Positive ({m}x{m}):", tr_P, (5*m-4)/m)
tr_P = np.trace(np.linalg.pinv(P[:,ind].T@P[:,ind]))           
print(f"Positive ({m}x{n}):", tr_P, (5*n-4)/m)

#N = N[1:,:] # leads to the same as with zeros !
tr_N = np.trace(np.linalg.pinv(N.T@N))   
print(f"Negative ({m}x{m}):", tr_N, 4*(m-1)**2/m**2)
tr_N = np.trace(np.linalg.pinv(N[:,ind].T@N[:,ind]))           
print(f"Negative ({m}x{n}):", tr_N, 4*(n-1)**2/m/n)

tr_F = np.trace(np.linalg.pinv(F.T@F))   
print(f"Split Hadamard ({2*m}x{m}):", tr_F, 2*m/(m+1))
tr_F = np.trace(np.linalg.pinv(F[:,ind].T@F[:,ind]))        
print(f"Split Hadamard ({2*m}x{n}):", tr_F, "?") #2*n/(m+1)

tr_S = np.trace(np.linalg.pinv(S.T@S))
print(f"S matrix ({h}x{h}):", tr_S, 4*h**2/(h+1)**2)
tr_S = np.trace(np.linalg.pinv(S[:,ind].T@S[:,ind]))
print(f"S matrix ({h}x{n}):", tr_S, 4*n**2/(h+1)/(n+1))