# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 16:03:50 2026

@author: ducros
"""

#%% Wuttig's codes lead to circulant symmetric matrices!
# Symmetric matrices are unitary (normal ?) matrices with real singular values 
# (or eigenvalues? or both?)
# Symmetric: S^T = S
# Circulant: Sf = s*f, where * is the cicular convolution

import numpy as np
import scipy as sp

N = 31

codes = {
 3: (1,1,0),
 7: (1,1,0,0,0,1,0),
 13: (1,1,0,1,0,0,0,0,0,1,0,0,0),
 21: (1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0),
 31: (1,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0),
 57: (1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0),
 73: (1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0),
 91:  (1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0)
}

s = codes[N]
S = sp.linalg.circulant(s).T        # the sequence goes into the first row
z = np.sum(codes[N])

_ , sigmas, _= sp.linalg.svd(S)
print("Wuttig, singular values:\n", sigmas)

lambdas, eigenvectors = np.linalg.eig(S)       # as S is normal, sigma = |lambda| 

S_inv_norm_F = np.sum(1/sigmas**2)
print("Wuttig, squared Frobenius norm (I):", S_inv_norm_F)


# Inverse matrix, numerical inversion
S_inv = sp.linalg.inv(S)
S_inv_norm_F = np.trace(S_inv @ S_inv.T)
print("Wuttig, squared Frobenius norm (II):", S_inv_norm_F)


Q0 = (N/S_inv_norm_F)**.5
print("Wuttig, Q0:", Q0)

# Inverse matrix, from convolution patterm
# Note: the convolution pattern is transposed.
# We keep the first entry, reverse from the end, i.e., [1 2 3 4] -> [1 4 3 2]
s_inv = np.insert(s[-1:0:-1], 0, 1)
s_inv = s_inv*(1/(N-1) + 1/z)  - 1/(N-1)
S_inv_b = sp.linalg.circulant(s_inv).T


# Inverse matrix from the full direct matrix
S_inv_c = S.T*(1/z + 1/(N-1)) - 1/(N-1)

np.linalg.norm(S_inv-S_inv_b)

s_inv_norm = s_inv@s_inv.T
print("Wuttig, squared Frobenius norm (III)", N*s_inv_norm) 


S_inv_norm_F_N = (z+1)/z**2
S_inv_norm_F_N_b =(N + (N-2)*(4*N-3)**.5)/2/(N-1)**2


print("Wuttig, noise amplification (I)", S_inv_norm_F_N) 
print("Wuttig, noise amplification (II)", S_inv_norm_F_N_b) 

#%% S-matrix

import spyrit.misc.walsh_hadamard as wh

N = 31
S = wh.walsh_S_matrix(N)
z = (N+1)/2

_ , s, _= sp.linalg.svd(S)
s_b = np.insert(np.ones(N-1)*(z/2)**.5, 0, z)

print("S-matrix, singular values (numerical):\n", s)
print("S-matrix, singular values (analytical):\n", s)


S_inv_norm_F = np.sum(1/s**2)
print("S-matrix inverse, squared Frobenius norm (I):", S_inv_norm_F)


S_inv = wh.iwalsh_S_matrix(N)
S_inv_norm_F = np.trace(S_inv @ S_inv.T)
print("S-matrix inverse, squared Frobenius norm (II)", S_inv_norm_F)

Q0 = (N/S_inv_norm_F)**.5
print("S-matrix inverse, Q0", Q0)

S_inv_norm_F_N_b = 4/N


#%%


SS2 = 4/(N+1)**2 * ((N+1)*np.eye(N) - 1)

print(SS2)

err = np.linalg.norm(SS1-SS2)
print(f'error: {err}')

#%%
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